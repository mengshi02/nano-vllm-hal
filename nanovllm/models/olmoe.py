import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import OlmoeConfig

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.parallel import (
    layer_range,
    get_tp_size,
    get_tp_rank,
    get_tp_group,
    get_ep_size,
    get_ep_rank,
    get_ep_group,
    is_data_parallel_ep,
)
from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ExpertColumnParallelLinear,
    ExpertRowParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.attention import Attention
from nanovllm.ops import get_moe_aggregate_fn
from nanovllm.platforms import current_platform


class OlmoeQKNorm(RMSNorm):
    """Normalize sharded Q/K over the full projected vector, as in HF OLMoE."""

    def __init__(self, local_size: int, eps: float):
        super().__init__(local_size, eps)
        self.tp_size = get_tp_size()
        self.tp_rank = get_tp_rank()
        self.tp_group = get_tp_group()
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return super().forward(x)
        x_fp32 = x.float()
        squared_sum = x_fp32.square().sum(dim=-1, keepdim=True)
        dist.all_reduce(squared_sum, group=self.tp_group)
        variance = squared_sum / (self.weight.numel() * self.tp_size)
        return (x_fp32 * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class OlmoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        clip_qkv: float | None = None,
        tp_fp32_matmul: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.clip_qkv = clip_qkv
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            fp32_matmul=tp_fp32_matmul,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # OLMoE normalizes the entire projected Q/K vectors before splitting heads.
        self.q_norm = OlmoeQKNorm(self.q_size, eps=rms_norm_eps)
        self.k_norm = OlmoeQKNorm(self.kv_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.clip_qkv is not None:
            q = q.clamp(-self.clip_qkv, self.clip_qkv)
            k = k.clamp(-self.clip_qkv, self.clip_qkv)
            v = v.clamp(-self.clip_qkv, self.clip_qkv)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

class OlmoeExperts(nn.ModuleDict):
    def __init__(self, config: OlmoeConfig, tp_fp32_matmul: bool = False):
        ep_size = get_ep_size()
        ep_rank = get_ep_rank()
        if config.num_experts % ep_size:
            raise ValueError("num_experts must be divisible by expert_parallel_size")
        experts_per_rank = config.num_experts // ep_size
        # Keep global expert IDs as module keys. Checkpoint names therefore
        # remain unchanged, while each rank materializes only its expert shard.
        local_expert_ids = list(range(
            ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank))
        super().__init__({
            str(expert_id): OlmoeExpert(config, tp_fp32_matmul)
            for expert_id in local_expert_ids
        })
        self.local_expert_ids = local_expert_ids
        self.aggregate_experts = get_moe_aggregate_fn(current_platform.backend_name)

    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(str(key))
        return super().__getitem__(key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.aggregate_experts(self, hidden_states, top_k_index, top_k_weights,
                                      self.local_expert_ids)


class OlmoeExpert(nn.Module):
    def __init__(self, config: OlmoeConfig, tp_fp32_matmul: bool = False):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported OLMoE expert activation: {config.hidden_act}")
        self.gate_proj = ExpertColumnParallelLinear(config.hidden_size,
                                                    config.intermediate_size)
        self.up_proj = ExpertColumnParallelLinear(config.hidden_size,
                                                   config.intermediate_size)
        self.down_proj = ExpertRowParallelLinear(
            config.intermediate_size, config.hidden_size, fp32_matmul=tp_fp32_matmul
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class OlmoeTopKRouter(nn.Module):
    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class OlmoeMLP(nn.Module):
    def __init__(
        self,
        config: OlmoeConfig,
        tp_fp32_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.gate = OlmoeTopKRouter(config)
        self.experts = OlmoeExperts(config, tp_fp32_matmul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The engine passes a flat [total_tokens, hidden_size] tensor.
        x = x.reshape(-1, x.shape[-1])
        ep_size = get_ep_size()
        if not is_data_parallel_ep():
            _, top_k_weights, top_k_index = self.gate(x)
            output = self.experts(x, top_k_index, top_k_weights)
            tp_size = get_tp_size()
            if tp_size > 1:
                dist.all_reduce(output, group=get_tp_group())
            return output

        # Legacy TP=1/DP=EP layout: DP attention gives every rank only its own
        # tokens.  Gather activations so every EP rank can compute the experts
        # it owns, then reduce-scatter partial sums back to each token owner.
        ep_group = get_ep_group()
        ep_rank = get_ep_rank()
        local_num_tokens = x.shape[0]
        # Some NCCL builds in this environment corrupt tiny int64 collectives.
        # Token counts are small enough for exact float32 representation.
        local_count = torch.tensor([local_num_tokens], dtype=torch.float32,
                                   device=x.device)
        all_counts = [torch.empty_like(local_count) for _ in range(ep_size)]
        dist.all_gather(all_counts, local_count, group=ep_group)
        max_num_tokens = int(torch.stack(all_counts).max().item())
        if max_num_tokens == 0:
            return x
        padded = x.new_zeros((max_num_tokens, x.shape[-1]))
        if local_num_tokens:
            padded[:local_num_tokens].copy_(x)
        gathered = [torch.empty_like(padded) for _ in range(ep_size)]
        dist.all_gather(gathered, padded.contiguous(), group=ep_group)
        all_hidden_states = torch.cat(gathered, dim=0)
        _, top_k_weights, top_k_index = self.gate(all_hidden_states)
        all_outputs = self.experts(all_hidden_states, top_k_index, top_k_weights)
        output_shards = [shard.contiguous()
                         for shard in torch.chunk(all_outputs, ep_size, dim=0)]
        local_output = torch.empty_like(output_shards[ep_rank])
        dist.reduce_scatter(local_output, output_shards, group=ep_group)
        return local_output[:local_num_tokens]


class OlmoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: OlmoeConfig,
        tp_fp32_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = OlmoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            clip_qkv=getattr(config, "clip_qkv", None),
            tp_fp32_matmul=tp_fp32_matmul,
        )
        self.mlp = OlmoeMLP(config, tp_fp32_matmul)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class OlmoeModel(nn.Module):
    def __init__(self,
        config: OlmoeConfig,
        pp_rank: int = 0,
        pp_size: int = 1,
        tp_fp32_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.is_first = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1
        self.local_layers = layer_range(config.num_hidden_layers, pp_size, pp_rank)
        if self.is_first or (self.is_last and config.tie_word_embeddings):
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            OlmoeDecoderLayer(config, tp_fp32_matmul) if i in self.local_layers else nn.Identity()
            for i in range(config.num_hidden_layers)
        ])
        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, hidden_states=None, residual=None):
        if self.is_first:
            hidden_states = self.embed_tokens(input_ids)
        for i in self.local_layers:
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)
        if self.is_last:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        return hidden_states, residual


class OlmoeForCausalLM(nn.Module):
    # Dynamic expert selection uses tensor-to-host synchronization.
    supports_cuda_graph = False

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    def __init__(self, 
        config: OlmoeConfig,
        pp_rank: int = 0,
        pp_size: int = 1,
        tp_fp32_matmul: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = OlmoeModel(config, pp_rank, pp_size, tp_fp32_matmul)
        if self.model.is_last:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.tie_word_embeddings = config.tie_word_embeddings
        if self.tie_word_embeddings and self.model.is_last:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, residual)

    def expected_weight_names(self) -> set[str]:
        expected = set()
        for name, _ in self.named_parameters():
            if name == "lm_head.weight" and self.tie_word_embeddings:
                continue
            unpacked_names = [
                name.replace(f".{packed}.", f".{unpacked}.", 1)
                for unpacked, (packed, _) in self.packed_modules_mapping.items()
                if f".{packed}." in name
            ]
            expected.update(unpacked_names or [name])
        return expected

    def tied_weight_groups(self) -> tuple[set[str], ...]:
        if self.tie_word_embeddings:
            return ({"model.embed_tokens.weight", "lm_head.weight"},)
        return ()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
