import torch


def aggregate_experts(experts, hidden_states, top_k_index, top_k_weights,
                     expert_ids=None):
    if expert_ids is None:
        selected_experts = torch.unique(top_k_index).tolist()
    else:
        selected = set(torch.unique(top_k_index).tolist())
        selected_experts = [expert for expert in expert_ids if expert in selected]
    output = torch.zeros_like(hidden_states)
    if not selected_experts:
        return output
    local_ids = torch.tensor(selected_experts, device=top_k_index.device)
    for local_id, expert_idx in zip(local_ids.tolist(), selected_experts):
        token_indices, selection_indices = torch.where(top_k_index == local_id)
        if token_indices.numel() == 0:
            continue
        expert_output = experts[expert_idx](hidden_states[token_indices])
        token_weights = top_k_weights[token_indices, selection_indices].unsqueeze(-1)
        output.index_add_(0, token_indices, expert_output * token_weights)
    return output
