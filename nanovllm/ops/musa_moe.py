import torch


def aggregate_experts(experts, hidden_states, top_k_index, top_k_weights,
                      expert_ids=None):
    slots = hidden_states.new_zeros((hidden_states.shape[0], top_k_index.shape[1],
                                    hidden_states.shape[1]))
    selected_experts = (expert_ids if expert_ids is not None
                         else torch.unique(top_k_index).tolist())
    for expert_idx in selected_experts:
        token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_output = experts[expert_idx](hidden_states[token_idx])
        weighted = expert_output * top_k_weights[token_idx, top_k_pos].unsqueeze(-1)
        slots[token_idx, top_k_pos] = weighted.to(slots.dtype)
    return slots.sum(dim=1)
