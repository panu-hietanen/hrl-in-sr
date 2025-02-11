import numpy as np
import torch

from agents.rlagent import RolloutBuffer

def risk_seeking_filter(
    memory: list[RolloutBuffer],
    risk_quantile: float = 0.2,
):
    final_rewards: list = [
        episode.sample()[4][-1].item() for episode in memory
    ]

    threshold = np.quantile(final_rewards, 1 - risk_quantile)
    indexes = np.where(final_rewards >= threshold)
    return [memory[i] for i in indexes[0].tolist()]

def encode_state(state, symbol_to_index: dict[str, int], max_seq_length: int):
    # Convert symbols to indices
    state_indices = [symbol_to_index[symbol] for symbol in state]
    # Pad sequence
    if len(state_indices) < max_seq_length:
        state_indices += [symbol_to_index['PAD']] * (max_seq_length - len(state_indices))
    else:
        state_indices = state_indices[:max_seq_length]
    return torch.tensor(state_indices, dtype=torch.long)

def compute_r_and_a(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = []
    running_return = 0

    for i, (r, d) in enumerate(zip(reversed(rewards), reversed(dones))):
        running_return = r + gamma * running_return * (1-d.item())
        returns.insert(0, running_return)

    returns = torch.tensor(returns, dtype=torch.float32).detach()
    advantages = (returns - values).detach()
    return returns, advantages

def combine_rollout_buffers(
        buffer_list: list[RolloutBuffer]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given a list of RolloutBuffer objects, combine their stored transitions into
    single tensors for states, actions, log_probs, values, rewards, and dones.
    """
    all_states = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_rewards = []
    all_dones = []
    
    for buf in buffer_list:
        all_states.append(torch.stack(buf.states))  # shape: (num_steps, ...)
        all_actions.append(torch.tensor(buf.actions, dtype=torch.long))  # shape: (num_steps,)
        all_log_probs.append(torch.stack(buf.log_probs))  # shape: (num_steps,)
        all_values.append(torch.stack(buf.values))  # shape: (num_steps,)
        all_rewards.append(torch.tensor(buf.rewards, dtype=torch.float32))  # shape: (num_steps,)
        all_dones.append(torch.tensor(buf.dones, dtype=torch.bool))  # shape: (num_steps,)
    
    states = torch.cat(all_states, dim=0)
    actions = torch.cat(all_actions, dim=0)
    log_probs = torch.cat(all_log_probs, dim=0)
    values = torch.cat(all_values, dim=0)
    rewards = torch.cat(all_rewards, dim=0)
    dones = torch.cat(all_dones, dim=0)
    
    return states, actions, log_probs, values, rewards, dones


