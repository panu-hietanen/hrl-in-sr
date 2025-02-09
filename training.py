import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy

from libs.srenv import SREnv
from agents.rlagent import PPOAgent, RolloutBuffer
from libs.utils import risk_seeking_filter, encode_state, compute_r_and_a, combine_rollout_buffers

import matplotlib.pyplot as plt
from datetime import datetime

def ppo_update(
    agent: PPOAgent,
    old_agent: PPOAgent,
    memory: list[RolloutBuffer],
    data_input: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_epsilon: float=0.2,
    value_coef: float=0.5,
    entropy_coef: float=0.01,
    gamma: float=0.99,
    n_epochs: int=4,
    batch_size: int=64,
    risk_quantile: float = 0.2
):
    (
        states,
        actions,
        log_probs,
        values,
        rewards,
        dones
    ) = combine_rollout_buffers(memory)

    dataset_size = len(states)
    indices = torch.arange(dataset_size)

    for epoch in range(n_epochs):
        # Shuffle data
        perm = indices[torch.randperm(dataset_size)]
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_indices = perm[start:end]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_values = values[batch_indices]
            batch_rewards = rewards[batch_indices]
            batch_dones = dones[batch_indices]

            batch_returns, batch_advantages = compute_r_and_a(batch_values, batch_rewards, batch_dones, gamma)
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            # Risk seeking policy gradient
            threshold = torch.quantile(batch_returns, 1 - risk_quantile)

            risk_weights = torch.where(batch_returns >= threshold, torch.tensor(risk_factor), torch.tensor(1.0))
            weighted_advantages = batch_advantages * risk_weights

            with torch.no_grad():
                old_logits, _ = old_agent.forward(data_input, batch_states)
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(batch_actions)

            new_logits, new_values = agent.forward(data_input, batch_states)
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(batch_actions)
            entropy = new_dist.entropy().mean()

            # Compute the probability ratio
            ratio = (new_log_probs - old_log_probs).exp()

            # Clipped surrogate objective
            unclipped = ratio * weighted_advantages
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * weighted_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss (mean-squared error between new value predictions and returns)
            value_loss = F.mse_loss(new_values, batch_returns)

            # Total loss (policy loss + weighted value loss - entropy bonus)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_rl_model(
    agent: PPOAgent,
    env: SREnv,
    action_symbols: list[str],
    symbol_to_index: dict[str, int],
    max_seq_length: int,
    data_input: torch.Tensor,
    num_iterations: int=1000,
    num_episodes_per_iteration: int=10,
    gamma: float=0.99,
    clip_epsilon: float=0.2,
    value_coef: float=0.5,
    entropy_coef: float=0.01,
    lr: float=1e-4,
    batch_size: int=64,
    n_epochs: int=4,
    it_eval: int=100,
    risk_quantile: float=0.2,
    logging: bool=False
):
    # Initialize optimizer, loss function, and replay buffer
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    memory = []
    episode_transitions = RolloutBuffer()

    best_reward: float = -float('inf')
    best_expression: list[str] = []

    history: list[tuple[int, float]] = []

    for iteration in range(num_iterations):
        memory = []
        episode = 0

        while episode < num_episodes_per_iteration:
            episode_transitions.clear()

            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
            done = False
            total_reward_ep = 0
            i = 0

            while not done and i < max_seq_length:
                mask = torch.ones(len(action_symbols))
                # mask[[4, 5]] = 0
                # Select action
                action_idx, log_prob, value = agent.act(data_input, state_encoded, mask)
                action_symbol = action_symbols[action_idx]

                try:
                    next_state_symbols, reward, done = env.step(action_symbol)
                except ValueError as e:
                    print(f'Error {e}. Ending episode...')
                    reward = 0
                    done = True
                    next_state_symbols = state_symbols  # Remain in the same state

                next_state_encoded = encode_state(next_state_symbols, symbol_to_index, max_seq_length)
                total_reward_ep += reward

                # Store transition
                episode_transitions.store(
                    state_encoded,
                    action_idx,
                    log_prob,
                    value,
                    reward,
                    done
                )

                state_encoded = next_state_encoded
                state_symbols = next_state_symbols

                i += 1

            memory.append(episode_transitions)
            episode += 1

        old_agent = deepcopy(agent)
        old_agent.eval()

        memory = risk_seeking_filter(memory, risk_quantile)

        ppo_update(
            agent,
            old_agent,
            memory,
            data_input,
            optimizer,
            clip_epsilon,
            value_coef,
            entropy_coef,
            gamma,
            n_epochs,
            batch_size,
        )

        # Evaluation
        if iteration % it_eval == 0:
            print('Evaluating...')
            expression, r = evaluate_agent(
                agent, 
                env, 
                action_symbols, 
                symbol_to_index, 
                max_seq_length, 
                data_input, 
                0
            )

            print(f"Batch {iteration} completed, Greedy Reward: {r}")

            if r > best_reward:
                best_reward = r
                best_expression = expression

            if logging:
                history.append((iteration, reward))

            if round(float(r), 3) == 1:
                print(f'Found expression! Stopping early...')
                return best_expression, best_reward, history
    return best_expression, best_reward, history

def evaluate_agent(
    agent: PPOAgent,
    env: SREnv,
    action_symbols: list[str],
    symbol_to_index: dict[str, int],
    max_seq_length: int,
    data_input: torch.Tensor,
    max_retries: int=100
):
    agent.eval()
    state_symbols = env.reset()
    state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
    done = False
    total_reward = 0
    expression_actions = []
    r = 0
    i = 0

    while not done:
        with torch.no_grad():
            logits, _value = agent.forward(data_input.unsqueeze(0), state_encoded.unsqueeze(0))
            action_idx = torch.argmax(logits).item()
        action_symbol = action_symbols[action_idx]
        expression_actions.append(action_symbol)
        
        try:
            next_state_symbols, reward, done = env.step(action_symbol)
        except ValueError as e:
            print(f'Error {e}. Exiting...')
            reward = -1.0
            done = True
            next_state_symbols = state_symbols

        next_state_encoded = encode_state(next_state_symbols, symbol_to_index, max_seq_length)
        total_reward += reward
        state_encoded = next_state_encoded
        state_symbols = next_state_symbols

        if i == max_seq_length and not done:
            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
            total_reward = 0
            expression_actions = []
            i = 0
            r += 1
            if r > max_retries:
                break
            print('restarting...')
        else:
            i += 1

    # Replace constant placeholders with actual values
    n_const = env.expression.n_constants
    const_count = 0
    for idx, token in enumerate(expression_actions):
        if const_count == n_const:
            break
        if token == 'C':
            const_val = env.expression.optimized_constants[const_count].item()
            expression_actions[idx] = str(round(const_val, 3))
            const_count += 1

    constructed_expression = ' '.join(expression_actions)
    return constructed_expression, total_reward

def main() -> None:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the library of operators and symbols
    library = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        # '^': 2,
        'sin': 1,
        'cos': 1,
        'C': 0,  # Placeholder for constants
    }

    # Create data and target tensors
    n_samples = 1000
    n_vars = 1

    for i in range(n_vars):
        var_name = f'X{i}'
        library[var_name] = 0

    diff = [torch.zeros(n_samples) + i for i in range(n_vars)]
    data = torch.randn([n_vars, n_samples]) + torch.stack(diff)  # Shape: (n_vars, n_samples)
    target = 2 * data[0] + 3

    # Precompute data input
    data_flat = data.view(-1)
    data_input = (data_flat - data_flat.mean()) / (data_flat.std() + 1e-8)
    data_input_dim = data_input.shape[0]

    # Maximum sequence length
    max_seq_length = 10

    # Initialize the environment
    env = SREnv(library=library, data=data, target=target, max_length=max_seq_length)

    # Define vocabulary
    vocab = list(library.keys()) + ['PAD']
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(vocab)}
    vocab_size = len(vocab)


    action_symbols = list(library.keys())
    action_size = len(action_symbols)

    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    num_iterations = 1000
    num_episodes_per_iteration = 100
    batch_size = 500
    gamma = 0.9
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    n_epochs = 4
    it_eval = 10
    lr = 1e-2
    logging = True

    # Initialize agent and target agent
    agent = PPOAgent(data_input_dim, vocab_size, embedding_dim, hidden_dim, action_size, max_seq_length)
    agent.train()

    # Train the RL model
    expression, reward, history = train_rl_model(
        agent=agent,
        env=env,
        action_symbols=action_symbols,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
        data_input=data_input,
        num_iterations=num_iterations,
        num_episodes_per_iteration=num_episodes_per_iteration,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        lr=lr,
        batch_size=batch_size,
        n_epochs=n_epochs,
        it_eval=it_eval,
        logging=logging
    )

    # Evaluate the agent
    constructed_expression, total_reward = evaluate_agent(
        agent=agent,
        env=env,
        action_symbols=action_symbols,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
        data_input=data_input,
    )

    print(f"Best Training Expression: '{expression}', reward = {reward}")
    print('---------------------------')
    print(f"Final Testing Expression: {constructed_expression}")
    print(f"Reward: {total_reward}")

    if logging:
        plt.plot(*zip(*history))
        plt.xlabel("Batch no.")
        plt.ylabel("Reward")
        plt.title("Greedy Reward Over Time")
        plt.savefig(f'plots/Training_history_{datetime.now()}.png')

if __name__ == "__main__":
    main()