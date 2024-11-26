import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from libs.srenv import SREnv
from agents.rlagent import DQNAgent, ReplayBuffer


def train_rl_model(
    library,
    data,
    target,
    max_depth=10,
    embedding_dim=128,
    hidden_dim=256,
    num_batches=1000,
    num_episodes_per_batch=10,
    batch_quantile=0.1,
    batch_size=500,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.25,
    epsilon_decay=0.9995,
    target_update=100,
    memory_capacity=1000000,
    batch_eval=10,
    lr=1e-4,
):
    # Initialize environment
    env = SREnv(library=library, data=data, target=target, max_depth=max_depth)

    # Define vocabulary
    vocab = list(library.keys()) + ['PAD']
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(vocab)}
    index_to_symbol = {idx: symbol for symbol, idx in symbol_to_index.items()}
    vocab_size = len(vocab)

    def encode_state(state):
        # Convert symbols to indices
        state_indices = [symbol_to_index[symbol] for symbol in state]
        # Pad sequence
        if len(state_indices) < max_depth:
            state_indices += [symbol_to_index['PAD']] * (max_depth - len(state_indices))
        else:
            state_indices = state_indices[:max_depth]
        return torch.tensor(state_indices, dtype=torch.long)

    action_symbols = list(library.keys())
    action_size = len(action_symbols)
    symbol_to_action_idx = {symbol: idx for idx, symbol in enumerate(action_symbols)}

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize agent and target network
    agent = DQNAgent(vocab_size, embedding_dim, hidden_dim, action_size)
    target_agent = DQNAgent(vocab_size, embedding_dim, hidden_dim, action_size)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    agent.train()

    optimizer = optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(memory_capacity)

    epsilon = epsilon_start

    # Training loop
    for batch in range(num_batches):
        episodes = []
        for episode in range(num_episodes_per_batch):
            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols)  # Shape: (seq_length,)
            done = False
            total_reward = 0
            transitions = []
            i = 0

            while not done and i < max_depth:
                # Select action
                action_idx = agent.act(state_encoded, epsilon)
                action_symbol = action_symbols[action_idx]

                try:
                    next_state_symbols, reward, done = env.step(action_symbol)
                except ValueError:
                    reward = 0
                    done = True
                    next_state_symbols = state_symbols  # Remain in the same state

                if np.isnan(reward):
                    print('Debug')

                next_state_encoded = encode_state(next_state_symbols)
                total_reward += float(reward)

                # Store transition
                transitions.append((
                    state_encoded,
                    action_idx,
                    reward,
                    next_state_encoded,
                    done
                ))

                state_encoded = next_state_encoded
                state_symbols = next_state_symbols

                i += 1

            if done:
                print(f"Batch {batch}.{episode} completed, Total Reward: {total_reward}")
            else:
                total_reward = -1
                print(f"Batch {batch}.{episode} failed, Total Reward: {total_reward}")

            transitions = [
                (
                    t[0],
                    t[1],
                    total_reward,
                    t[3],
                    t[4]
                )
                for t in transitions
            ]

            episodes.append((transitions, total_reward))

        total_rewards = [episode[1] for episode in episodes]
        threshold = np.quantile(total_rewards, 1 - batch_quantile)
        top_episodes = [episode[0] for episode in episodes if episode[1] >= threshold]

        # Store transitions from top episodes in the replay buffer
        for episode_transitions in top_episodes:
            for transition in episode_transitions:
                memory.push(*transition)

        # Experience replay
        if len(memory) >= batch_size:
            # Sample from memory
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = memory.sample(batch_size)

            try:
                # Compute current Q-values
                q_values = agent(states_batch)
                q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_agent(next_states_batch).max(dim=1)[0]
                    target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

                # Compute loss
                loss = criterion(q_values, target_q_values)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f'Training failed due to {e}. Skipping this iteration...')

        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        # Update target network
        if batch % target_update == 0:
            target_agent.load_state_dict(agent.state_dict())

        # Evaluation
        if batch % batch_eval == 0:
            print('---------------------')
            print('Evaluating...')
            print('---------------------')
            agent.eval()
            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols)
            done = False
            total_reward = 0
            expression_actions = []
            i = 0

            while not done and i < max_depth:
                with torch.no_grad():
                    q_values = agent(state_encoded.unsqueeze(0))  # Add batch dimension
                    action_idx = torch.argmax(q_values).item()
                action_symbol = action_symbols[action_idx]
                expression_actions.append(action_symbol)
                try:
                    next_state_symbols, reward, done = env.step(action_symbol)
                except ValueError:
                    reward = -1.0
                    done = True
                    next_state_symbols = state_symbols
                next_state_encoded = encode_state(next_state_symbols)
                total_reward += float(reward)
                state_encoded = next_state_encoded
                state_symbols = next_state_symbols
                i += 1

            agent.train()  # Set back to training mode

            if done and round(total_reward, 2) == 1:
                print('Found expression! Stopping early...')
                break

    # Save the model
    torch.save(agent.state_dict(), "agent.pth")


if __name__ == "__main__":
    # Define the library of operators and symbols
    library = {
        '+': 2,
        '-': 2,
        '*': 2,
        '/': 2,
        'sin': 1,
        'cos': 1,
        'C': 0,  # Placeholder for constants
    }

    n_samples = 1000
    n_vars = 1

    # Add variables to the library
    for i in range(n_vars):
        var_name = f'X{i}'
        library[var_name] = 0

    diff = [torch.zeros(n_samples) + i for i in range(n_vars)]
    data = torch.randn([n_vars, n_samples]) + torch.stack(diff)  # Shape: (n_vars, n_samples)
    target = 2 * data[0] + 10

    # Train the RL model
    train_rl_model(library, data, target)
