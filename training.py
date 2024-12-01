import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from libs.srenv import SREnv
from agents.rlagent import DQNAgent, ReplayBuffer

import matplotlib.pyplot as plt
from datetime import datetime

def encode_state(state, symbol_to_index: dict[str, int], max_seq_length: int):
    # Convert symbols to indices
    state_indices = [symbol_to_index[symbol] for symbol in state]
    # Pad sequence
    if len(state_indices) < max_seq_length:
        state_indices += [symbol_to_index['PAD']] * (max_seq_length - len(state_indices))
    else:
        state_indices = state_indices[:max_seq_length]
    return torch.tensor(state_indices, dtype=torch.long)


def train_rl_model(
    agent: DQNAgent,
    target_agent: DQNAgent,
    env: SREnv,
    action_symbols: list[str],
    symbol_to_index: dict[str, int],
    max_seq_length: int,
    data_input: torch.Tensor,
    num_episodes: int=10,
    batch_size: int=250,
    gamma: float=0.99,
    epsilon_start: float=1.0,
    epsilon_end: float=0.25,
    epsilon_decay: float=0.9995,
    target_update: int=None,
    memory_capacity: int=None,
    ep_eval: int=10,
    lr: float=1e-4,
    logging: bool=False
):
    # Initialize optimizer, loss function, and replay buffer
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if memory_capacity is None:
        memory_capacity = max_seq_length * num_episodes
    memory = ReplayBuffer(memory_capacity)

    if target_update is None:
        target_update = num_episodes // 10

    epsilon = epsilon_start

    best_reward: float = 0.0
    best_expression: list[str] = []

    history: list[tuple[int, float]] = []

    for episode in range(num_episodes):
        state_symbols = env.reset()
        state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
        done = False
        total_reward = 0
        i = 0

        while not done and i < max_seq_length:
            mask = env.get_action_mask()
            # mask[[4, 5]] = 0
            # Select action
            action_idx = agent.act(data_input, state_encoded, epsilon, mask)
            action_symbol = action_symbols[action_idx]

            try:
                next_state_symbols, reward, done = env.step(action_symbol)
            except ValueError as e:
                reward = 0
                done = True
                next_state_symbols = state_symbols  # Remain in the same state

            next_state_encoded = encode_state(next_state_symbols, symbol_to_index, max_seq_length)
            total_reward += reward

            # Store transition
            memory.push(
                state_encoded,
                action_idx,
                reward,
                next_state_encoded,
                done
            )

            state_encoded = next_state_encoded
            state_symbols = next_state_symbols

            # Experience replay
            replay(agent, target_agent, memory, optimizer, criterion, batch_size, gamma, data_input)

            i += 1

        # Decay epsilon
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        # Update target network
        if episode % target_update == 0:
            target_agent.load_state_dict(agent.state_dict())

        # Evaluation
        if episode % ep_eval == 0:
            print('Evaluating...')
            expression, r = evaluate_agent(agent, env, action_symbols, symbol_to_index, max_seq_length, data_input, 0)

            print(f"Batch {episode} completed, Greedy Reward: {r}")

            if r > best_reward:
                best_reward = r
                best_expression = expression

            if logging:
                history.append((episode, r))

            if round(float(r), 3) == 1:
                print(f'Found expression! Stopping early...')
                return best_expression, best_reward, history
    return best_expression, best_reward, history

def replay(agent, target_agent, memory, optimizer, criterion, batch_size, gamma, data_input):
    """
    Perform experience replay for the given agent using the replay buffer.
    """
    if len(memory) >= batch_size:
        # Sample a batch of transitions from memory
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = memory.sample(batch_size)
        
        try:
            # Compute current Q-values
            q_values = agent(data_input, states_batch)
            q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = target_agent(data_input, next_states_batch).max(dim=1)[0]
                target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)

            # Compute loss
            loss = criterion(q_values, target_q_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

            optimizer.step()
        except Exception as e:
            print(f"Replay failed due to {e}. Skipping this replay step.")

def evaluate_agent(
    agent: DQNAgent,
    env: SREnv,
    action_symbols: list[str],
    symbol_to_index: dict[str, int],
    max_seq_length: int,
    data_input: torch.Tensor,
    max_retries: int=20
):
    agent.eval()
    state_symbols = env.reset()
    state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
    done = False
    expression_actions = []
    r = 0
    i = 0

    while not done:
        with torch.no_grad():
            q_values = agent(data_input.unsqueeze(0), state_encoded.unsqueeze(0))
            action_idx = torch.argmax(q_values).item()
        action_symbol = action_symbols[action_idx]
        expression_actions.append(action_symbol)
        
        try:
            next_state_symbols, _, done = env.step(action_symbol)
        except ValueError as e:
            print(f'Error {e}. Exiting...')
            done = True
            next_state_symbols = state_symbols

        next_state_encoded = encode_state(next_state_symbols, symbol_to_index, max_seq_length)
        state_encoded = next_state_encoded
        state_symbols = next_state_symbols

        if i == max_seq_length and not done:
            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
            expression_actions = []
            i = 0
            r += 1
            if r > max_retries:
                break
            print('restarting...')
        else:
            i += 1

    if not done:
        return 'No expression found', 0.0

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
    return constructed_expression, env.get_reward()

if __name__ == "__main__":
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
    n_vars = 2

    for i in range(n_vars):
        var_name = f'X{i}'
        library[var_name] = 0

    diff = [torch.zeros(n_samples) + i for i in range(n_vars)]
    data = torch.randn([n_vars, n_samples]) + torch.stack(diff)  # Shape: (n_vars, n_samples)
    target = 2 * np.cos(data[0])

    # Precompute data input
    data_flat = data.view(-1)
    target_flat = target.view(-1)
    # data_input = torch.cat([data_flat, target_flat], dim=0)
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
    num_episodes = 1000
    batch_quantile = 0.1
    batch_size = 500
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.3
    epsilon_decay = 0.995
    target_update = 10
    memory_capacity = max_seq_length * num_episodes
    ep_eval = 10
    lr = 1e-4
    logging = False

    # Initialize agent and target agent
    agent = DQNAgent(data_input_dim, vocab_size, embedding_dim, hidden_dim, action_size, max_seq_length)
    target_agent = DQNAgent(data_input_dim, vocab_size, embedding_dim, hidden_dim, action_size, max_seq_length)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    agent.train()

    # Train the RL model
    expression, reward, history = train_rl_model(
        agent=agent,
        target_agent=target_agent,
        env=env,
        action_symbols=action_symbols,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
        data_input=data_input,
        num_episodes=num_episodes,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update=target_update,
        memory_capacity=memory_capacity,
        ep_eval=ep_eval,
        lr=lr,
        logging=logging,
    )

    # Evaluate the agent
    constructed_expression, total_reward = evaluate_agent(
        agent=agent,
        env=env,
        action_symbols=action_symbols,
        data_input=data_input,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
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