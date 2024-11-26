import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from libs.srenv import SREnv
from agents.rlagent import DQNAgent, ReplayBuffer

def encode_state(state, symbol_to_index, max_seq_length):
    # Convert symbols to indices
    state_indices = [symbol_to_index[symbol] for symbol in state]
    # Pad sequence
    if len(state_indices) < max_seq_length:
        state_indices += [symbol_to_index['PAD']] * (max_seq_length - len(state_indices))
    else:
        state_indices = state_indices[:max_seq_length]
    return torch.tensor(state_indices, dtype=torch.long)

def train_rl_model(
    agent,
    target_agent,
    env,
    action_symbols,
    symbol_to_index,
    max_seq_length,
    num_batches=1000,
    num_episodes_per_batch=10,
    batch_quantile=0.1,
    batch_size=500,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.25,
    epsilon_decay=0.9995,
    target_update=10,
    memory_capacity=None,
    batch_eval=10,
    lr=1e-4,
):
    # Initialize optimizer, loss function, and replay buffer
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if memory_capacity is None:
        memory_capacity = max_seq_length * num_episodes_per_batch * num_batches
    memory = ReplayBuffer(memory_capacity)

    if target_update is None:
        target_update = num_batches // 10

    epsilon = epsilon_start

    for batch in range(num_batches):
        episodes = []
        for episode in range(num_episodes_per_batch):
            state_symbols = env.reset()
            state_encoded = encode_state(state_symbols, symbol_to_index, max_seq_length)
            done = False
            total_reward = 0
            transitions = []
            i = 0

            while not done and i < max_seq_length:
                # Select action
                action_idx = agent.act(state_encoded, epsilon)
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

            if not done:
                total_reward = -1

            # Assign total reward to all transitions (since intermediate rewards are zero)
            transitions = [
                (
                    t[0],  # state_encoded
                    t[1],  # action_idx
                    total_reward,
                    t[3],  # next_state_encoded
                    t[4]   # done
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
            _, r = evaluate_agent(agent, env, action_symbols, symbol_to_index, max_seq_length, 0)

            print(f"Batch {batch} completed, Greedy Reward: {r}")

            if round(float(r), 2) == 1:
                print(f'Found expression! Stopping early...')
                return

def evaluate_agent(
    agent,
    env,
    action_symbols,
    symbol_to_index,
    max_seq_length,
    max_retries=10
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
            q_values = agent(state_encoded.unsqueeze(0))
            action_idx = torch.argmax(q_values).item()
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

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Create data and target tensors
    n_samples = 1000
    n_vars = 1

    for i in range(n_vars):
        var_name = f'X{i}'
        library[var_name] = 0

    diff = [torch.zeros(n_samples) + i for i in range(n_vars)]
    data = torch.randn([n_vars, n_samples]) + torch.stack(diff)  # Shape: (n_vars, n_samples)
    target = 2 * data[0] + 10

    # Initialize the environment
    max_depth = 10
    env = SREnv(library=library, data=data, target=target, max_depth=max_depth)

    # Define vocabulary
    vocab = list(library.keys()) + ['PAD']
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(vocab)}
    vocab_size = len(vocab)

    # Maximum sequence length
    max_seq_length = max_depth

    action_symbols = list(library.keys())
    action_size = len(action_symbols)

    # Hyperparameters
    embedding_dim = 128
    hidden_dim = 256
    num_batches = 1000
    num_episodes_per_batch = 10
    batch_quantile = 0.1
    batch_size = 500
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.25
    epsilon_decay = 0.9995
    target_update = 10
    memory_capacity = max_seq_length * num_episodes_per_batch * num_batches
    batch_eval = 10
    lr = 1e-4

    # Initialize agent and target agent
    agent = DQNAgent(vocab_size, embedding_dim, hidden_dim, action_size)
    target_agent = DQNAgent(vocab_size, embedding_dim, hidden_dim, action_size)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    agent.train()

    # Train the RL model
    train_rl_model(
        agent=agent,
        target_agent=target_agent,
        env=env,
        action_symbols=action_symbols,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
        num_batches=num_batches,
        num_episodes_per_batch=num_episodes_per_batch,
        batch_quantile=batch_quantile,
        batch_size=batch_size,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update=target_update,
        memory_capacity=memory_capacity,
        batch_eval=batch_eval,
        lr=lr,
    )

    # Evaluate the agent
    constructed_expression, total_reward = evaluate_agent(
        agent=agent,
        env=env,
        action_symbols=action_symbols,
        symbol_to_index=symbol_to_index,
        max_seq_length=max_seq_length,
    )

    print(f"Final Expression: {constructed_expression}")
    print(f"Reward: {total_reward}")