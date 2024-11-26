import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: torch.Tensor, action, reward, next_state: torch.Tensor, done):
        self.memory.append((state.detach(), action, reward, next_state.detach(), done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = [item[0] for item in batch]
        actions = [item[1] for item in batch]
        rewards = [item[2] for item in batch]
        next_states = [item[3] for item in batch]
        dones = [item[4] for item in batch]
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )
      
    def __len__(self):
        return len(self.memory)
    
    def prioritise(self, done: bool) -> None:
        if not done:
            state, action, _, next_state, _ = self.memory.pop()
            self.memory.append((state, action, -1, next_state, False))

class DQNAgent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, action_size):
        super(DQNAgent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_size)
        self.action_size = action_size
    
    def forward(self, state):
        x = self.embedding(state)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n = h_n[-1]
        q_values = self.fc(h_n)
        return q_values

    def act(self, state, epsilon):
        if random.random() < epsilon:
            # Random action
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self(state)
                action_idx = torch.argmax(q_values).item()
        return action_idx
