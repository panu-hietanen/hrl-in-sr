import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)
    
    def push(
        self,
        state: torch.Tensor,
        action: str,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple[torch.Tensor]:
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
      
    def __len__(self) -> int:
        return len(self.memory)
    
    def prioritise(self, done: bool) -> None:
        if not done:
            state, action, _, next_state, _ = self.memory.pop()
            self.memory.append((state, action, -1, next_state, False))

class DQNAgent(nn.Module):
    def __init__(
        self,
        data_input_dim: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        action_size: int,
        max_seq_length: int
        ) -> None:
        super(DQNAgent, self).__init__()
        # Encoder for data
        self.data_encoder = nn.Sequential(
            nn.Linear(data_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Encoder for tree expression
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tree_encoder = nn.Sequential(
            nn.Linear(embedding_dim * max_seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        # Output layer
        self.fc = nn.Linear(hidden_dim, action_size)
        self.action_size = action_size
        self.max_seq_length = max_seq_length
    
    def forward(self, data_input: int, state: torch.Tensor) -> torch.Tensor:
        # Encode data
        data_embedding = self.data_encoder(data_input)
        # Encode tree expression
        x = self.embedding(state)  # Shape: (batch_size, seq_length, embedding_dim)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, seq_length * embedding_dim)
        tree_embedding = self.tree_encoder(x)
        # Fuse embeddings
        batch_size = tree_embedding.shape[0]
        data_embedding = data_embedding.expand(batch_size, -1)
        combined = torch.cat((data_embedding, tree_embedding), dim=1)
        fused_embedding = self.fusion(combined)
        # Output Q-values
        q_values = self.fc(fused_embedding)
        return q_values

    def act(
        self,
        data_input: torch.Tensor,
        state: torch.Tensor,
        epsilon: float,
        mask: torch.Tensor
        ) -> int:
        if random.random() < epsilon:
            # Random action
            valid_actions = torch.where(mask)[0].tolist()
            action_idx = random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(data_input.unsqueeze(0), state.unsqueeze(0))
                mask = mask.ge(1.0)
                q_values = torch.masked_select(q_values, mask)  # Mask invalid actions
                action_idx = torch.argmax(q_values).item()
        return action_idx
