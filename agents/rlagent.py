import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from agents.encoder import TreeEncoder, SetEncoder

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
            torch.stack(states).squeeze(1),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states).squeeze(1),
            torch.tensor(dones, dtype=torch.float32),
        )

    
    def __len__(self):
        return len(self.memory)

class DQNAgent(nn.Module):
    def __init__(self, vocab_size, input_dim, embedding_dim, hidden_dim, action_size, num_heads=4, num_layers=2, max_seq_length=50):
        super(DQNAgent, self).__init__()
        # Encoders
        self.set_encoder = SetEncoder(input_dim=input_dim, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers)
        self.tree_encoder = TreeEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers, max_seq_length=max_seq_length)
        # Fusion Layer
        self.fusion_layer = nn.Linear(embedding_dim * 2, hidden_dim)
        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, action_size)
    
    def forward(self, data, tree_sequence):
        # data: (batch_size, set_size, input_dim)
        # tree_sequence: (batch_size, seq_length)
        data_embedding = self.set_encoder(data)
        tree_embedding = self.tree_encoder(tree_sequence)
        # Concatenate embeddings
        batch_size = tree_embedding.shape[0]
        data_embedding = data_embedding.expand(batch_size, -1)
        combined_embedding = torch.cat((data_embedding, tree_embedding), dim=1)
        # Pass through fusion and output layers
        x = F.relu(self.fusion_layer(combined_embedding))
        q_values = self.output_layer(x)
        return q_values
    
    def act(self, data: torch.Tensor, tree_sequence: list[str], epsilon: float):
        if random.random() < epsilon:
            # Random action
            action_idx = random.randint(0, self.output_layer.out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(data, tree_sequence)
                action_idx = torch.argmax(q_values).item()
        return action_idx

