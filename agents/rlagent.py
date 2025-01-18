import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

class RolloutBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store(
            self, 
            state: torch.Tensor, 
            action: str, 
            log_prob: float, 
            value: float, 
            reward: float, 
            done: bool
            ) -> None:
        """
        Store a single timestep of data.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        """
        Clear buffer after an update step.
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def sample(self) -> tuple[list, list, list, list, list, list]:
        return (
            self.states, 
            self.actions, 
            self.log_probs, 
            self.values, 
            self.values, 
            self.rewards, 
            self.dones
        )
      
    def __len__(self) -> int:
        return len(self.states)

class PPOAgent(nn.Module):
    def __init__(
        self,
        data_input_dim: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        action_size: int,
        max_seq_length: int
    ) -> None:
        super(PPOAgent, self).__init__()
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
        
        # Policy head (logits over discrete actions)
        self.policy_head = nn.Linear(hidden_dim, action_size)
        # Value head (estimates V(s))
        self.value_head = nn.Linear(hidden_dim, 1)

        self.action_size = action_size
        self.max_seq_length = max_seq_length
    
    def forward(self, data_input: int, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode data
        data_embedding = self.data_encoder(data_input)
        # Encode tree expression
        x = self.embedding(state)  # Shape: (batch_size, seq_length, embedding_dim)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, seq_length * embedding_dim)
        tree_embedding = self.tree_encoder(x)
        # Fuse embeddings
        batch_size = tree_embedding.shape[0]
        if data_embedding.dim() == 1:
            data_embedding = data_embedding.unsqueeze(0).expand(batch_size, -1)
        elif data_embedding.size(0) != batch_size:
            raise ValueError("Mismatch in batch sizes between data_input and tree states.")
        
        combined = torch.cat((data_embedding, tree_embedding), dim=1)
        fused_embedding = self.fusion(combined)
        
        logits = self.policy_head(fused_embedding)
        value = self.value_head(fused_embedding).squeeze(-1)

        return logits, value

    def act(
        self,
        data_input: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor = None
        ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if data_input.dim() == 1:
            data_input = data_input.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.forward(data_input, state)

        if mask is not None:
            logits = logits.clone()
            logits[0, mask == 0] = -float('inf')

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0), value.squeeze(0)
