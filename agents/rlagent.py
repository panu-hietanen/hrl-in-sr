import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
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

    def sample(self):
        """
        Convert stored lists into PyTorch tensors.
        Returns:
          states:  (N, ...)  [stacked from stored states]
          actions: (N,)      [long tensor of action indices]
          log_probs: (N,)    [float tensor]
          values:   (N,)     [value predictions at each state]
          rewards:  (N,)     [float tensor of rewards]
          dones:    (N,)     [bool or float tensor indicating done]
        """
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        log_probs = torch.stack(self.log_probs).squeeze(-1)
        values = torch.stack(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.bool)

        return states, actions, log_probs, values, rewards, dones

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
        self.data_encoder = nn.Sequential(
            nn.Linear(data_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tree_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.tree_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fusion_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_activation = nn.ReLU()
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.action_size = action_size
        self.max_seq_length = max_seq_length

    def forward(self, data_input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        data_input: (batch_size, data_input_dim)
        state: (batch_size, max_seq_length) with token indices
        """
        # Encode the data input
        data_embedding = self.data_encoder(data_input)  # shape: (batch_size, hidden_dim)
        
        # Encode the symbolic expression (tree state)
        # Get embeddings: shape (batch_size, max_seq_length, embedding_dim)
        x = self.embedding(state)
        # Process the sequence with an LSTM: use the last hidden state as the summary
        lstm_out, (h_n, _) = self.tree_lstm(x)
        tree_embedding = h_n[-1]  # shape: (batch_size, hidden_dim)
        # Optionally, apply an additional projection
        tree_embedding = self.tree_projection(tree_embedding)
        
        # Ensure data_embedding is properly broadcast to match batch size
        batch_size = tree_embedding.shape[0]
        if data_embedding.dim() == 1:
            data_embedding = data_embedding.unsqueeze(0).expand(batch_size, -1)
        elif data_embedding.size(0) != batch_size:
            raise ValueError("Mismatch in batch sizes between data_input and tree states.")
        
        # Fuse the two representations
        combined = torch.cat((data_embedding, tree_embedding), dim=1)  # shape: (batch_size, hidden_dim*2)
        fusion_out = self.fusion_linear(combined)
        fusion_out = self.fusion_activation(fusion_out)
        fused_embedding = self.fusion_norm(fusion_out)  # shape: (batch_size, hidden_dim)
        
        # Produce policy logits and value estimate
        logits = self.policy_head(fused_embedding)         # shape: (batch_size, action_size)
        value = self.value_head(fused_embedding).squeeze(-1)  # shape: (batch_size)
        return logits, value

    def act(
        self,
        data_input: torch.Tensor,
        state: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        data_input: (data_input_dim,) or (1, data_input_dim)
        state: (max_seq_length,) or (1, max_seq_length)
        mask: (action_size,) (optional) indicating valid actions (1 = valid, 0 = invalid)
        """
        if data_input.dim() == 1:
            data_input = data_input.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits, value = self.forward(data_input, state)
        
        if mask is not None:
            # Assume mask shape is (action_size,) and apply to the first batch element
            logits = logits.clone()
            logits[0, mask == 0] = -float('inf')
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze(0), value.squeeze(0)
