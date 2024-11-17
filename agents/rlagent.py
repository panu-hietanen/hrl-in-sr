import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolicRegressionAgent(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, action_size: int):
        super(SymbolicRegressionAgent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action_mask):
        x = self.embedding(state)
        # Assuming state is of shape (batch_size, sequence_length)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)  # Remove the layer dimension
        logits = self.policy_head(h_n)
        # Apply action mask
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = F.softmax(logits, dim=-1)
        value = self.value_head(h_n).squeeze(-1)
        return action_probs, value
