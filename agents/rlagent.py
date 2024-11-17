import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolicRegressionAgent(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, action_size):
        super(SymbolicRegressionAgent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action_mask):
        x = self.embedding(state)  # Shape: (batch_size, seq_length, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, batch_size, hidden_dim)
        h_n = h_n[-1]  # Get the output of the last LSTM layer
        logits = self.policy_head(h_n)  # Shape: (batch_size, action_size)
        # Apply action mask before softmax
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, action_size)
        value = self.value_head(h_n).squeeze(-1)  # Shape: (batch_size,)
        return action_probs, value
