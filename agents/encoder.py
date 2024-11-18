import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, n_heads: int = 4, n_layers: int = 2) -> None:
        super(SetEncoder, self).__init__()
        self.input_layer = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=n_heads,
                    dim_feedforward=embedding_dim * 2,
                    activation='relu',
                )
                for _ in range(n_layers)
            ]
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 2, 0)
        x = self.pooling(x).squeeze(-1)
        return x

class TreeEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int = 4, num_layers: int = 2, max_seq_length: int = 50) -> None:
        super(TreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 2,
                activation='relu'
            )
            for _ in range(num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 2, 0)
        x = self.pooling(x).squeeze(-1)
        return x

