import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np


class NextWordModel(nn.Module):
    """
    Recurrent Neural Network (GRU-based) for next-token / next-word prediction.
    Designed for Federated Learning text prediction and privacy leakage evaluation.
    
    Architecture:
      - Embedding layer: Maps token IDs to continuous dense vectors.
      - Recurrent layer: Multi-step GRU capturing temporal token dependencies.
      - Linear head: Maps hidden state representations to vocabulary logit distribution.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for next-token prediction.
        
        Args:
            x: Tensor of shape (batch_size, seq_len) with token indices.
            hidden: Optional initial hidden state for GRU of shape (num_layers, batch_size, hidden_dim).
            
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size).
            hidden: Updated hidden state of shape (num_layers, batch_size, hidden_dim).
        """
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initializes zero tensor for the hidden state."""
        dev = device or next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=dev)

    def get_weights(self) -> List[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays for Flower FedAvg."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        state_dict = dict(zip(self.state_dict().keys(), [torch.tensor(w) for w in weights]))
        self.load_state_dict(state_dict, strict=True)
