# model.py
"""
Tiny neural network for chess move evaluation.
Imports the architecture from training folder.
"""
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """
    Ultra-small MLP for evaluating chess moves.
    
    Input features (12 dims):
      - Attacker piece type (6 one-hot: pawn, knight, bishop, rook, queen, king)
      - Material delta (1 value)
      - PST delta (1 value)
      - Is capture (1 value: 0 or 1)
      - Is check (1 value: 0 or 1)
      - From square centrality (1 value)
      - To square centrality (1 value)
    
    Architecture: Linear(12, 8) -> ReLU -> Linear(8, 1)
    """
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 12) containing move features
        
        Returns:
            Tensor of shape (batch, 1) with move evaluation scores
        """
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


