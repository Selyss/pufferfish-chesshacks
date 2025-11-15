import torch
import torch.nn as nn

from config import FEATURE_DIM, ACC_UNITS, HIDDEN1, HIDDEN2

class NNUEModel(nn.Module):
    """
    256x2-32-32-1 style architecture.

    - Two 256 unit accumulator projections (friendly, enemy)
    - Concatenate to 512
    - FC: 512 -> 32 -> 32 -> 1
    """

    def __init__(self):
        super().__init__()
        input_dim = FEATURE_DIM

        # Accumulator projections
        self.acc_friendly = nn.Linear(input_dim, ACC_UNITS)
        self.acc_enemy = nn.Linear(input_dim, ACC_UNITS)

        # Fully connected part
        self.fc1 = nn.Linear(2 * ACC_UNITS, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc_out = nn.Linear(HIDDEN2, 1)

    def forward(self, x):
        """
        x: (batch, FEATURE_DIM) binary or small floats.
        """
        acc_f = self.acc_friendly(x)
        acc_e = self.acc_enemy(x)

        acc_f = torch.relu(acc_f)
        acc_e = torch.relu(acc_e)

        combined = torch.cat([acc_f, acc_e], dim=1)  # (batch, 512)

        y = torch.relu(self.fc1(combined))
        y = torch.relu(self.fc2(y))
        y = self.fc_out(y)
        return y  # scalar per position, eg centipawns
