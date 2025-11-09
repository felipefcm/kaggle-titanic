from torch import Tensor
import torch.nn as nn


class TitanicNN(nn.Module):
    def __init__(self, input_size):
        super(TitanicNN, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x)
