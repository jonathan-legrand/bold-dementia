# %%
import torch
from torch import nn
import torch.nn.functional as F


class MementoCNN(nn.Module):
    def __init__(self, n_parcellation=48) -> None:
        super().__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv1d(
                n_parcellation,
                out_channels=64,
                kernel_size=16,
                stride=4,
                dilation=5
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(5)
        )
        
        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        y = self.convblock1(x)
        y = self.flatten(y)
        y = F.relu(self.linear1(y))
        return self.linear2(y)



# %%
input = torch.rand((64, 48, 250))
cnn = MementoCNN()
output = cnn(input)

# %%
