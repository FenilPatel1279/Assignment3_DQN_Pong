import torch
import torch.nn as nn

class DQNCNN(nn.Module):
    def __init__(self, action_size=6):
        super(DQNCNN, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # Output: (32, 20, 19)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Output: (64, 9, 8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Output: (64, 7, 6)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.conv_net(x)
        return self.fc(x)

