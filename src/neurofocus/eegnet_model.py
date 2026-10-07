import torch
from torch import nn


class EEGNet(nn.Module):
    """
    Compact EEGNet-style neural network for binary EEG classification.

    Expected input:
        (batch, 1, channels, time)

    For DEAP:
        (batch, 1, 32, 7680)
    """

    def __init__(
        self,
        num_channels=32,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        # Step 1:
        # Learn temporal/frequency patterns from the EEG.
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                8,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        # Step 2:
        # Learn relationships across all 32 EEG channels.
        self.spatial = nn.Sequential(
            nn.Conv2d(
                8,
                16,
                kernel_size=(num_channels, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),
            nn.Dropout(dropout),
        )

        # Step 3:
        # Learn additional temporal patterns.
        self.separable = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 16),
                padding="same",
                groups=16,
                bias=False,
            ),
            nn.Conv2d(
                16,
                16,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),
            nn.Dropout(dropout),
        )

        # Makes the model independent of exact time length.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            16,
            num_classes,
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return self.classifier(x)
