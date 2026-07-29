"""CNN architecture with progressively increasing channel width."""


import torch
import torch.nn as nn
from models.blocks.batchNorm_conv_block import BNConvBlock
from models.classifier import Classifier


class ProgressiveCNN(nn.Module):

    """
    CNN architecture with progressive channel expansion.

    This model builds upon the Batch Normalization architecture by
    progressively increasing the number of feature channels throughout the
    network. Instead of using a fixed width for all convolutional layers, the
    channel dimensions double after each pooling stage (8 → 16 → 32 → 64),
    allowing the network to learn increasingly rich feature representations at
    deeper layers. Max pooling is used for spatial downsampling, while
    Batch Normalization is retained after every convolution. Additionally
    dropout can be activated.
    """
    def __init__(self, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [
            BNConvBlock(3, 8),
            BNConvBlock(8, 8),
            nn.MaxPool2d(2),
        ]

        if use_dropout:
            layers.append(nn.Dropout2d(0.1))

        layers.extend([
            BNConvBlock(8, 16),
            BNConvBlock(16, 16),
            nn.MaxPool2d(2),
        ])

        if use_dropout:
            layers.append(nn.Dropout2d(0.2))

        layers.extend([
            BNConvBlock(16, 32),
            BNConvBlock(32, 32),
            nn.MaxPool2d(2),
        ])

        if use_dropout:
            layers.append(nn.Dropout2d(0.3))

        layers.extend([
            BNConvBlock(32, 64),
            BNConvBlock(64, 64),
        ])

        self.features = nn.Sequential(*layers)
        p = 0.2 if use_dropout else 0.0
        self.classifier = Classifier(in_channels=64, num_classes=10, dropout=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x