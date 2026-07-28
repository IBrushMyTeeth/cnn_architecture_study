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
    Batch Normalization is retained after every convolution.
    """
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            BNConvBlock(3, 8),
            BNConvBlock(8, 8),
            nn.MaxPool2d(2),

            BNConvBlock(8, 16),
            BNConvBlock(16, 16),
            nn.MaxPool2d(2),

            BNConvBlock(16, 32),
            BNConvBlock(32, 32),
            nn.MaxPool2d(2),

            BNConvBlock(32, 64),
            BNConvBlock(64, 64)     
        )
        self.classifier = Classifier(in_channels=64, num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x