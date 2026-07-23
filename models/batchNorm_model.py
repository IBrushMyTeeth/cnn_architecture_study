"""CNN architecture with Batch Normalization after each convolution."""


import torch
import torch.nn as nn
from models.blocks.batchNorm_conv_block import BNConvBlock
from models.classifier import Classifier


class BatchNormCNN(nn.Module):
    
    """CNN architecture with Batch Normalization.

    This model extends the baseline CNN by replacing each convolutional block
    with a Batch Normalization variant. The overall architecture remains
    unchanged, allowing the effect of Batch Normalization on optimization,
    convergence speed, and classification performance to be evaluated in
    isolation.
    """
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            BNConvBlock(3, 8),
            BNConvBlock(8, 8),
            nn.MaxPool2d(2),

            BNConvBlock(8, 8),
            BNConvBlock(8, 8),
            nn.MaxPool2d(2),

            BNConvBlock(8, 8),
            BNConvBlock(8, 8)
        )

        self.classifier = Classifier(in_channels=8, num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x