"""Identical to the batchNorm model, but deeper."""


import torch
import torch.nn as nn
from models.blocks.batchNorm_conv_block import BNConvBlock
from models.classifier import Classifier


class DeepBatchNormCNN(nn.Module):
    
    """
    This model doubles the depth of the BatchNorm model while keeping
    everything else fixed.
    """
    def __init__(self, width: int = 8) -> None:
        super().__init__()

        self.features = nn.Sequential(
            BNConvBlock(3, width),
            BNConvBlock(width, width),
            nn.MaxPool2d(2),

            BNConvBlock(width, width),
            BNConvBlock(width, width),
            nn.MaxPool2d(2),

            BNConvBlock(width, width),
            BNConvBlock(width, width),
            nn.MaxPool2d(2),

            BNConvBlock(width, width),
            BNConvBlock(width, width),
            nn.MaxPool2d(2),

            BNConvBlock(width, width),
            BNConvBlock(width, width)
        )

        self.classifier = Classifier(in_channels=width, num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x