"""Baseline CNN architecture used throughout the experiments."""


import torch
import torch.nn as nn
from models.blocks.minimal_conv_block import ConvBlock
from models.classifier import Classifier


class PlainCNN(nn.Module):

    """Reference CNN architecture for architectural experiments.

    This model serves as the baseline against which subsequent architectural
    modifications are evaluated. It consists of six convolutional blocks with
    two downsampling stages, followed by a lightweight classifier to encourage
    feature learning within the convolutional backbone.
    """
    
    def __init__(self, width: int = 8) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, width),
            ConvBlock(width, width),
            nn.MaxPool2d(2),

            ConvBlock(width, width),
            ConvBlock(width, width),
            nn.MaxPool2d(2),

            ConvBlock(width, width),
            ConvBlock(width, width)
        )

        self.classifier = Classifier(in_channels=width, num_classes=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x