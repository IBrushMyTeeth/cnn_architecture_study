"""Baseline CNN architecture used throughout the experiments."""


import torch
import torch.nn as nn
from models.blocks.conv_block import ConvBlock
from classifier import Classifier


class PlainCNN(nn.Module):

    """Reference CNN architecture for architectural experiments.

    This model serves as the baseline against which subsequent architectural
    modifications are evaluated. It consists of six convolutional blocks with
    two downsampling stages, followed by a lightweight classifier to encourage
    feature learning within the convolutional backbone.
    """
    
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 8),
            ConvBlock(8, 8),
            nn.MaxPool2d(2),

            ConvBlock(8, 8),
            ConvBlock(8, 8),
            nn.MaxPool2d(2),

            ConvBlock(8, 8),
            ConvBlock(8, 8)
        )

        self.classifier = Classifier(in_channels=8, num_classes=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x