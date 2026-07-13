"""Classifier heads used by CNN architectures."""


import torch
import torch.nn as nn


class Classifier(nn.Module):
    """A lightweight classification head.

    The classifier is intentionally kept small so that predictive performance
    depends primarily on the quality of the learned convolutional features
    rather than the capacity of the fully connected layers. This makes it
    easier to isolate and evaluate the impact of architectural changes in the
    feature extractor.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes.
    """
    
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)