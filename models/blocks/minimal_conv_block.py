"""Building blocks for convolutional neural network architectures."""


import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A convolution followed by a ReLU activation.
    This block serves as the basic feature extraction unit for the CNN
    architectures in this project. It intentionally excludes components such
    as pooling, normalization, or dropout so they can be introduced and
    evaluated independently in later experiments.
    """

    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int = 3,
                padding: int = 1
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)