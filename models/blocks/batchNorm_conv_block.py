"""Building blocks for convolutional neural network architectures."""


import torch
import torch.nn as nn


class BNConvBlock(nn.Module):
    """
    A convolution followed by Batch Normalization and a ReLU activation.

    This block extends the baseline convolutional block by introducing Batch
    Normalization after the convolution. It is used to evaluate the impact of
    feature normalization on optimization, convergence speed, and
    classification performance while keeping the overall architecture
    unchanged.
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)