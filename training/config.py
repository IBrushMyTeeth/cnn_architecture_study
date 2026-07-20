"""
Module used to ensure identical training configuration across all models. By 
standardizing training, the effect of architectural decisions is isolated, and
can fairly be measured.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Training configuration shared across all experiments."""
    # Optimization
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64

    # Dataloader
    num_workers: int = 2

    # Reproducability
    seed: int = 23

CFG = Config()