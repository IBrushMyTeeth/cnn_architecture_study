"""
Utilities for standardized model training.

Provides helper functions for creating dataloaders and training
models using the shared experiment configuration.
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import CFG


def create_loader(
    data: Dataset,
    shuffle: bool = True
)-> DataLoader:
    """Create standardized dataloaders from config file"""
    return DataLoader(
        dataset=data,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers
    )

def train(
    loader : DataLoader,
    model : nn.Module,
    track : bool = False
) -> None | list[float]:
    """Train a model and optionally return the corresponding loss historic"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
    )

    criterion = nn.CrossEntropyLoss()

    if track:
        loss_history = []

    model.train()
    for epoch in range(CFG.epochs):
        epoch_loss = 0

        for x, y in loader:
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        if track:
            loss_history.append(epoch_loss / len(loader))
    
    if track:
        return loss_history