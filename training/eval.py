"""
Utility functions for model evaluation and visualization.

Includes helpers for computing predictions, measuring accuracy,
evaluating model performance, and plotting learning curves and
confusion matrices.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Prediction and evaluation

def predict(logits: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    return pred

def accuracy(
        logits: torch.Tensor,
        y: torch.Tensor
) -> float:
    
    pred = predict(logits)
    acc = (pred == y).float().mean()
    return acc.item()

def evaluate(
        loader: DataLoader,
        model: nn.Module
) -> tuple[float, float]:
    
    loss = 0
    acc = 0
    criterion = CrossEntropyLoss()

    model.eval()

    with torch.inference_mode():
        for x, y in loader:

            logits = model(x)

            batch_loss = criterion(logits, y).item()
            batch_acc = accuracy(logits, y)

            acc += batch_acc
            loss += batch_loss

    avg_loss = loss / len(loader)
    avg_acc = acc / len(loader)

    return avg_loss, avg_acc

# Visualization

def plot_learning_curve(learning_history: list[float]) -> None:
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(learning_history) + 1), learning_history, marker="o")
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
        predictions: torch.Tensor,
        y: torch.Tensor,
        class_names: list[str] | None = None
) -> None:
    
    labels = y.detach().numpy()
    preds = predictions.detach().numpy()
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def evaluate_confusion_matrix(
        loader: DataLoader,
        model: nn.Module,
        class_names: list[str] | None = None
) -> None:
    
    model.eval()

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for x, y in loader:
            logits = model(x)
            preds = predict(logits)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    plot_confusion_matrix(
        all_preds,
        all_labels,
        class_names=class_names,
    )

def plot_learning_curves(histories: dict[str, list[float]]) -> None:
    plt.figure(figsize=(7, 5))
    for name, history in histories.items():
        plt.plot(range(1, len(history) + 1), history, marker="o", label=name)
    plt.title("Learning Curves by Transform")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()