from pathlib import Path
from datasets import load_dataset

from data.dataset import CIFAR10Dataset
from data.transforms import basic_transform

import torch
import torch.nn as nn
from training.train import create_loader, set_seed
from training.eval import evaluate, plot_learning_curves

from models.progressive_batchNorm_model import ProgressiveCNN
from training.config import CFG

OPTIMIZERS = {
    "AdamW": lambda params: torch.optim.AdamW(
        params,
        lr=1e-3,
        weight_decay=1e-2,
    ),
    "SGD/Momentum": lambda params: torch.optim.SGD(
        params,
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    ),
}

def main():
    project_root = Path(__file__).resolve().parent
    cache_dir = project_root / "data" / "hf_cache"

    dataset = load_dataset(
        "uoft-cs/cifar10",
        cache_dir=str(cache_dir),
    )

    test_loader = create_loader(
        CIFAR10Dataset(
            dataset["test"],
            transform=basic_transform,
        ),
        shuffle=False,
    )

    train_loader = create_loader(
        CIFAR10Dataset(
            dataset["train"],
            transform=basic_transform,
        ),
        shuffle=True,
    )

    results = {}
    best_acc = {}

    for name, optim in OPTIMIZERS.items():

        train_loss_history = []
        test_loss_history = []

        train_acc_history = []
        test_acc_history = []

        best_train_acc = 0.0
        best_test_acc = 0.0
        
        set_seed()
        model = ProgressiveCNN(use_dropout=True)

        print(f"Currently training with {name}...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim(model.parameters())

        for epoch in range(CFG.epochs):
            model.train()

            for x, y in train_loader:
                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

            train_loss, train_acc = evaluate(train_loader, model)
            test_loss, test_acc = evaluate(test_loader, model)

            print(f"Train accuracy at epoch {epoch + 1}: {train_acc * 100:5.2f}%")
            print(f"Test accuracy at epoch {epoch + 1}: {test_acc * 100:5.2f}%")

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)


            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)

        best_train_acc = max(train_acc_history)
        best_test_acc = max(test_acc_history)

        best_acc[f"Best train acc with {name}"] = best_train_acc
        best_acc[f"Best test acc with {name}"] = best_test_acc

        results[f"Train loss + {name}"] = train_loss_history
        results[f"Test loss + {name}"] = test_loss_history
        results[f"Train acc + {name}"] = train_acc_history
        results[f"Test acc + {name}"] = test_acc_history

    print()
    print()
    print("\nExperiment: ProgressiveCNN")
    print("=" * 40)
    print("Report")
    print("-" * 40)

    for opt, acc in best_acc.items():
        print(f"{opt} peaked at {acc * 100:5.2f}")
    plot_learning_curves(results)


if __name__ == "__main__":
    main()


"""
Results from terminal:

Currently training with AdamW...
Train accuracy at epoch 1: 43.78%
Test accuracy at epoch 1: 43.46%
Train accuracy at epoch 2: 51.24%
Test accuracy at epoch 2: 50.60%
Train accuracy at epoch 3: 55.39%
Test accuracy at epoch 3: 54.05%
Train accuracy at epoch 4: 59.88%
Test accuracy at epoch 4: 58.54%
Train accuracy at epoch 5: 61.56%
Test accuracy at epoch 5: 60.73%
Train accuracy at epoch 6: 62.78%
Test accuracy at epoch 6: 62.26%
Train accuracy at epoch 7: 65.41%
Test accuracy at epoch 7: 63.95%
Train accuracy at epoch 8: 66.71%
Test accuracy at epoch 8: 65.12%
Train accuracy at epoch 9: 67.46%
Test accuracy at epoch 9: 65.93%
Train accuracy at epoch 10: 68.70%
Test accuracy at epoch 10: 66.40%
Train accuracy at epoch 11: 70.55%
Test accuracy at epoch 11: 68.07%
Train accuracy at epoch 12: 69.54%
Test accuracy at epoch 12: 66.78%
Train accuracy at epoch 13: 72.43%
Test accuracy at epoch 13: 69.62%
Train accuracy at epoch 14: 72.41%
Test accuracy at epoch 14: 69.86%
Train accuracy at epoch 15: 73.81%
Test accuracy at epoch 15: 70.45%
Train accuracy at epoch 16: 74.28%
Test accuracy at epoch 16: 70.73%
Train accuracy at epoch 17: 73.91%
Test accuracy at epoch 17: 70.09%
Train accuracy at epoch 18: 74.73%
Test accuracy at epoch 18: 71.21%
Train accuracy at epoch 19: 75.77%
Test accuracy at epoch 19: 72.24%
Train accuracy at epoch 20: 76.16%
Test accuracy at epoch 20: 72.39%
Currently training with SGD/Momentum...
Train accuracy at epoch 1: 37.76%
Test accuracy at epoch 1: 37.28%
Train accuracy at epoch 2: 45.66%
Test accuracy at epoch 2: 45.25%
Train accuracy at epoch 3: 53.12%
Test accuracy at epoch 3: 52.27%
Train accuracy at epoch 4: 56.49%
Test accuracy at epoch 4: 55.80%
Train accuracy at epoch 5: 57.02%
Test accuracy at epoch 5: 56.30%
Train accuracy at epoch 6: 60.70%
Test accuracy at epoch 6: 59.00%
Train accuracy at epoch 7: 60.37%
Test accuracy at epoch 7: 59.10%
Train accuracy at epoch 8: 64.55%
Test accuracy at epoch 8: 62.70%
Train accuracy at epoch 9: 65.39%
Test accuracy at epoch 9: 64.28%
Train accuracy at epoch 10: 66.91%
Test accuracy at epoch 10: 64.94%
Train accuracy at epoch 11: 64.80%
Test accuracy at epoch 11: 63.48%
Train accuracy at epoch 12: 68.68%
Test accuracy at epoch 12: 66.60%
Train accuracy at epoch 13: 69.62%
Test accuracy at epoch 13: 67.47%
Train accuracy at epoch 14: 70.71%
Test accuracy at epoch 14: 68.63%
Train accuracy at epoch 15: 70.30%
Test accuracy at epoch 15: 67.77%
Train accuracy at epoch 16: 69.67%
Test accuracy at epoch 16: 67.38%
Train accuracy at epoch 17: 72.45%
Test accuracy at epoch 17: 69.10%
Train accuracy at epoch 18: 73.22%
Test accuracy at epoch 18: 70.33%
Train accuracy at epoch 19: 70.79%
Test accuracy at epoch 19: 68.53%
Train accuracy at epoch 20: 73.76%
Test accuracy at epoch 20: 70.98%



Experiment: ProgressiveCNN
========================================
Report
----------------------------------------
Best train acc with AdamW peaked at 76.16
Best test acc with AdamW peaked at 72.39
Best train acc with SGD/Momentum peaked at 73.76
Best test acc with SGD/Momentum peaked at 70.98


Summary:

The optimizer comparison demonstrates that AdamW consistently outperformed SGD
with Momentum when training the ProgressiveCNN under identical experimental
conditions. AdamW converged more rapidly during the early stages of training,
maintained a consistent performance advantage throughout all twenty epochs, and
achieved the highest final training and test accuracies. These results suggest
that AdamW provides a more effective optimization strategy for this network and
training budget.

Observations:
* AdamW exhibited substantially faster convergence during the first several
  epochs. After only five epochs, AdamW achieved a test accuracy of 60.73%
  compared to 56.30% for SGD with Momentum, indicating that the adaptive
  learning rates enabled more efficient optimization early in training.

* Throughout all twenty epochs, AdamW maintained a consistent lead in both
  training and test accuracy. Unlike many optimizer comparisons where the
  performance gap narrows over time, SGD never surpassed AdamW at any point
  during training.

* AdamW achieved the highest overall performance, reaching 76.16% training
  accuracy and 72.39% test accuracy, while SGD with Momentum reached
  73.76% and 70.98%, respectively. This corresponds to an improvement of
  approximately 2.40 percentage points in training accuracy and
  1.41 percentage points in test accuracy.

* The observed performance difference is likely attributable to AdamW's
  adaptive parameter updates and decoupled weight decay, which together allow
  more efficient optimization than classical stochastic gradient descent for
  this architecture and training schedule.

Conclusion

The optimizer comparison demonstrates that optimization strategy has a
measurable impact on CNN performance, even when the network architecture,
regularization techniques, and training procedure remain unchanged. Under the
controlled conditions of this experiment, AdamW consistently outperformed SGD
with Momentum by converging more rapidly, producing smoother optimization, and
achieving the highest overall classification accuracy.
"""