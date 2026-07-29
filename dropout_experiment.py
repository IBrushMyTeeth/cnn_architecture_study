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

    print("Training the model...")
    set_seed()
    model = ProgressiveCNN(use_dropout=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

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

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    results = {
        "train_loss": train_loss_history,
        "test_loss": test_loss_history,
        "train_acc": train_acc_history,
        "test_acc": test_acc_history
    }

    print("\nExperiment: Dropout / ProgressiveCNN")
    print("=" * 40)
    print("Report")
    print("-" * 40)

    final_tr_loss = train_loss_history[-1]
    final_ts_loss = test_loss_history[-1]
    final_tr_acc = train_acc_history[-1]
    final_ts_acc = test_acc_history[-1]

    gap = final_tr_acc - final_ts_acc

    print(f"Training loss:     {final_tr_loss:.3f}")
    print(f"Training accuracy: {final_tr_acc*100:5.2f}%")
    print(f"Test loss:         {final_ts_loss:.3f}")
    print(f"Test accuracy:     {final_ts_acc*100:5.2f}%")
    print(f"Generalization gap: {gap*100:.2f}%")

    print("train_loss_history:")
    print(train_loss_history)
    print("test_loss_history:")
    print(test_loss_history)
    print("train_acc_history:")
    print(train_acc_history)
    print("test_acc_history:")
    print(test_acc_history)

    plot_learning_curves(results)

if __name__ == "__main__":
    main()

"""
Results from terminal:

Experiment: Dropout / ProgressiveCNN
========================================
Report
----------------------------------------
Training loss:     0.686
Training accuracy: 76.16%
Test loss:         0.795
Test accuracy:     72.39%
Generalization gap: 3.76%

train_loss_history:
[1.5061952791860342, 1.320352479624931, 1.2170669119376356, 1.1130141503823079, 1.059076127417557, 1.0176116614542958, 0.9502157998816742, 0.9273786419797736, 0.9062359046448222, 0.8809843014573198, 0.8220459327978247, 0.863211114312072, 0.7806411685846041, 0.7904796265351498, 0.7463952958050286, 0.7393427090266781, 0.7422389385797789, 0.7195333862853477, 0.6911965917672038, 0.685925607752922]
test_loss_history:
[1.5064284725553672, 1.3272457737831553, 1.2386425384290658, 1.1468067750050004, 1.093994812980579, 1.0598045860885814, 1.002170766994452, 0.9773019977435944, 0.9583868467883699, 0.9454697063014765, 0.891690897713801, 0.9320870364547535, 0.862090884120601, 0.8651664367147313, 0.8371223048040062, 0.8342158889314931, 0.8455158593548331, 0.8190873876498763, 0.7984314234393417, 0.7952366139099096]
train_acc_history:
[0.4377597506393862, 0.5123881074168798, 0.5538882672634271, 0.5987651854219949, 0.6156289961636828, 0.6277973145780051, 0.6540720907928389, 0.6670596227621484, 0.6746323529411765, 0.6870204603580563, 0.7055027173913043, 0.6953924232736572, 0.7242647058823529, 0.7241048593350383, 0.7381114130434783, 0.7427869245524297, 0.7391104539641944, 0.7472826086956522, 0.7576526534526854, 0.7615688938618926]
test_acc_history:
[0.4346138535031847, 0.5059713375796179, 0.5405055732484076, 0.585390127388535, 0.6072850318471338, 0.6226114649681529, 0.63953025477707, 0.6511743630573248, 0.6593351910828026, 0.6640127388535032, 0.6807324840764332, 0.6677945859872612, 0.6961584394904459, 0.6986464968152867, 0.7045183121019108, 0.7073049363057324, 0.7009355095541401, 0.7120820063694268, 0.7224323248407644, 0.7239251592356688]

Summary: 

Introducing dropout into the progressive CNN substantially reduced overfitting
by keeping the training and test performance closely aligned throughout
training. An initial dropout configuration proved overly aggressive, so the
regularization strength was reduced and the experiment repeated to obtain a
better balance between optimization and generalization. Although the tuned model
achieved a lower final test accuracy than the baseline ProgressiveCNN architecture,
it maintained a dramatically smaller generalization gap while continuing to improve
throughout the final training epochs, indicating that the network had not yet
fully converged.


Observations:

* An initial dropout configuration introduced excessive regularization,
  substantially reducing both training and test accuracy. After reducing
  the dropout strength, the model recovered much of the lost performance
  while preserving the strong regularization effect.

| Metric             | No Dropout | Previous Dropout | Tuned Dropout |
| ------------------ | ---------: | ---------------: | ------------: |
| Train accuracy     | **89.50%** |           73.47% |    **76.16%** |
| Test accuracy      | **75.76%** |           69.76% |    **72.39%** |
| Train loss         |  **0.299** |            0.755 |     **0.686** |
| Test loss          |      0.802 |            0.850 |     **0.795** |
| Generalization gap |     13.74% |            3.72% |     **3.76%** |



* Unlike the baseline model, the training and test losses remained closely
  aligned throughout training. The small separation between the two curves
  demonstrates that dropout effectively reduced overfitting and improved the
  consistency of the model's generalization.

* Training accuracy increased steadily to 76.16%, while the final test accuracy
  reached 72.39%. Although both values remained below those of the baseline model,
  they represented a substantial improvement over the initial dropout configuration.

* Test accuracy continued to improve throughout the entire training process without
  exhibiting the early plateau observed in the baseline architecture.
Epoch 16 : 70.73%
Epoch 17 : 70.09%
Epoch 18 : 71.21%
Epoch 19 : 72.24%
Epoch 20 : 72.39%

* The final generalization gap was reduced from 13.74% in the baseline model
  to only 3.76%, demonstrating that dropout successfully improved the model's
  generalization despite the reduction in final accuracy.

* The continued improvement of both the training and test curves at the final
  epoch suggests that the tuned dropout model had not yet converged. A longer
  training schedule would likely allow the network to recover additional
  performance while retaining its improved resistance to overfitting.

  
Conclusion:

Adding dropout successfully addressed the overfitting observed in the progressive
CNN by greatly reducing the divergence between training and test performance.
After tuning the dropout strength, the model achieved a substantially better
balance between optimization and regularization than the initial dropout
configuration, recovering much of the lost accuracy while maintaining a very
small generalization gap.

Although the tuned model did not surpass the baseline in final test accuracy,
it exhibited significantly more stable learning dynamics and continued to
improve until the final epoch, indicating that optimization had not yet
converged. Overall, this experiment demonstrates the expected trade-off
introduced by dropout: stronger regularization reduces overfitting and improves
generalization consistency but also slows optimization. Future work should
investigate longer training schedules to determine whether the remaining
accuracy gap can be reduced while preserving the improved generalization
behavior.
"""