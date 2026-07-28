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
    model = ProgressiveCNN()

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

    print("\nExperiment: ProgressiveCNN")
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

Experiment: ProgressiveCNN
========================================
Report
----------------------------------------
Training loss:     0.299
Training accuracy: 89.50%
Test loss:         0.802
Test accuracy:     75.76%
Generalization gap: 13.74%
train_loss_history:
[1.1797181268024932, 0.9060815457645279, 0.8050479317260215, 0.689478429915655, 0.6465904318615604, 0.6522803573352297, 0.572856320330249, 0.5490227396911977, 0.475431357579463, 0.4581132146441723, 0.4448702854992789, 0.4891212589829169, 0.4011909348885422, 0.38177516829708347, 0.44899677458550313, 0.3542886570934444, 0.34202630082359703, 0.31361031374129494, 0.302688391569554, 0.29945713868531426]
test_loss_history:
[1.2316761992539569, 0.9741061451328787, 0.8892534170181129, 0.796157530158948, 0.7867302135297447, 0.7963077196270038, 0.7573484098835356, 0.7483071622195517, 0.7142125339644729, 0.7235047162338427, 0.7155133918592125, 0.7808792511369013, 0.7225156601067562, 0.7410801134671375, 0.8352145436842731, 0.7504726840052635, 0.7771789125955788, 0.7765964983375209, 0.7749140467613366, 0.8024455694256315]
train_acc_history:
[0.570192615089514, 0.6776094948849105, 0.7150135869565217, 0.7599304667519181, 0.7727781329923273, 0.7731777493606138, 0.7990529092071611, 0.8091432225063938, 0.8355378836317136, 0.8395140664961637, 0.8437300191815856, 0.8256473785166241, 0.8612531969309463, 0.8679267902813299, 0.8398937020460358, 0.8768981777493606, 0.8775375639386189, 0.890005594629156, 0.8941216432225064, 0.8950007992327366]
test_acc_history:
[0.5553343949044586, 0.6576433121019108, 0.6843152866242038, 0.7205414012738853, 0.7281050955414012, 0.7303941082802548, 0.7341759554140127, 0.7413415605095541, 0.7590565286624203, 0.7581608280254777, 0.7586584394904459, 0.7417396496815286, 0.7581608280254777, 0.7584593949044586, 0.7361664012738853, 0.7589570063694268, 0.7530851910828026, 0.7622412420382165, 0.7586584394904459, 0.7575636942675159]


Summary:

Replacing the fixed-width architecture with a progressively wider network
resulted in the highest test accuracy observed so far. Despite using fewer
convolutional layers than the previous deep model, the progressive
architecture improved test accuracy by approximately two percentage points,
indicating that allocating more feature channels in deeper layers is a more
effective use of model capacity than maintaining a fixed width throughout the
network.

Observations:

* The training loss decreased smoothly throughout the 20 epochs, indicating
stable optimization. Unlike previous experiments, the model continued to
improve until the final epoch without exhibiting optimization difficulties.

* Training accuracy steadily increased to 89.50%, while the final test
accuracy reached 75.76%, representing an improvement of approximately two
percentage points over the previous fixed-width deep architecture.

* Test accuracy plateaued around 76% after approximately the ninth epoch and
fluctuated only slightly thereafter, despite continued improvements in
training performance. This indicates that the model is beginning to overfit
the training data.

Epoch 15 : 73.62%
Epoch 16 : 75.90%
Epoch 17 : 75.31%
Epoch 18 : 76.22%  <-- best
Epoch 19 : 75.87%
Epoch 20 : 75.76%

* The test loss initially followed the training loss but began to plateau
after roughly the thirteenth epoch. Although test accuracy remained
relatively stable, the increasing test loss suggests that the model became
progressively more confident in its incorrect predictions.

* The final generalization gap of 13.74% indicates that the model is beginning
to overfit the training data. While the network continues to learn useful
representations, additional optimization alone is unlikely to yield
substantial improvements in test performance.

* Compared to the previous deep architecture, the progressive design achieved
higher classification accuracy using fewer convolutional layers. This
suggests that progressively increasing the number of feature channels is a
more effective architectural choice than simply increasing network depth.

Conclusion:

The progressive architecture successfully improved the representational
capacity of the network while maintaining stable optimization throughout
training, producing the best overall classification performance obtained in
this project so far.
The primary limitation of the current architecture is no longer optimization
but generalization. Future experiments should therefore investigate
regularization techniques, such as dropout or stronger data augmentation,
rather than simply increasing network depth.
"""