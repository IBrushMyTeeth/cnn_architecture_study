from pathlib import Path
from datasets import load_dataset

from data.dataset import CIFAR10Dataset
from data.transforms import (
    basic_transform,
    flip_transform,
    crop_transform,
    color_transform,
    train_transform,
)

from training.train import create_loader, train, set_seed
from training.eval import evaluate, plot_learning_curves

from models.plain_model import PlainCNN



TRANSFORMS = {
    "baseline": basic_transform,
    "flip": flip_transform,
    "crop": crop_transform,
    "color": color_transform,
    "full": train_transform,
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

    results = []
    histories = {}

    for name, transform in TRANSFORMS.items():

        print(f"\n===== Transform: {name.upper()} =====")

        set_seed()

        train_loader = create_loader(
            CIFAR10Dataset(
                dataset["train"],
                transform=transform,
            )
        )

        model = PlainCNN()
        history = train(train_loader, model, track=True)
        histories[name] = history

        loss, acc = evaluate(test_loader, model)
        results.append((name, loss, acc))

    print("Report")
    print("-" * 35)

    for name, loss, acc in results:
        print(f"{name:<10} {loss:.3f} {acc*100:5.2f}%")

    plot_learning_curves(histories)


if __name__ == "__main__":
    main()

"""
Results from terminal (10 epochs):

Report
-----------------------------------
baseline   1.274 53.49%
flip       1.327 50.08%
crop       1.408 48.98%
color      1.335 50.86%
full       1.335 49.89%

Summary:
For this baseline CNN trained for 10 epochs, data augmentation reduced the
test accuracy compared to simple normalization alone.

Possible explanations:

* The model is likely underfitting. With only ~53% test accuracy, the baseline
  network has not yet learned the dataset well. Data augmentation increases
  the difficulty of the optimization problem, making convergence slower.

* Ten epochs may be insufficient for augmented models. The learning curves
  suggest that the augmented models are still improving and may require
  longer training to outperform the baseline.

Conclusion:
Let's try again with 20 or 30 epochs.


Results from terminal (20 epochs):

Report
-----------------------------------
baseline   1.185 57.16%
flip       1.198 56.41%
crop       1.243 55.55%
color      1.215 56.18%
full       1.264 53.73%

Summary:
After doubling the training time to 20 epochs, all models achieved higher
test accuracy. However, the baseline model without augmentation remained
the best performing configuration.

Observations:

* Individual augmentations (horizontal flip and color jitter) narrowed the
  gap to the baseline, while the full augmentation pipeline remained the
  weakest performer.

* The learning curves show that all models continue to improve, but the
  augmented models remain consistently above the baseline in training loss.
  They do not appear to rapidly converge toward the baseline.

Possible explanations:

* The baseline architecture likely has insufficient expressive power to
  fully benefit from data augmentation. Learning invariance to cropping,
  flipping, and color changes increases the difficulty of the optimization
  problem.

* Since the baseline model is intentionally simple (no BatchNorm, Dropout,
  , or residual connections), the additional variability introduced
  by augmentation may exceed the model's representational capacity.


Conclusion:

* The augmentation study should be repeated after introducing architectural
improvements such as pooling and BatchNorm.


"""