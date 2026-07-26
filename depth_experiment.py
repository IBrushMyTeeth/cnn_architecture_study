from pathlib import Path
from datasets import load_dataset

from data.dataset import CIFAR10Dataset
from data.transforms import basic_transform

from training.train import create_loader, train, set_seed
from training.eval import evaluate, plot_learning_curves

from models.batchNorm_model import BatchNormCNN
from models.deep_batchNorm_model import DeepBatchNormCNN


MODELS = {
    "BatchNormCNN16": BatchNormCNN,
    "DeepBatchNormCNN16": DeepBatchNormCNN
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

    results = []
    histories = {} 


    for name, Model in MODELS.items():
        set_seed()

        model = Model(16)

        print(f"Currently training {name}:") 
        history = train(train_loader, model, track=True)
        print(history)
        histories[name] = history

        loss, acc = evaluate(test_loader, model)
        results.append((name, loss, acc))


    print("\nExperiment: Depth")
    print("=" * 40)
    print("Report")
    print("-" * 40)

    for name, loss, acc in results:
        print(F"{name:<10} {loss:.3f} {acc*100:5.2f}%")

    plot_learning_curves(histories)

if __name__ == "__main__":
    main()


"""
Results from terminal:
Currently training BatchNormCNN16:
[1.5079873099808803, 1.1266702337338186, 0.9981052026419384, 0.9360195195964535, 0.8883849476914272, 0.8519329877613145, 0.8196620416763188, 0.7968040696342887, 0.7775738576184148, 0.760190819840297, 0.7417060073531802, 0.7313553403939128, 0.7201663874985312, 0.705653405669705, 0.6942467829760384, 0.6878135745482676, 0.6735368971629521, 0.6686905641918597, 0.6587821045876159, 0.6552066016380135]
Currently training DeepBatchNormCNN16:
[1.5258365441161348, 1.1028393423160934, 0.9615899866346813, 0.8744654700426799, 0.8203023483076364, 0.7771068017196168, 0.7488384614591403, 0.725168640808681, 0.6964514997151806, 0.6784696148713226, 0.6616428035985479, 0.6457811413366167, 0.6298223021618851, 0.6171160434060694, 0.6020041569267087, 0.5957987188073375, 0.5866546797401765, 0.5767480198608335, 0.5664397532982595, 0.5592454051422646]

** Make sure to look at the curves.

Experiment: Depth
========================================
Report
----------------------------------------
BatchNormCNN16 0.770 72.66%
DeepBatchNormCNN16 0.761 73.96%


Summary:
Increasing the network depth while keeping the width fixed at 16 improved
optimization throughout training. The deeper model consistently achieved lower
training losses and slightly higher classification accuracy on the test set.

Observations:

* The deeper model reduced the training loss throughout the entire training
  process, indicating that the additional convolutional layers increased the
  model's representational capacity.

* Both models converged rapidly during the first epochs, but the deeper model
  maintained a lower training loss after the initial convergence and continued
  improving throughout all 20 epochs.

* Test accuracy increased from 72.66% for the original BatchNormCNN16 to
  73.96% for the deeper architecture, while the test loss remained nearly
  unchanged (0.770 vs. 0.761).

* Unlike the shallower model, the learning curve of the deeper model showed no
  clear plateau after 20 epochs, suggesting that additional training may
  further improve performance.

* Although the deeper network learned the training data substantially better,
  this resulted in only a modest improvement in test performance, indicating
  diminishing returns from increasing depth alone.

Conclusion:

* Increasing network depth successfully improved the model's optimization and
  representational capacity, as evidenced by the consistently lower training
  loss.

* However, the improvement in generalization was relatively small compared with
  the reduction in training loss, suggesting that additional depth alone is
  insufficient to produce large gains in classification performance under the
  current training configuration.

* Future work could investigate whether longer training, learning-rate
  scheduling, or regularization techniques allow the deeper architecture to
  better translate its increased capacity into improved generalization.

"""