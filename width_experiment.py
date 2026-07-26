from pathlib import Path
from datasets import load_dataset

from data.dataset import CIFAR10Dataset
from data.transforms import basic_transform

from training.train import create_loader, train, set_seed
from training.eval import evaluate, plot_learning_curves

from models.batchNorm_model import BatchNormCNN


MODELS = {
    "BatchNormCNN8": 8,
    "BatchNormCNN12": 12,
    "BatchNormCNN16": 16
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


    for name, wd in MODELS.items():
        set_seed()

        model = BatchNormCNN(width=wd)

        print(f"Currently training {name}:") 
        history = train(train_loader, model, track=True)
        print(history)
        histories[name] = history

        loss, acc = evaluate(test_loader, model)
        results.append((name, loss, acc))


    print("\nExperiment: Width")
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

Currently training BatchNormCNN8:
[1.6813055404921626, 1.3891060000185467, 1.2757134767596985, 1.2200636016895703, 1.179351651607572, 1.1438715883990382, 1.1185693894810689, 1.0961192817334324, 1.0795675060328316, 1.0699033375133944, 1.0509279735405426, 1.0431835235232283, 1.033439100703315, 1.0228139560698244, 1.0130353848952467, 1.010277665408371, 1.001176499878354, 0.9934336767934472, 0.9856832296875737, 0.9838415722712837]
Currently training BatchNormCNN12:
[1.6195876999279422, 1.2431728022787578, 1.1121290879481285, 1.0421319301323513, 0.9971448401813312, 0.9642395952625957, 0.9376887315526947, 0.9160612387120571, 0.8991956470719994, 0.8846137149407126, 0.870011376374213, 0.8577820928505314, 0.8491148256584812, 0.8371043332549922, 0.8255198800655277, 0.8176038076962961, 0.8095135356459167, 0.8013070848439355, 0.792967449254392, 0.7876666515227169]
Currently training BatchNormCNN16:
[1.5079873099808803, 1.1266702337338186, 0.9981052026419384, 0.9360195195964535, 0.8883849476914272, 0.8519329877613145, 0.8196620416763188, 0.7968040696342887, 0.7775738576184148, 0.760190819840297, 0.7417060073531802, 0.7313553403939128, 0.7201663874985312, 0.705653405669705, 0.6942467829760384, 0.6878135745482676, 0.6735368971629521, 0.6686905641918597, 0.6587821045876159, 0.6552066016380135]

Experiment: Width
========================================
Report
----------------------------------------
BatchNormCNN8 1.091 60.53%
BatchNormCNN12 0.859 69.69%
BatchNormCNN16 0.770 72.66%


Summary:
Increasing the network width while keeping the architecture and training
configuration fixed led to substantial improvements in classification performance.
Wider models achieved lower training losses, converged more rapidly, and produced
higher test accuracies. Drawn curves depict this clearly.

Observations:

* Increasing the width consistently reduced the training loss throughout
  training, indicating that the additional feature channels increased the
  model's representational capacity.

* Larger models converged faster, with the widest model (width=16) exhibiting
  the lowest loss from the first epoch onward.

* Test accuracy improved from 60.53% for width=8 to 69.69% for width=12 and
  further to 72.66% for width=16. The largest improvement occurred between
  width=8 and width=12, while the gain from width=12 to width=16 was smaller,
  suggesting diminishing returns from further widening.

* Drawn curves suggest that the learning curve of largest model width=16,
  stringly plateaus starting from the 18th epoch. Additional training alone is
  therefore unlikely to provide significant improvement.

* None of the models showed signs of overfitting during the 20 training
  epochs, as improvements in training performance were accompanied by improved
  generalization.

Conclusion:

* Increasing network width successfully addressed part of the capacity
  limitation identified in the previous experiment, resulting in both lower
  training loss and higher test accuracy.

* Although wider models continue to improve performance, the diminishing
  returns suggest that simply adding more feature channels may become less
  effective. Future experiments should investigate increasing network depth to
  determine whether additional hierarchical feature extraction yields further
  improvements.
"""