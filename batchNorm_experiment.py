from pathlib import Path
from datasets import load_dataset

from data.dataset import CIFAR10Dataset
from data.transforms import basic_transform

from training.train import create_loader, train, set_seed
from training.eval import evaluate, plot_learning_curves

from models.plain_model import PlainCNN
from models.batchNorm_model import BatchNormCNN


MODELS = {
    "baseline": PlainCNN,
    "batchnorm": BatchNormCNN,
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

        model = Model()

        print(f"Currently training {name}:") 
        history = train(train_loader, model, track=True)
        print(history)
        histories[name] = history

        loss, acc = evaluate(test_loader, model)
        results.append((name, loss, acc))


    print("\nExperiment: Batch Normalization")
    print("=" * 40)
    print("Report")
    print("-" * 40)

    for name, loss, acc in results:
        print(F"{name:<10} {loss:.3f} {acc*100:5.2f}%")

    plot_learning_curves(histories)

if __name__ == "__main__":
    main()


"""
Results from terminal (20 epochs):

baseline:
[1.9558605278849297, 1.7079807320214293, 1.6149326828129762, 1.5302255193290808, 1.4780191913285219, 1.428949769957901, 1.38857613606831, 1.3532572077668232, 1.3170271162181864, 1.289755463142834, 1.2702433852588428, 1.2456624314303288, 1.2270073006525064, 1.2121269830962276, 1.1937304258803882, 1.181621731394697, 1.1701377254465353, 1.1613203398406964, 1.1470730297858147, 1.1386909553461977]
batchnorm:
[1.6813055404921626, 1.3891060000185467, 1.2757134767596985, 1.2200636016895703, 1.179351651607572, 1.1438715883990382, 1.1185693894810689, 1.0961192817334324, 1.0795675060328316, 1.0699033375133944, 1.0509279735405426, 1.0431835235232283, 1.033439100703315, 1.0228139560698244, 1.0130353848952467, 1.010277665408371, 1.001176499878354, 0.9934336767934472, 0.9856832296875737, 0.9838415722712837]

** Also look at the curves plotted.

Experiment: Batch Normalization
========================================
Report
----------------------------------------
baseline   1.185 57.16%
batchnorm  1.091 60.53%

Summary:
Introducing Batch Normalization into every convolutional block improved
optimization. Compared to the baseline model, the BatchNorm variant achieved
a lower training loss throughout training and increased the test accuracy by
approximately 3.4 percentage points.

Observations:

* The BatchNorm model converged noticeably faster during the first few epochs,
  indicating more stable optimization and improved gradient flow.

* Throughout training, the BatchNorm model maintained a consistently lower
  training loss than the baseline. The gap established early in training
  persisted until the final epoch.

* Both learning curves began to flatten after approximately 17 epochs,
  suggesting diminishing returns from additional training alone.

Conclusion:

* Batch Normalization provides a clear improvement over the baseline CNN and
  should be retained in subsequent experiments.

* With optimization improved, the primary limitation of the current model now
  appears to be its limited capacity. Future experiments should investigate
  architectural changes such as increasing the network width or depth while
  keeping Batch Normalization in place.

"""