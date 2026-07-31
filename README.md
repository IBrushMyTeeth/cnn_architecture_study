# CIFAR-10 CNN Architecture Experiments

A research-oriented PyTorch project exploring the effect of individual convolutional neural network design choices on image classification performance.

Rather than pursuing state-of-the-art accuracy, this repository focuses on **controlled experimentation**. Every model is trained under the same conditions while changing **one architectural component at a time**, making it possible to observe the direct effect of that design decision.

The project was built as a practical study of CNN architectures, experimental methodology, and reproducible deep learning workflows.

---

# Project Goals

The primary goals of this project are:

- Build a clean and reproducible CNN training pipeline
- Isolate architectural variables through controlled experiments
- Compare common CNN design choices under identical training conditions
- Document both quantitative results and qualitative observations
- Produce readable, well-structured code suitable for learning and extension

Instead of asking *"Which model performs best?"*, the project asks:

> **"What happens when only this one component changes?"**

---

# Experimental Philosophy

Every experiment attempts to isolate **exactly one variable**.

Examples include:

- Network width
- Network depth
- Data augmentation
- Batch Normalization
- Dropout
- Progressive channel scaling
- Optimizer comparison

Everything else remains unchanged:

- Dataset
- Learning rate
- Batch size
- Number of epochs
- Random seed
- Classifier
- Evaluation procedure

This allows observed performance differences to be attributed primarily to the architectural change being investigated.

Each experiment concludes with a short report discussing:

- Results
- Interpretation
- Limitations
- Personal observations

rather than simply presenting accuracy numbers.

---

# Project Structure

## Dataset Pipeline

The project includes a complete dataset pipeline built around the Hugging Face CIFAR-10 dataset.

Features include:

- Automatic dataset download
- Local dataset caching
- Simple loading scripts
- PyTorch Dataset wrapper

```text
HuggingFace Dataset
        │
        ▼
CIFAR10Dataset Wrapper
        │
        ▼
Transforms
        │
        ▼
PyTorch DataLoader
```

The wrapper provides a clean interface while remaining fully compatible with standard PyTorch workflows.

---

## Standardized Transformations

All preprocessing and augmentation pipelines are centralized in a single module.

Examples include:

- Basic normalization
- Random horizontal flips
- Color jitter
- Random cropping
- Combined augmentation pipelines

Centralizing transformations ensures that:

- experiments remain reproducible
- augmentations are easy to compare
- preprocessing stays consistent across every model

---

## Modular CNN Blocks

Rather than rewriting entire models, the repository is built from reusable building blocks.

Examples include:

- Minimal convolution block
- BatchNorm convolution block
- Dropout variants

Entire architectures are constructed by combining these reusable modules.

This makes it straightforward to compare architectural ideas while minimizing duplicated code.

---

## Shared Classifier Head

Every architecture uses the same lightweight classifier:

```
Adaptive Average Pool
        ↓
Flatten
        ↓
Linear
        ↓
ReLU
        ↓
Dropout (optional)
        ↓
Output Layer
```

The classifier intentionally contains relatively little capacity.

This encourages the convolutional feature extractor—not the fully connected layers—to be responsible for learning useful representations.

Keeping the classifier fixed across experiments also prevents it from becoming an additional confounding variable.

---

## Reproducible Training Pipeline

Every experiment uses the same training utilities.

The project includes:

- deterministic seeding
- reusable training loop
- evaluation utilities
- plotting utilities
- configurable dataclass-based training configuration
- standardized DataLoaders

A shared configuration object ensures that every experiment is trained under identical conditions unless the experiment explicitly changes one parameter.

This greatly improves reproducibility and fairness between experiments.

---

# Experiments

The repository currently investigates topics including:

- CNN width scaling
- CNN depth scaling
- Batch Normalization
- Dropout
- Progressive channel scaling
- Data augmentation strategies
- Optimizer choice

Each experiment includes:

- dedicated training script
- learning curves
- validation accuracy
- observations
- discussion and conclusions

---

# Deliberately Excluded Experiments

Not every popular CNN technique is included.

Some ideas were intentionally excluded because they would not produce meaningful conclusions within the scope of this project.

## Residual Connections

Residual blocks are designed to improve optimization in **very deep networks**.

The architectures explored here are relatively shallow, meaning residual connections would provide little practical benefit while adding unnecessary complexity.

Because the goal is to isolate meaningful architectural effects, residual networks were intentionally omitted.

---

## Larger Convolution Kernels

Experiments comparing larger kernels (for example 5×5 or 7×7 convolutions) were also excluded.

CIFAR-10 images are only **32×32 pixels**.

Even relatively shallow CNNs quickly obtain an effective receptive field covering most of the image, meaning larger kernels provide little additional spatial information while increasing parameter count and computational cost.

For this reason, receptive field experiments were not expected to yield particularly informative results.

---

# Technologies

- Python
- PyTorch
- TorchVision
- Hugging Face Datasets
- Matplotlib
- NumPy

---

# Repository Highlights

- Clean modular architecture
- Reusable CNN building blocks
- Standardized preprocessing
- Reproducible training pipeline
- Configuration-driven experiments
- Controlled variable isolation
- Well-documented code
- Experiment reports with discussion and conclusions

---

# Example Workflow

```
Download Dataset
        │
        ▼
Cache Dataset Locally
        │
        ▼
Apply Standardized Transform
        │
        ▼
Create DataLoaders
        │
        ▼
Train Model
        │
        ▼
Evaluate Performance
        │
        ▼
Plot Learning Curves
        │
        ▼
Analyze Results
```

---

# Future Work

Possible future extensions include:

- Learning rate scheduler comparisons
- Transfer learning experiments
- Larger image datasets
- Deeper architectures where residual learning becomes meaningful

These additions would naturally extend the existing experimental framework while maintaining the project's emphasis on controlled, reproducible comparisons.

---

## Final Remarks

This project is not intended to produce the highest possible CIFAR-10 accuracy.

Its purpose is to build a **reproducible experimental framework** for understanding how individual CNN design choices influence optimization and classification performance. By changing one variable at a time and keeping all other training conditions fixed, the repository aims to provide clear, interpretable insights into convolutional neural network design rather than simply reporting benchmark results.