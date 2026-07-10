"""
Predefined image transformation pipelines for CIFAR-10 experiments.

This module centralizes all preprocessing and data augmentation
configurations used throughout the project. Keeping the transformations
in a single location ensures that experiments remain consistent and
makes it easy to compare the effect of individual augmentations against
the full training pipeline.
"""

from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
)

# Channel-wise mean and standard deviation computed for the CIFAR-10 dataset.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Baseline preprocessing without data augmentation.
basic_transform = Compose([
    ToTensor(), Normalize(mean=MEAN, std=STD)
])

# Horizontal and vertical flipping only.
flip_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor()
])

# Brightness and contrast augmentation only.
color_transform = Compose([
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor()
])

# Random cropping only.
crop_transform = Compose([
    RandomCrop(size=32, padding=4), ToTensor()
])

# Full augmentation pipeline used during training.
train_transform = Compose([
    RandomCrop(size=32, padding=4),
    RandomHorizontalFlip(p=0.2),
    RandomVerticalFlip(0.2),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])

# Preprocessing applied during evaluation.
test_transform = Compose([
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])