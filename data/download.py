"""
Download and cache the CIFAR-10 dataset locally.

Run this script once before training to download the Hugging Face
CIFAR-10 dataset into the project's local cache directory. Future calls
to `load_dataset` using the same cache location will reuse the cached
files instead of downloading them again.

Ideally we would download the dataset from the official source, university of
Toronto, but since the server is too slow, this alternative has been used for
a more reliable download experience.
"""

from pathlib import Path
from datasets import load_dataset


# Store the dataset inside the project/data directory so all scripts use the
# same local cache.
data_folder = Path(__file__).resolve().parent
cache_dir = data_folder / "hf_cache"

print("Downloading CIFAR-10...")
dataset = load_dataset(
    "uoft-cs/cifar10",
    cache_dir=str(cache_dir),
)
print(f"Dataset cached in:\n{cache_dir}")