from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from typing import Callable
from PIL import Image
import torch
from torchvision.transforms import ToTensor


Transform = Callable[[Image.Image], torch.Tensor]


class CIFAR10Dataset(Dataset):
    """
    PyTorch Dataset wrapper for the Hugging Face CIFAR-10 dataset.

    Parameters
    ----------
    hf_dataset : HFDataset
        Hugging Face dataset split (e.g. training or test).
    transform : Transform | None
        Transformation applied to every image before it is returned.
        If no transform is provided, the image is converted to a tensor
        using torchvision.transforms.ToTensor().
    """
    def __init__(
        self,
        hf_dataset: HFDataset,
        transform: Transform | None
    ):
        self.dataset = hf_dataset
        self.transform = transform or ToTensor()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieve a single sample from the dataset.
        Applies a transformation (the default is ToTensor) and returns a tuple
        (torch.Tensor, int).
        """
        sample = self.dataset[idx]

        # Extract the image and class label from the Hugging Face sample.
        image = sample["img"]
        label = sample["label"]

        image = self.transform(image)

        return image, label