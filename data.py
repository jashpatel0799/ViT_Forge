"""
data.py

Data loading utilities for curriculum learning across multiple vision datasets.

- All datasets are resized to a common resolution (default: 96x96).
- Grayscale datasets (MNIST, EMNIST, FashionMNIST) are converted to 3 channels to match ViT input.
- Returns PyTorch DataLoaders *and* the number of classes for the active dataset.
"""

from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---- Common transforms -------------------------------------------------------
def build_transforms(image_size: int, grayscale_to_rgb: bool) -> transforms.Compose:
    """
    Build a minimal, consistent transform pipeline for all datasets.
    Grayscale datasets are expanded to 3 channels for ViT compatibility.
    """
    
    t_list = []
    t_list.append(transforms.Resize((image_size, image_size)))
    
    if grayscale_to_rgb:
        t_list.append(transforms.Grayscale(num_output_channels=3))
        
    t_list.append(transforms.ToTensor())
    
    return transforms.Compose(t_list)


# ---- Dataset factory ---------------------------------------------------------
def get_dataset(
    name: str,
    root: str,
    image_size: int,
    train: bool
) -> Tuple[torch.utils.data.Dataset, int]:
    """
    Create a torchvision dataset and return (dataset, num_classes).

    Supported names (case-insensitive):
      - "mnist" (10 classes)
      - "emnist" (47 classes, 'balanced' split)
      - "fashionmnist" (10 classes)
      - "svhn" (10 classes; fixes target 10 -> 0)
      - "cifar10" (10 classes)
      - "cifar100" (100 classes)
      - "stl10" (10 classes; test-only stage at the end)

    Notes:
      - EMNIST uses the 'balanced' split (47 classes).
      - SVHN labels the digit 0 as "10" → remap to 0 with target_transform.
    """
    key = name.strip().lower()
    
    if key == 'mnist':
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=True)
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
        return dataset, 10
    
    elif key == "emnist":
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=True)
        # Using 'balanced' split → 47 classes
        dataset = datasets.EMNIST(root=root, split="balanced", train=train, transform=transform, download=True)
        return dataset, 47
    
    elif key == "fashionmnist":
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=True)
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
        return dataset, 10
    
    elif key == 'svhn':
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=False)
        # SVHN: split="train"/"test"; label '10' means digit '0' → remap to 0
        target_tf = lambda y: 0 if y == 10 else y
        split = "train" if train else "test"
        
        dataset = datasets.SVHN(root=root, split=split, transform=transform, target_transform=target_tf, download=True)
        return dataset, 10
    
    elif key == 'cifar10':
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=False)
        dataset = datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        return dataset, 10
    
    elif key == 'cifar100':
        transform = build_transforms(image_size=image_size, grayscale_to_rgb=False)
        dataset = datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
        return dataset, 100
    
    else:
        raise ValueError(f"Unsupported dataset: {name}")
        
    
# ----Public API---------------------------------------------------------------------------------------
def prepare_dataloader(
    dataset_name: str,
    batch_size: int, 
    image_size: int,
    root: str = "./data",
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build train/test dataloaders *and* return the number of classes.

    Returns:
        train_loader, test_loader, num_classes
    """
    train_set, num_classes = get_dataset(name=dataset_name, root=root, image_size=image_size, train=True)
    test_set, _ = get_dataset(name=dataset_name, root=root, image_size=image_size, train=False)
    
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    
    test_dataloader = DataLoader(test_set, batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    
    return train_dataloader, test_dataloader, num_classes