import torch
import torchvision
from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


train_dataloader = CIFAR10(root = "./data", train=True, transform=transform, download=True)
test_dataloader = CIFAR10(root = "./data", train=False, transform=transform, download=True)