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

def prepare_dataloader(args):
    
    train_dataset = CIFAR10(root = "./data", train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root = "./data", train=False, transform=transform, download=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
    
    return train_dataloader, test_dataloader