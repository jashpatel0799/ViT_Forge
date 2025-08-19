"""
data.py

Dataset preparation and utilities for CIFAR-10 traininh and evalution
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




def prepare_dataloader(args):
    """
    Prepare training and testing dataloader for CIFAR-10 dataset.
    
    Args:
        args(dict):
            - input_image_size(int): Target Image size (resize applied)
            - batch_size (int): Nmumber of sample per batch.
            
        Return:
            tuple: (train_dataloader, test_dataloader)
                - train_dataloader (torch.data.utils.DataLoader)
                - test_dataloader (torch.data.utils.DataLoader)
            
        Example:
            >>> args = {
                input_image_size = 32,
                batch_szie = 64
            }
            >>> train_loader, test_loader = prepare_dataloader(args)
        
    """
    # ----------------------------------------------------------------------------------------
    # Data preprocessing pipeline
    # ----------------------------------------------------------------------------------------
    # Best practice: Keep transformers minimal for CIFAR-10 baseline,
    # later augment (RandomCrop, RandomFlip) for better generalization
    transform = transforms.Compose([
                    transforms.Resize(args['input_image_size']),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),   # CIFAR-10 channel means
                        std=(0.2023, 0.1994, 0.2010)     # CIFAR-10 channel stds
                    )
                ])
    
    # ----------------------------------------------------------------------------------------
    # Dataset Loading
    # ----------------------------------------------------------------------------------------
    
    
    train_dataset = datasets.CIFAR10(
        root = "./data", 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root = "./data", 
        train=False, 
        transform=transform, 
        download=True
    )
    
    
    # ------------------------------------------------------------------------------------------
    # Dataloader Preparation
    # ------------------------------------------------------------------------------------------
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = args['batch_size'], 
        shuffle = True, 
        drop_last = True
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = args['batch_size'], 
        shuffle = False, 
        drop_last = False
    )
    
    return train_dataloader, test_dataloader