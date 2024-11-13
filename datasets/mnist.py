# datasets.py (formerly cifar10.py)
import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np

def get_mnist(subset_fraction=0.001):
    path = "./data"
    os.makedirs(path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
    ])
    
    # Load the datasets
    train = MNIST(root=path, train=True, transform=transform, download=True)
    test = MNIST(root=path, train=False, transform=transform, download=True)
    
    # Optionally reduce dataset size
    if subset_fraction < 1.0:
        train_indices = np.random.choice(len(train), int(len(train) * subset_fraction), replace=False)
        test_indices = np.random.choice(len(test), int(len(test) * subset_fraction), replace=False)
        train = Subset(train, train_indices)
        test = Subset(test, test_indices)
    
    return train, test
