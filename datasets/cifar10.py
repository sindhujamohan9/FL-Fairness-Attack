import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np

def get_cifar10(subset_fraction=1.0):
    # Specify a local directory (relative to your project or home directory) with write permissions
    path = "./data"  # Use a local subdirectory to store the dataset
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists

    # Define any transformations for training and testing
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    # Load full training and testing datasets
    train = CIFAR10(root=path, train=True, transform=train_transform, download=True)
    test = CIFAR10(root=path, train=False, transform=test_transform, download=True)

    # Reduce dataset size based on subset_fraction
    if subset_fraction < 1.0:
        train_indices = np.random.choice(len(train), int(len(train) * subset_fraction), replace=False)
        test_indices = np.random.choice(len(test), int(len(test) * subset_fraction), replace=False)
        train = Subset(train, train_indices)
        test = Subset(test, test_indices)

    return train, test




# from os.path import isdir
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
# import os

# def get_cifar10():
#     # Specify a local directory (relative to your project or home directory) with write permissions
#     path = "./data"  # Use a local subdirectory to store the dataset
#     os.makedirs(path, exist_ok=True)  # Ensure the directory exists

#     # Define any transformations for training and testing
#     train_transform = transforms.Compose([transforms.ToTensor()])
#     test_transform = transforms.Compose([transforms.ToTensor()])

#     # Load training and testing datasets with download=True to fetch if not available
#     train = CIFAR10(root=path, train=True, transform=train_transform, download=True)
#     test = CIFAR10(root=path, train=False, transform=test_transform, download=True)
#     return train, test


# def get_cifar10(path="/datasets/CIFAR10"):

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ])

#     train = CIFAR10(path, train=True, transform=train_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
#     test = CIFAR10(path, train=False, transform=test_transform, download=(path != "/datasets/CIFAR10") and (not isdir(path)))
#     return train, test