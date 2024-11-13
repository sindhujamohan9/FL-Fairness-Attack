# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # Input channels = 1 for grayscale images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Same padding
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Input x shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch_size, 32, 28, 28)
        x = self.pool1(x)          # (batch_size, 32, 14, 14)
        x = F.relu(self.conv2(x))  # (batch_size, 64, 14, 14)
        x = self.pool2(x)          # (batch_size, 64, 7, 7)
        x = x.view(-1, 7 * 7 * 64) # Flatten
        x = F.relu(self.fc1(x))    # (batch_size, 1024)
        x = self.fc2(x)            # (batch_size, num_classes)
        return x
