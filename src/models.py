import torch
from torch import nn

class PrototypeCNN(nn.Module):
    """Explicit prototype; not claimed to be the authors' exact architecture."""
    def __init__(self, channels=1, classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channels, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d((4, 4)))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(512, 64), nn.ReLU())
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        return self.fc2(self.fc1(self.conv2(self.conv1(x))))

class FedAvgCNN(nn.Module):
    """McMahan et al. 2017 Section 3, 1,663,370 parameters for 28x28.

    Kiani et al. cite this architecture; their exact initialization and
    preprocessing remain unverified. Softmax is supplied by cross entropy.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(7*7*64, 512), nn.ReLU())
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc2(self.fc1(self.conv2(self.conv1(x))))

def build_model(config):
    if config['model']['name'] == 'upstream_cnn':
        from .upstream_runner import source_model
        return source_model()('fmnist',config['train']['batch_size'])
    channels = 3 if config['data']['dataset'].startswith('cifar') else 1
    classes = 100 if config['data']['dataset'] == 'cifar100' else 10
    if config['model']['name'] == 'prototype_cnn':
        return PrototypeCNN(channels, classes)
    if config['model']['name'] == 'fedavg_cnn':
        if channels != 1:
            raise ValueError('Reference FedAvg CNN requires 28x28 grayscale input')
        return FedAvgCNN()
    if config['model']['name'] == 'resnet18_gn':
        from torchvision.models import resnet18
        return resnet18(num_classes=classes, norm_layer=lambda c: nn.GroupNorm( min(32, c), c))
    raise ValueError('Unknown model; exact paper architecture is not available')
