import torch
import torch.nn as nn
import torch.nn.functional as F


class FmnistAndMnistNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class AdultIncomeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AdultIncomeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(dataset_name, batch_size):
    if dataset_name in ['fmnist', 'mnist']:
        return FmnistAndMnistNet(input_channels=1, num_classes=10)
    elif dataset_name == 'adult_income':
        input_size = 14
        return AdultIncomeNet(input_size=input_size, num_classes=2)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")