import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    loss, correct = 0., 0
    for x, y in DataLoader(dataset, batch_size=512):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
    return loss/len(dataset), 100*correct/len(dataset)
