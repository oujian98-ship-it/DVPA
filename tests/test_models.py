import torch
from src.models import FedAvgCNN
from src.layer_registry import LayerRegistry

def test_reference_cnn_matches_published_parameter_count():
    model=FedAvgCNN()
    assert sum(p.numel() for p in model.parameters()) == 1663370
    assert model(torch.randn(2,1,28,28)).shape == (2,10)
    assert list(LayerRegistry(model).dimensions) == ['conv1','conv2','fc1','fc2']
