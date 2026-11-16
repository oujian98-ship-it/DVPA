import numpy as np
import torch
from src.data import normalize_images, load_data

def test_normalization_large_multithreaded_tensor():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        x = np.full((60000, 1, 28, 28), 255, dtype=np.uint8)
        x[0, 0, 0, :3] = [0, 128, 255]
        y = normalize_images(x)
        assert y.min() == 0 and y.max() == 1
        assert torch.all(y[1:] == 1)
        assert y[0, 0, 0, 1].item() == np.float32(128/255)
        assert x[1, 0, 0, 0] == 255
    finally:
        torch.set_num_threads(previous)
