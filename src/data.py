import gzip
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset

def normalize_images(images):
    # Avoid CPU Tensor.div_: the installed torch/OpenMP runtime can execute
    # an in-place division repeatedly over overlapping chunks with >1 thread.
    normalized = images.astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(normalized)

def idx(path, offset):
    path = Path(path)
    if not path.exists() and path.suffix == '.gz':
        path = path.with_suffix('')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset).copy()

def load_mnist_split(root, name, split):
    if name not in {'fmnist', 'mnist'} or split not in {'train', 'test'}:
        raise ValueError('Expected MNIST/FMNIST and a single official split')
    root = Path(root)
    folder = root/'fashion-mnist-master'/'data'/('fashion' if name == 'fmnist' else 'mnist')
    if name == 'mnist' and not any(folder.glob('train-images-idx3-ubyte*')):
        folder = root/'MNIST'/'raw'
    prefix = 'train' if split == 'train' else 't10k'
    x = idx(folder/f'{prefix}-images-idx3-ubyte.gz', 16).reshape(-1, 1, 28, 28)
    y = idx(folder/f'{prefix}-labels-idx1-ubyte.gz', 8)
    return TensorDataset(normalize_images(x), torch.from_numpy(y).long())

def load_data(config):
    root = Path(config.get('root', 'D:/data'))
    name = config['dataset']
    if name in {'fmnist', 'mnist'}:
        train, test = (load_mnist_split(root, name, split).tensors for split in ('train', 'test'))
    elif name == 'cifar10':
        folder = root/'cifar-10-python'/'cifar-10-batches-py'
        def read(files):
            chunks = []
            for file in files:
                with (folder/file).open('rb') as f:
                    chunks.append(pickle.load(f, encoding='bytes'))
            x = np.concatenate([c[b'data'] for c in chunks]).reshape(-1, 3, 32, 32)
            y = np.concatenate([c[b'labels'] for c in chunks])
            return normalize_images(x), torch.from_numpy(y).long()
        train = read([f'data_batch_{i}' for i in range(1, 6)])
        test = read(['test_batch'])
    else:
        raise ValueError('Local loader supports fmnist, mnist and cifar10; no automatic downloads')
    limits = config.get('limit_train'), config.get('limit_test')
    result = []
    for pair, limit in zip([train, test], limits):
        result.append(TensorDataset(*(v[:limit] if limit else v for v in pair)))
    return result

def dirichlet_split(labels, clients, alpha, seed):
    if alpha <= 0 or clients > len(labels) or clients < 1:
        raise ValueError('Invalid partition parameters')
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(clients)]
    labels = np.asarray(labels)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        chunks = np.split(indices, (np.cumsum(rng.dirichlet([alpha]*clients))[:-1]*len(indices)).astype(int))
        for part, chunk in zip(parts, chunks):
            part.extend(chunk.tolist())
    # Deterministic empty-client repair; document this departure from pure Dirichlet.
    for part in parts:
        if not part:
            donor = max(parts, key=len)
            part.append(donor.pop())
    return parts

def split_checksum(parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()
