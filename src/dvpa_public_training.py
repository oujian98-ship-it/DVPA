"""Route pinned training-public protocols without changing legacy artifacts."""

import hashlib

import numpy as np
import torch
from torch.utils.data import TensorDataset


VALIDATION_SPLIT_SEED = 202611171

def module(base):
    count=base.get('public_train_count',1000)
    if type(count) is not int or count not in (64,1000):
        raise ValueError('public_train_count must be 64 or 1000')
    if count==64:
        from . import dvpa_public64 as selected
    else:
        from . import dvpa_train_public as selected
    return selected

def validate_public_train_config(base):return module(base).validate_public_train_config(base)
def remove_public_training_examples(base,train):return module(base).remove_public_training_examples(base,train)
def split_remaining(base):
    record = module(base).partition_record(base)
    pool = np.asarray(record["private_source_indices"], dtype=np.int64)
    chosen = np.random.default_rng(VALIDATION_SPLIT_SEED).choice(pool, base["validation_count"], replace=False)
    return np.setdiff1d(pool, chosen), np.sort(chosen)


def load_train_validation(base):
    from .data import load_mnist_split

    official = load_mnist_split(base["data_root"], base["dataset"], "train")
    train = TensorDataset(official.tensors[0]*2-1, official.tensors[1])
    # Verify that the pinned auxiliary data actually matches the local dataset.
    remove_public_training_examples(base, train)
    indices = split_remaining(base)
    return tuple(TensorDataset(*(v[torch.from_numpy(ids)] for v in train.tensors)) for ids in indices)


def partition_record(base):
    record = module(base).partition_record(base)
    if "validation_count" in base:
        client, validation = split_remaining(base)
        record.update(protocol="train_public_with_independent_validation_v1",
                      client_training_examples=len(client), private_source_indices=client.tolist(),
                      validation_source_indices=validation.tolist(), validation_split_seed=VALIDATION_SPLIT_SEED,
                      protected_population=f"{len(client)} records in fixed client slots; public and validation excluded")
    return record


def evaluation_record(base, sha):
    if "validation_count" not in base:
        return module(base).evaluation_record(base, sha)
    client, validation = split_remaining(base)
    return dict(dataset=base["dataset"].upper(), evaluation_split="independent_validation",
                evaluation_examples=len(validation), client_training_examples=len(client),
                public_examples=base.get("public_train_count", 1000), public_data_sha256=sha,
                validation_ids_sha256=hashlib.sha256(validation.astype("<i8").tobytes()).hexdigest(),
                official_test_used=False, original_test_has_historical_evaluations=True)


def client_training_count(base):
    return 60000 - (base.get("public_train_count", 1000) if "public_train_file" in base else 0) - base.get("validation_count", 0)
def PublicCurvatureProxy(config):return module(config['base']).PublicCurvatureProxy(config)
def verify_validation(config,proof=None):return module(config['base']).verify_validation(config,proof)
