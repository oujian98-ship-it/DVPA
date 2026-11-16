"""Explicit 64-public/59936-client FMNIST training split, full 10000 test.

The public subset is selected by indices before the label-based client split.
Public membership is fixed across client-neighbor comparisons. These 64
records are deliberately outside the protected client population.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from .dvpa_public_proxy import PublicCurvatureProxy as LegacyProxy,curvature_traces,tensor_sha256

ROOT=Path(__file__).resolve().parents[1]
SPLIT_SEED=20261106


def validate_public_train_config(base):
    if base.get('public_train_count')!=64:raise ValueError('Explicit public_train_count=64 required')
    keys=('public_train_file','public_train_sha256')
    if not all(k in base for k in keys):raise ValueError('Public training package and SHA256 required together')
    if (base['dataset']!='fmnist' or base.get('dataset_mode','official')!='official'
        or base['method']!='JOINT' or base['mechanism']!='gaussian'
        or 'evaluation_public_file' in base or 'evaluation_public_sha256' in base):
        raise ValueError('Train-public FMNIST Gaussian DVPA cannot mix test-public or synthetic protocols')
    path=base['public_train_file'];sha=base['public_train_sha256']
    if (not isinstance(path,str) or not path or not (ROOT/path).resolve().is_relative_to(ROOT)
        or not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
        raise ValueError('Pin an inside-project training-public package and SHA256')


def make_package(path,data_root):
    """Read ONLY official train IDX files; never open test data to build a proxy."""
    from .data import idx,normalize_images
    folder=Path(data_root)/'fashion-mnist-master/data/fashion'
    image=folder/'train-images-idx3-ubyte.gz';label=folder/'train-labels-idx1-ubyte.gz'
    x=normalize_images(idx(image,16).reshape(-1,1,28,28))*2-1
    y=torch.from_numpy(idx(label,8)).long()
    if len(x)!=60000 or len(y)!=60000:raise ValueError('Official 60000 training examples required')
    public=np.random.default_rng(SPLIT_SEED).permutation(60000)[:64]
    mask=np.ones(60000,dtype=bool);mask[public]=False
    private=np.flatnonzero(mask)
    value=dict(images=x[public],labels=y[public],public_source_indices=torch.tensor(public,dtype=torch.long),
               private_source_indices=torch.tensor(private,dtype=torch.long),
               indices=dict(calibration=torch.arange(64),validation=torch.empty(0,dtype=torch.long),confirmation=torch.empty(0,dtype=torch.long)),
               metadata=dict(dataset='FMNIST official training split',protocol='train_public_64_clients_59936_test_10000',
                             split_seed=SPLIT_SEED,public_examples=64,private_training_examples=59936,
                             final_evaluation_examples=10000,public_records_are_not_client_private=True,
                             independent_validation_examples=0,
                             historical_disclosure='936 additional records were public in earlier experiments; this run does not query them as public auxiliary data',
                             source_train_images_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                             source_train_labels_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                             input_sha256=tensor_sha256(x[public]),official_test_used_for_public_data=False))
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
    return value


def read_package(base):
    validate_public_train_config(base)
    path=(ROOT/base['public_train_file']).resolve()
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=base['public_train_sha256']:raise ValueError('Training-public artifact hash mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    ids=data['public_source_indices'];private=data['private_source_indices'];m=data['metadata']
    expected=np.random.default_rng(SPLIT_SEED).permutation(60000)[:64]
    if (m['dataset']!='FMNIST official training split' or m['split_seed']!=SPLIT_SEED
        or m.get('protocol')!='train_public_64_clients_59936_test_10000'
        or m.get('public_examples')!=64 or m.get('private_training_examples')!=59936
        or m.get('final_evaluation_examples')!=10000 or m.get('independent_validation_examples')!=0
        or m['official_test_used_for_public_data'] is not False or m['public_records_are_not_client_private'] is not True
        or ids.dtype!=torch.int64 or private.dtype!=torch.int64 or len(ids)!=64 or len(private)!=59936
        or ids.tolist()!=expected.tolist()
        or sorted(ids.tolist()+private.tolist())!=list(range(60000))
        or private.tolist()!=sorted(private.tolist()) or data['images'].shape!=(64,1,28,28)
        or not torch.isfinite(data['images']).all() or data['labels'].shape!=(64,)
        or data['indices']['calibration'].tolist()!=list(range(64))
        or data['indices']['validation'].tolist()!=[]
        or data['indices']['confirmation'].tolist()!=[]
        or tensor_sha256(data['images'])!=m['input_sha256']):
        raise ValueError('Train-public split is not fixed, disjoint, and complete')
    return path,sha,data


def remove_public_training_examples(base,train):
    _,_,data=read_package(base)
    if len(train)!=60000:raise ValueError('Reserve public data before partitioning the full training set')
    ids=data['public_source_indices'];private=data['private_source_indices']
    if not torch.equal(train.tensors[0][ids],data['images']) or not torch.equal(train.tensors[1][ids],data['labels']):
        raise ValueError('Public package differs from actual official training data')
    return TensorDataset(train.tensors[0][private].clone(),train.tensors[1][private].clone())


def partition_record(base):
    _,sha,data=read_package(base)
    return dict(protocol=data['metadata']['protocol'],public_data_sha256=sha,
                split_seed=SPLIT_SEED,source_training_examples=60000,public_examples=64,
                client_training_examples=59936,curvature_examples=64,
                proxy_validation_examples=0,proxy_confirmation_examples=0,
                public_source_indices=data['public_source_indices'].tolist(),
                private_source_indices=data['private_source_indices'].tolist(),
                protected_population='59936 records in fixed client slots; public64 excluded')


def evaluation_record(base,sha):
    return dict(dataset='FMNIST',public_examples=64,public_source='official_training_split',
                client_training_examples=59936,final_evaluation_examples=10000,source_test_examples=10000,
                public_data_sha256=sha,official_test_used_for_public_data=False,
                original_test_has_historical_evaluations=True,
                scope='All 10000 official test examples are evaluation only in this protocol; historical evaluations exist')


class PublicCurvatureProxy(LegacyProxy):
    def __init__(self,config):
        self.cfg=config['feedback'];base=config['base']
        self.path,self.sha,data=read_package(base)
        if (self.cfg['public_proxy_file']!=base['public_train_file']
            or self.cfg['public_proxy_sha256']!=self.sha):
            raise ValueError('Feedback must use the same fixed training-public package')
        self.images=data['images'][data['indices']['calibration']].clone()
        self.final_indices=list(range(10000));self.metadata=data['metadata']


def verify_validation(config,proof=None):
    """Require correctness checks on these 64 inputs, not an independent holdout."""
    from .dvpa_forecast import parameters_from_config
    f=config['feedback'];proxy=PublicCurvatureProxy(config)
    if proof is None:proof=json.loads((ROOT/f['public_proxy_validation']).read_text(encoding='utf-8'))
    expected={'split_isolation','finite_positive_curvature','scale_preserved','forecast_causal','no_test_data_for_validation'}
    if (proof.get('status')!='completed' or proof.get('validation_kind')!='public64_only_fixed_protocol_check'
        or proof.get('public_data_sha256')!=proxy.sha
        or proof.get('parameters')!=parameters_from_config(f)
        or proof.get('probes')!=f['public_proxy_probes']
        or proof.get('signal_settings')!={k:f[k] for k in ('lambda0','lambda1','signal_scale')}
        or proof.get('independent_validation_examples')!=0
        or proof.get('official_test_examples_queried')!=0
        or proof.get('auxiliary_examples')!=64
        or set(proof.get('required_checks',{}))!=expected
        or not all(proof['required_checks'].values())
        or proof.get('public_controlled_check_completed') is not True):
        raise ValueError('Current public64-only numerical/controlled check required')
    for name in ['src/dvpa_public64.py','src/dvpa_forecast.py','src/dvpa_public_proxy.py','scripts/run_train_public64.py']:
        if proof.get('source_sha256',{}).get(name)!=hashlib.sha256((ROOT/name).read_bytes()).hexdigest():
            raise ValueError('Public64 check source hash changed')
    for name,sha in proof.get('history_manifest',{}).items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Published DP history used for public64 checks changed')
    checks=proof.get('checks',[])
    if (len(checks)!=6 or sorted({v['round'] for v in checks})!=[4,13,21]
        or any(v['split']!='same_public64' or v['examples']!=64 or v['draws']!=64 for v in checks)):
        raise ValueError('Public64 controlled check coverage mismatch')
    return proof


