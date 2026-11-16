"""Offline diagnostics on the public FMNIST benchmark, never DP telemetry.

Reproduces the local runtime normalization failure and measures model/noise
scales at initialization. Does not feed raw statistics to the training loop.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from src.data import load_data, dirichlet_split
from src.models import build_model, FedAvgCNN
from src.layer_registry import LayerRegistry
from src.clients import local_update
from src.original_scheduler import PrivacyScheduler

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source=np.full((60000,1,28,28),255,dtype=np.uint8)
    broken=torch.from_numpy(source).float().div_(255)
    report={'scope':'offline public FMNIST only', 'normalization_expected_max':1.,
            'legacy_inplace_min':broken.min().item(),'legacy_inplace_max':broken.max().item()}
    del source,broken
    c=yaml.safe_load(Path('configs/original_fmnist.yaml').read_text())
    train,test=load_data(c['data'])
    report['legacy_parallel_reduction_mean']=train.tensors[0].mean().item()
    torch.set_num_threads(1)
    report.update(fixed_pixel_min=train.tensors[0].min().item(),fixed_pixel_max=train.tensors[0].max().item(),
                  fixed_pixel_mean=float(train.tensors[0].numpy().mean()),train_samples=len(train),test_samples=len(test),
                  label_counts=torch.bincount(train.tensors[1]).tolist())
    parts=dirichlet_split(train.tensors[1],100,.1,42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(c).to(device);reg=LayerRegistry(model)
    delta=local_update(model,reg,train,parts[0],dict.fromkeys(reg.names,True),c['train'],42,device,.001)
    report['initial_model_norm']=sum(v.square().sum() for v in reg.pack(model).values()).sqrt().item()
    report['example_local_update_norm']=sum(v.square().sum() for v in delta.values()).sqrt().item()
    rp=PrivacyScheduler(c).params(1)
    std=250*rp.sigma_global/(100*rp.q_bar)
    report.update(original_first_round_noise_std=std,original_first_round_noise_norm_rms=std*np.sqrt(sum(reg.dimensions.values())))
    torch.manual_seed(42)
    reference=FedAvgCNN()
    report['reference_cnn_parameters']=sum(p.numel() for p in reference.parameters())
    report['reference_cnn_initial_norm']=float(sum(p.detach().square().sum() for p in reference.parameters()).sqrt())
    report['reference_cnn_noise_norm_rms']=std*np.sqrt(report['reference_cnn_parameters'])
    Path('reports/p0_diagnosis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
