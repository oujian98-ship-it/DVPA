"""Public, irreversible block/coordinate masks and local freeze enforcement."""
import math
import numpy as np
import torch

class StaticMask:
    def __init__(self, model, mode, freeze_rounds, seed=42, static_order=None):
        modes={'NF','BU','TD','RAND','SAME-DIM','NO-DP'}
        if mode not in modes:
            raise ValueError(f'Unknown P1 mode: {mode}')
        self.mode=mode
        self.shapes={n:tuple(p.shape) for n,p in model.named_parameters()}
        self.names=list(dict.fromkeys(n.split('.')[0] for n in self.shapes))
        self.dims={g:sum(p.numel() for n,p in model.named_parameters() if n.split('.')[0]==g) for g in self.names}
        self.events=[int(t) for t in freeze_rounds]
        if any(t<1 for t in self.events) or self.events!=sorted(self.events) or len(self.events)>=len(self.names):
            raise ValueError('Freeze rounds must be ordered, positive and leave a trainable block')
        self.order=list(self.names)
        rng=np.random.default_rng(seed)
        if mode=='TD':self.order.reverse()
        if mode=='RAND':rng.shuffle(self.order)
        if static_order is not None:
            if mode!='BU' or not isinstance(static_order,list) or len(static_order)!=len(self.names) or set(static_order)!=set(self.names):
                raise ValueError('Public static order must be a full layer permutation in BU mode')
            self.order=list(static_order)
        if mode=='SAME-DIM' and list(model.named_buffers()):
            raise ValueError('Coordinate SAME-DIM currently requires a buffer-free model')
        # Deep-first public coordinates match BU dimensions exactly, including
        # architectures with no alternate subset of whole blocks of equal size.
        self.coordinate_order=[]
        if mode=='SAME-DIM':
            for g in reversed(self.names):
                for n,shape in self.shapes.items():
                    if n.split('.')[0]==g:
                        self.coordinate_order.append((n,rng.permutation(math.prod(shape))))
        self.previous={n:torch.ones(shape,dtype=torch.bool) for n,shape in self.shapes.items()}
        self.last_round=0

    def decide(self, t):
        if t<self.last_round:
            raise ValueError('Mask decisions must advance monotonically in time')
        masks={n:torch.ones(shape,dtype=torch.bool) for n,shape in self.shapes.items()}
        count=sum(t>=when for when in self.events) if self.mode!='NF' else 0
        if self.mode=='SAME-DIM':
            left=sum(self.dims[g] for g in self.names[:count])
            for n,indices in self.coordinate_order:
                take=min(left,len(indices));masks[n].view(-1)[indices[:take]]=False;left-=take
        else:
            frozen=set(self.order[:count])
            for n in masks:
                if n.split('.')[0] in frozen:masks[n].zero_()
        if any(torch.any(masks[n]&~self.previous[n]) for n in masks):
            raise ValueError('Reactivation is not supported')
        self.previous={n:m.clone() for n,m in masks.items()};self.last_round=t
        return masks

def group_counts(masks):
    result={}
    for n,m in masks.items():
        g=n.split('.')[0];result[g]=result.get(g,0)+int(m.sum())
    return result

def joint_group_clip(parameters, radius, masks, weights=None):
    if not math.isfinite(float(radius)) or radius<0:
        raise ValueError('Invalid total clipping radius')
    counts=group_counts(masks);active=[g for g,d in counts.items() if d]
    if not active:raise ValueError('Empty active set')
    if weights is None:weights={g:(1/len(active) if counts[g] else 0.) for g in counts}
    from .layer_clip import validate
    validate({g:bool(d) for g,d in counts.items()},weights)
    equal_active=all(weights[g]==1/len(active) for g in active)
    for n,v in parameters.items():
        if n not in masks:raise ValueError('Unregistered buffer/parameter in DP update')
        v.masked_fill_(~masks[n].to(v.device),0)
        if not torch.isfinite(v).all():v.zero_()
    for g in active:
        vectors=[v for n,v in parameters.items() if n.split('.')[0]==g]
        norm=math.sqrt(sum(v.double().square().sum().item() for v in vectors))
        group_radius=float(radius)/math.sqrt(len(active)) if equal_active else float(radius)*math.sqrt(weights[g])
        factor=min(1.,group_radius/(norm+1e-12))
        for v in vectors:v.mul_(factor)

def masked_noise(parameters, masks, stdev, generator):
    for n,v in parameters.items():
        mask=masks[n].to(v.device)
        v.masked_fill_(~mask,0)
        count=int(mask.sum())
        if count:
            v[mask]+=torch.randn(count,generator=generator,dtype=v.dtype).to(v.device)*float(stdev)

def install_local_freeze(client, masks):
    """Keep persistent SGD state; restore masked coordinates AFTER decay/momentum.

    A grad hook alone is insufficient when weight decay or momentum is present.
    Whole frozen blocks enter eval again after source Client.train calls train().
    """
    model=client.local_model
    device_masks={n:m.to(dict(model.named_parameters())[n].device) for n,m in masks.items()}
    active=group_counts(masks)
    handles=[]
    partial=[]
    for n,p in model.named_parameters():
        m=device_masks[n]
        any_active=bool(masks[n].any());all_active=bool(masks[n].all())
        p.requires_grad_(any_active);p.grad=None
        if any_active and not all_active:
            partial.append((p,m))
            handles.append(p.register_hook(lambda grad,mask=m:grad*mask))
    def freeze_modules(*args):
        for g,module in model.named_children():
            if not active.get(g,0):module.eval()
    handles.append(model.register_forward_pre_hook(freeze_modules))
    original_step=client.optimizer.step
    def step(*args,**kwargs):
        snapshots=[]
        for p,mask in partial:
            state=client.optimizer.state.get(p,{})
            momentum=state.get('momentum_buffer')
            snapshots.append((p,mask,p.detach()[~mask].clone(),None if momentum is None else momentum[~mask].clone()))
        result=original_step(*args,**kwargs)
        with torch.no_grad():
            for p,mask,values,momentum in snapshots:
                p[~mask]=values
                buffer=client.optimizer.state.get(p,{}).get('momentum_buffer')
                if buffer is not None:buffer[~mask]=0 if momentum is None else momentum
        return result
    step._wrapped_by_lr_sched=getattr(original_step,'_wrapped_by_lr_sched',False)
    # Preserve the scheduler's wrapped step: never replace it with a fresh SGD.
    if partial:client.optimizer.step=step
    def cleanup():
        if partial:client.optimizer.step=original_step
        for h in handles:h.remove()
    return cleanup

def aggregate_active(messages, template, masks, n, q_bar, physical_scale, generator):
    if n<1 or len(messages)>n or not 0<q_bar<=1:
        raise ValueError('Invalid aggregation parameters')
    total={k:torch.zeros_like(v) for k,v in template.items()}
    for message in messages:
        for k in total:total[k]+=message[k]
    masked_noise(total,masks,float(physical_scale)*math.sqrt((n-len(messages))/n),generator)
    return {k:v/(n*q_bar) for k,v in total.items()},(float(physical_scale)/(n*q_bar))**2
