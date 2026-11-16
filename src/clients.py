import copy
import torch
from torch.utils.data import DataLoader, Subset

def local_update(global_model, registry, dataset, indices, active, train, seed, device, lr):
    local = copy.deepcopy(global_model).to(device)
    local.train()
    registry.freeze(local, active)
    before = registry.pack(global_model)
    optimizer = torch.optim.SGD([p for p in local.parameters() if p.requires_grad], lr=lr, momentum=train['momentum'])
    loader = DataLoader(Subset(dataset, indices), batch_size=train['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    semantics = train.get('local_semantics', 'epochs')
    if semantics not in {'epochs', 'steps'}:
        raise ValueError('local_semantics must be epochs or steps')
    steps = 0
    for _ in range(train['local_epochs_or_iterations']):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(local(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            steps += 1
            if semantics == 'steps' and steps >= train['local_epochs_or_iterations']:
                break
        if semantics == 'steps' and steps >= train['local_epochs_or_iterations']:
            break
    after = registry.pack(local)
    return {g: after[g]-before[g] for g in before}
