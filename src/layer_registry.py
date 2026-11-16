import torch

class LayerRegistry:
    def __init__(self, model):
        self.names = list(dict.fromkeys(n.split('.')[0] for n, _ in model.named_parameters()))
        self.parameters = {g: [n for n, _ in model.named_parameters() if n.split('.')[0] == g] for g in self.names}
        self.dimensions = {g: sum(p.numel() for n, p in model.named_parameters() if n in self.parameters[g]) for g in self.names}

    def pack(self, model):
        params = dict(model.named_parameters())
        return {g: torch.cat([params[n].detach().flatten() for n in self.parameters[g]]) for g in self.names}

    def apply(self, model, delta):
        with torch.no_grad():
            params = dict(model.named_parameters())
            for g, vector in delta.items():
                offset = 0
                for n in self.parameters[g]:
                    p = params[n]
                    p.add_(vector[offset:offset+p.numel()].view_as(p))
                    offset += p.numel()

    def freeze(self, model, active):
        for name, p in model.named_parameters():
            p.requires_grad_(active[name.split('.')[0]])
            p.grad = None
        # Entire group is in eval mode: includes BN buffers and dropout.
        for name, module in model.named_children():
            if name in active and not active[name]:
                module.eval()
