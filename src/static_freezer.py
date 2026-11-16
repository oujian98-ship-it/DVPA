import numpy as np

class StaticFreezer:
    def __init__(self, names, config, seed=42):
        self.active = dict.fromkeys(names, True)
        self.names, self.config = list(names), config
        self.order = list(names)
        mode = config.get('layer_freeze', 'none')
        if mode == 'static_top_down':
            self.order.reverse()
        elif mode == 'static_random':
            np.random.default_rng(config.get('mask_seed', seed)).shuffle(self.order)
        elif mode not in {'none', 'static_bottom_up', 'dp_feedback'}:
            raise ValueError('Unsupported freeze mode; exact SAME-DIM needs compatible groups')

    def decide(self, t):
        if self.config.get('layer_freeze', 'none') not in {'none', 'dp_feedback'}:
            for g, when in zip(self.order, self.config.get('freeze_rounds', [])):
                if when is not None and t >= when and sum(self.active.values()) > 1:
                    self.active[g] = False
        return self.active.copy()
