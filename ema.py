import copy
import torch


class EMA:
    def __init__(self, model, decay=0.9999, warmup=True):
        self.ema = copy.deepcopy(model).eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1

        if self.warmup:
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        else:
            decay = self.decay

        # EMA update
        msd = model.state_dict()
        esd = self.ema.state_dict()

        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(decay).add_(msd[k], alpha=1 - decay)
            else:
                esd[k].copy_(msd[k])

    def to(self, device):
        self.ema.to(device)
        return self

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)