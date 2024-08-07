from typing import Generic, TypeVar
from torch import nn
import torch
from ..utils import copy_model,set_eval


class ModelEma(nn.Module):
    def __init__(self, model:nn.Module, decay=0.9997, dev=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy_model(model)
        set_eval(self.module)
        self.decay = decay
        self.device = dev  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=dev)
            
    @torch.no_grad()
    def _update(self, model:nn.Module, update_fn):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model:nn.Module):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model:nn.Module):
        self._update(model, update_fn=lambda e, m: m)
