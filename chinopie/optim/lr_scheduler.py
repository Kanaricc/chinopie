from typing import Callable, List, Union
import torch
from torch.optim.lr_scheduler import _LRScheduler,LambdaLR
from torch.optim import Optimizer

class ConstantWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, successor: _LRScheduler, warmup_epoch:int, const_lr_factor:float, last_epoch: int = -1, verbose: bool = False) -> None:
        self.successor=successor
        self.warmup_epoch=warmup_epoch
        self.const_lr_factor=const_lr_factor
        super().__init__(optimizer, last_epoch, verbose)
    
    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr=self.successor.get_last_lr()
        else:
            super().step(epoch)
    
    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.const_lr_factor * base for base in self.base_lrs]

class LinearWarmupScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1, verbose: bool = False) -> None:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)
