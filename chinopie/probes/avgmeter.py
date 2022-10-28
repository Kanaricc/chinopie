from torch import Tensor
import torch
from torch.types import Number

class NumericMeter:
    def __init__(self,name: str) -> None:
        self._name=name
        self._list=torch.zeros(0)
    
    def update(self, val:Tensor):
        self._list=torch.cat([self._list,val.detach().cpu().flatten()])
    
    @property
    def val(self):
        return self._list

class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self._val = 0
        self._sum = 0
        self._cnt = 0
        self._avg=0
        pass

    def update(self, val: Number, n=1):
        self._val = val
        self._sum += val*n
        self._cnt += n
        self._avg = self._sum/self._cnt
    
    def has_data(self):
        return self._cnt!=0

    def average(self) -> float:
        return self._avg

    def value(self) -> Number:
        return self._val
    
    def reset(self):
        self._val=0
        self._sum=0
        self._cnt=0
        self._avg=0
        
    def __str__(self) -> str:
        return f"{self.name}: {self.value():.5f}(avg {self.average():.5f})"
