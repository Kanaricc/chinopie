from torch import Tensor
import torch
from torch.types import Number

class DistributionMeter:
    def __init__(self,name: str) -> None:
        self._name=name
        self._max=None
        self._min=None
        self._sum=None
        self._cnt=0
    
    def update(self, val:Tensor):
        if self._max==None:
            self._max=val
        if self._min==None:
            self._min=val
        if self._sum==None:
            self._sum=torch.zeros_like(val,dtype=torch.float)
        
        self._max=torch.max(self._max,val)
        self._min=torch.min(self._min,val)
        self._sum+=val
        self._cnt+=1
    
    @property
    def max(self):
        assert self._max!=None, f"{self._name} is not updated: {self._max}"
        return self._max
    @property
    def min(self):
        assert self._min!=None, f"{self._name} is not updated: {self._min}"
        return self._min
    @property
    def avg(self):
        assert self._sum!=None, f"{self._name} is not updated: {self._sum}"
        return self._sum/self._cnt

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
