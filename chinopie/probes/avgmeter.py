from torch import Tensor
import torch
from torch.types import Number
from collections import deque

class SmoothMeanMeter:
    def __init__(self,length:int,level1:float=0.1,level2:float=0.25,level3:float=0.5) -> None:
        self._levels=[level1*length,level2*length,level3*length]
        self._qs=[deque(maxlen=int(x)) for x in self._levels]
    
    def add(self,x:float):
        for q in self._qs:
            q.append(x)
    
    def _sync_dist_nodes(self):
        raise NotImplemented

    def __str__(self):
        res=[]
        for q in self._qs:
            t=torch.tensor(list(q),dtype=torch.float)
            res.append(t.mean().item())
        return ', '.join(map(lambda x: f"{x:.2f}",res))

class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self._val = 0
        self._sum = 0
        self._cnt = 0
        self._avg=0
        pass

    def _sync_dist_nodes(self):
        raise NotImplemented

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
