from torch import Tensor
import torch
from torch.types import Number
from collections import deque

class SmoothMeanMeter:
    def __init__(self,level1:int=5,level2:int=25,level3:int=75) -> None:
        self._levels=[level1,level2,level3]
        self._qs=[deque(maxlen=x) for x in self._levels]
    
    def add(self,x:float,n:int=1):
        for q in self._qs:
            for i in range(n):
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
