import math
from typing import Optional,Any

from torch import Tensor
import torch
from torch.types import Number
from collections import deque
from .. import iddp as dist

class SmoothMeanMeter:
    def __init__(self,length:int,level1:float=0.1,level2:float=0.25,level3:float=0.5) -> None:
        self._levels=[level1*length,level2*length,level3*length]
        self._decays=[math.pow(1/x,1/x) for x in self._levels]
        self._norm=[1 for x in self._levels]
        self._qs=[0 for x in self._levels]
    
    def add(self,x:float):
        self._norm=[qs*d+1 for qs,d in zip(self._norm,self._decays)]
        self._qs=[qs*d+x for qs,d in zip(self._qs,self._decays)]
    

    def __str__(self):
        # TODO: sync
        t=[x/norm for x,norm in zip(self._qs,self._norm)]
        return ', '.join(map(lambda x: f"{x:.2f}",t))

class AverageMeter:
    def __init__(self, name: str,dev:Any) -> None:
        self.name = name
        self._val = 0
        self._sum = 0
        self._cnt = 0
        self._avg=0
        self._dev=dev
        pass


    def update(self, val: Number, n=1):
        self._val = val
        self._sum += val*n
        self._cnt += n
        self._avg = self._sum/self._cnt
    
    def has_data(self):
        return self._cnt!=0

    def average(self) -> Optional[float]:
        if not dist.is_initialized():
            x=self._avg
            return x
        else:
            x=torch.tensor(self._avg,device=self._dev)
            dist.all_reduce(x,op=dist.ReduceOp.AVG)
            return x.item() # type: ignore

    def value(self) -> Number:
        return self._val
    
    def reset(self):
        self._val=0
        self._sum=0
        self._cnt=0
        self._avg=0
        
    def __str__(self) -> str:
        return f"{self.name}: {self.value():.5f}(avg {self.average():.5f})"
