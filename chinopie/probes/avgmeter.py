import math
from typing import Optional,Any

from torch import Tensor
import torch
from torch.types import Number
from collections import deque
from .. import iddp as dist

"""
m=(a1+a2+a3+...+a_{n-1})/(n-1)
m'=(a1+a2+a3+...+a_{n-1}+an)/n
  =(mn-m+an)/n
  =m+(an-m)/n
"""

class SmoothMeanMeter:
    def __init__(self,length:int,level1:float=0.1,level2:float=0.25,level3:float=0.5) -> None:
        assert length>0, "the length should larger than 0"
        self._levels=list(map(int,[level1*length,level2*length,level3*length]))
        self._means=[0.]*len(self._levels)
        self._cnt=[0]*len(self._levels)
    
    def add(self,x:float):
        for i in range(len(self._levels)):
            self._cnt[i]+=1
            self._means[i]=self._means[i]+(x-self._means[i])/self._cnt[i]
    

    def __str__(self):
        return ', '.join(map(lambda x: f"{x:.2f}",self._means))

class AverageMeter:
    def __init__(self, name: str,dev:Any) -> None:
        self.name = name
        self._val = 0.
        self._sum = 0.
        self._cnt = 0.
        self._avg=0.
        self._dev=dev
        pass


    def update(self, val: Number, n=1):
        self._val = val
        self._sum += val*n
        self._cnt += n
        self._avg = self._sum/self._cnt
    
    def has_data(self):
        return self._cnt!=0

    def average(self) -> float:
        if not dist.is_initialized():
            x=self._avg
            return x
        else:
            x=torch.tensor(self._avg,device=self._dev,dtype=torch.float)
            dist.all_reduce(x,op=dist.ReduceOp.SUM)
            x/=float(dist.get_world_size())
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
