import os
import subprocess
import json
from typing import Any, Dict, List, Set
from torch.functional import Tensor
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import typing

class FakeOutput(typing.TypedDict):
    input:Tensor
    target:Tensor

class FakeEmptySet(Dataset):
    def __init__(self) -> None:
        pass

    def __len__(self):
        return 0
    
    def __getitem__(self, index: int):
        raise RuntimeError("trying to fetch data from empty dataset")

class FakeConstSet(Dataset):
    def __init__(self,input_like:Tensor,output_like:Tensor,size:int=128) -> None:
        self.input_like=input_like.detach().clone()
        self.output_like=output_like.detach().clone()
        self.size=size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index: int)->FakeOutput:
        input=self.input_like.clone()
        output=self.output_like.clone()

        return {
            'input':input,
            'target':output,
        }

class FakeRandomSet(Dataset):
    def __init__(self,input_like:Tensor,output_like:Tensor,size:int=128) -> None:
        self.input_like=input_like.detach()
        self.output_like=output_like.detach()
        self.size=size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index: int)->FakeOutput:
        input=generate_similar_tensor(self.input_like)
        output=generate_similar_tensor(self.output_like)

        return {
            'input':input,
            'target':output,
        }

class FakeNormalSet(Dataset):
    def __init__(self, means:List[float], stds:List[float], dim_feat:int, size:int) -> None:
        self._means=means
        self._stds=stds
        self._dim_feat=dim_feat
        self._size=size
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index:int) -> FakeOutput:
        idx=torch.randint(0,len(self._means),(1,))
        input=torch.normal(self._means[idx],self._stds[idx],size=(self._dim_feat,))

        return {
            'input':input,
            'target':torch.tensor(self._stds[idx]),
        }

def generate_similar_tensor(tensor:Tensor):
    if tensor.dtype==torch.float:
        output=torch.rand_like(tensor)
    else:
        maximum=tensor.max().item()
        output=torch.randint_like(tensor,int(maximum))
    
    return output