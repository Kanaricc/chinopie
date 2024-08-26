import abc
import random
from typing import TypedDict,Any,Optional,Sequence,List

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

class MultiClassSample(TypedDict):
    index:int
    name:str
    image:Tensor
    extra_image:Optional[Tensor]
    target:Tensor

class MultiClassDataset(abc.ABC,Dataset[MultiClassSample]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_defined_labels(self)->Sequence[str]:
        ...
    
    @abc.abstractmethod
    def __getitem__(self, index) -> MultiClassSample:
        ...
    
    @abc.abstractmethod
    def __len__(self)->int:
        ...
    
    @abc.abstractmethod
    def get_all_labels(self)->Tensor:
        ...
    
    @abc.abstractmethod
    def apply_new_labels(self,labels:List[int]):
        ...

class MultiClassLocalDataset(MultiClassDataset):
    def __init__(self, img_paths:List[str],num_labels:int, annotations:List[int],annotation_labels:List[str], preprocess: Any,extra_preprocess:Optional[Any], negatives_as_neg1=False) -> None:
        assert len(img_paths)==len(annotations)

        self._img_paths=img_paths
        self._num_labels=num_labels
        self._annotations=annotations
        self._annotation_labels=annotation_labels
        self._negative1=negatives_as_neg1
        self._preprocess=preprocess
        self._extra_preprocess=extra_preprocess
    
    def get_raw_data(self):
        return self._img_paths,self._annotations

    def get_defined_labels(self) -> List[str]:
        return self._annotation_labels
    
    def get_all_labels(self):
        t=torch.zeros((len(self),self._num_labels),dtype=torch.int)
        for k,v in enumerate(self._annotations):
            t[k,v]=1
        return t
    
    def apply_new_labels(self, labels: List[int]):
        assert len(labels)==len(self._annotations)
        self._annotations=labels
    
    def shuffle(self,seed):
        rng=random.Random(seed)
        packed=list(zip(self._img_paths,self._annotations))
        rng.shuffle(packed)
        self._img_paths=[x[0] for x in packed]
        self._annotation=[x[1] for x in packed]
    
    
    def __getitem__(self, index) -> MultiClassSample:
        path = self._img_paths[index]
        filename = path
        labels = self._annotations[index]
        rgb_image=Image.open(path).convert(
                "RGB"
            )
        image = self._preprocess(rgb_image)
        extra_image= self._extra_preprocess(rgb_image) if self._extra_preprocess else None

        target = torch.zeros(self._num_labels, dtype=torch.int)
        if self._negative1:
            target.fill_(-1)
        target[labels] = 1

        res:MultiClassSample={
            "index": index,
            "name": filename,
            "image": image,
            "extra_image":extra_image,
            "target": target,
        }
        if extra_image is None:
            # TODO: this should be fixed in python 3.11
            del res['extra_image'] # type: ignore
        return res

    def __len__(self) -> int:
        return len(self._img_paths)


class MultiClassInMemoryDataset(MultiClassDataset):
    def __init__(self, imgs:np.ndarray,num_labels:int, annotations:List[int],annotation_labels:List[str], preprocess: Any,extra_preprocess:Optional[Any], negatives_as_neg1=False) -> None:
        assert len(imgs)==len(annotations)

        self._imgs=imgs
        self._num_labels=num_labels
        self._annotations=annotations
        self._annotation_labels=annotation_labels
        self._negative1=negatives_as_neg1
        self._preprocess=preprocess
        self._extra_preprocess=extra_preprocess
    
    def get_raw_data(self):
        return self._imgs,self._annotations

    def get_defined_labels(self) -> List[str]:
        return self._annotation_labels
    
    def get_all_labels(self):
        t=torch.zeros((len(self),self._num_labels),dtype=torch.int)
        for k,v in enumerate(self._annotations):
            t[k,v]=1
        return t
    
    def apply_new_labels(self, labels: List[int]):
        assert len(labels)==len(self._annotations)
        self._annotations=labels
    
    
    def __getitem__(self, index) -> MultiClassSample:
        label = self._annotations[index]
        rgb_image=Image.fromarray(self._imgs[index])
        image = self._preprocess(rgb_image)
        extra_image= self._extra_preprocess(rgb_image) if self._extra_preprocess else None

        target = torch.zeros(self._num_labels, dtype=torch.int)
        if self._negative1:
            target.fill_(-1)
        target[label] = 1

        res:MultiClassSample={
            "index": index,
            "name": index,
            "image": image,
            "extra_image":extra_image,
            "target": target,
        }
        if extra_image is None:
            # TODO: this should be fixed in python 3.11
            del res['extra_image'] # type: ignore
        return res

    def __len__(self) -> int:
        return self._imgs.shape[0]


from .cifar import CIFAR10,CIFAR100
from .cub200 import Cub200Dataset

__all__=[
    "CIFAR10",
    "CIFAR100",
    "Cub200Dataset",
    "MultiClassInMemoryDataset",
    "MultiClassSample",
]