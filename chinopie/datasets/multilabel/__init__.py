import abc
from typing import TypedDict,Any,Optional,Sequence,List
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

class MultiLabelSample(TypedDict):
    index:int
    name:str
    image:Tensor
    extra_image:Optional[Tensor]
    target:Tensor

class MultiLabelDataset(abc.ABC,Dataset[MultiLabelSample]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_defined_labels(self)->Sequence[str]:
        ...
    
    @abc.abstractmethod
    def __getitem__(self, index) -> MultiLabelSample:
        ...
    
    @abc.abstractmethod
    def __len__(self)->int:
        ...
    
    @abc.abstractmethod
    def get_all_labels(self)->Tensor:
        ...
    
    @abc.abstractmethod
    def apply_new_labels(self,labels:Tensor):
        ...

class MultiLabelLocalDataset(MultiLabelDataset):
    def __init__(self, img_paths:List[str],num_labels:int, annotations:List[List[int]],annotation_labels:List[str], preprocess: Any,extra_preprocess:Optional[Any], negatives_as_neg1=False) -> None:
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
    
    def apply_new_labels(self, labels: Tensor):
        assert labels.size(0)==len(self) and labels.size(1)==self._num_labels
        assert labels.dtype==torch.int or labels.dtype==torch.long
        for k,v in enumerate(labels):
            self._annotations[k]=(v==1).nonzero(as_tuple=True)[0].tolist()
    
    
    def __getitem__(self, index) -> MultiLabelSample:
        path = self._img_paths[index]
        filename = path
        labels = sorted(self._annotations[index])
        rgb_image=Image.open(path).convert(
                "RGB"
            )
        image = self._preprocess(rgb_image)
        extra_image= self._extra_preprocess(rgb_image) if self._extra_preprocess else None

        target = torch.zeros(self._num_labels, dtype=torch.int)
        if self._negative1:
            target.fill_(-1)
        target[labels] = 1

        res:MultiLabelSample={
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

        
from .coco2014 import COCO2014Dataset