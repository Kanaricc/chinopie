import abc
from typing import TypedDict,Any,Optional,Sequence
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

    @classmethod
    @abc.abstractmethod
    def get_defined_labels(cls)->Sequence[str]:
        ...
    
    @abc.abstractmethod
    def __getitem__(self, index) -> MultiLabelSample:
        ...
    
    @abc.abstractmethod
    def __len__(self)->int:
        ...
    
    def reg_extra_preprocess(self,preprocess:Any):
        self.extra_preprocess=preprocess

    @abc.abstractmethod
    def retain_range(self,l:int,r:int):
        ...
    
    @abc.abstractmethod
    def get_all_labels(self):
        ...
    
    @abc.abstractmethod
    def apply_new_labels(self,labels:Tensor):
        ...

class MultiLabelCommonDataset(MultiLabelDataset):
    def __init__(self, preprocess: Any, negatives_as_neg1=False) -> None:
        


from .coco2014 import COCO2014Dataset