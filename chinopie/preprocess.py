from typing import Any
from torch.utils.data import Dataset
from loguru import logger


class DatasetWrapper(Dataset):
    def __init__(self,dataset:Dataset,preprocess:Any) -> None:
        self.dataset=dataset
        self.preprocess=preprocess
    

    def __getitem__(self, index: Any):
        item=self.dataset.__getitem__(index)
        item=self.preprocess(item)
        
        return item
    
    def __len__(self):
        invert_op = getattr(self.dataset, "__len__", None)
        if callable(invert_op):
            return invert_op()
        else:
            raise Exception("not implemented")

class DatasetDualWrapper(Dataset):
    def __init__(self,dataset:Dataset,preprocess1:Any,preprocess2:Any) -> None:
        self.dataset=dataset
        self.preprocess1=preprocess1
        self.preprocess2=preprocess2
    

    def __getitem__(self, index: Any):
        item=self.dataset.__getitem__(index)
        item1=self.preprocess1(item)
        item2=self.preprocess2(item)
        
        return item1,item2
    
    def __len__(self):
        invert_op = getattr(self.dataset, "__len__", None)
        if callable(invert_op):
            return invert_op()
        else:
            raise Exception("not implemented")