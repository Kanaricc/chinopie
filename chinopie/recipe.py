from typing import Sequence,Any
from abc import ABC
import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader
from chinopie.modelhelper import TrainHelper
from chinopie.phasehelper import PhaseHelper

class ModuleRecipe:
    def __init__(self) -> None:
        ...
    
    def _set_global_deps(self,helper:TrainHelper):
        self._helper=helper
        self._model:nn.Module=None
    
    @property
    def model(self):
        return self._model
    
    @property
    def trainset(self):
        return self._helper._data_train,self._helper._dataloader_train
    
    @property
    def valset(self):
        return self._helper._data_val,self._helper._dataloader_val
    
    @property
    def testset(self):
        return self._helper._data_test,self._helper._dataloader_test
    
    def run_train_phase(self,dataloader:DataLoader):
        self.model.train()
        for data in dataloader:
            self.run_train_iter(data)
        pass

    def run_val_phase(self,dataloader:DataLoader):
        self.model.eval()
        for data in dataloader:
            self.run_val_iter(data)
        pass

    def run_test_phase(self,dataloader:DataLoader):
        self.model.eval()
        for data in dataloader:
            self.run_test_iter(data)
        pass
    
    def run_train_iter(self,data:Any):
        output=self.forward(data)
        loss=self.cal_loss(data,output)
        self.update_probe()
    
    def run_val_iter(self,data):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
    
    def run_test_iter(self,data):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
    
    def forward(self,data)->Tensor:
        raise NotImplemented
    
    def cal_loss(self,data,output:Tensor)->Tensor:
        raise NotImplemented
    
    def update_probe(self):
        raise NotImplemented

    def restore_ckpt(self,ckpt):
        raise NotImplemented
    
    def export_ckpt(self):
        raise NotImplemented
    
    def fit(self)->float|Sequence[float]:
        for epochi in self._helper.range_epoch():
            with self._helper.phase_train() as p:
                