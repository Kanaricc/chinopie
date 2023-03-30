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
    
    def run_phase(self,phase:str):
        assert phase in ['train','val','test']
        if phase=='train':self.run_train_phase(self.trainset[1])
        elif phase=='val':self.run_val_phase(self.valset[1])
        elif phase=='test':self.run_test_phase(self.testset[1])
        else: raise RuntimeError(f"unknown phase `{phase}`")

    
    def run_train_phase(self,dataloader:DataLoader):
        self.model.train()
        for data in dataloader:
            self.run_iter(data,'train')
        pass

    def run_val_phase(self,dataloader:DataLoader):
        self.model.eval()
        for data in dataloader:
            self.run_iter(data,'val')
        pass

    def run_test_phase(self,dataloader:DataLoader):
        self.model.eval()
        for data in dataloader:
            self.run_iter(data,'test')
        pass

    def run_iter(self,data,phase:str):
        assert phase in ['train','val','test']
        if phase=='train':self.run_train_iter(data)
        elif phase=='val':self.run_val_iter(data)
        elif phase=='test':self.run_test_iter(data)
    
    def run_train_iter(self,data:Any):
        output=self.forward(data)
        loss=self.cal_loss(data,output)
        self.update_probe('train')
    
    def run_val_iter(self,data):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
            self.update_probe('val')
    
    def run_test_iter(self,data):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
            self.update_probe('test')
    
    def forward(self,data)->Tensor:
        ...
    
    def cal_loss(self,data,output:Tensor)->Tensor:
        ...
    
    def update_probe(self,probes,output:Tensor,phase:str):
        assert phase in ['train','val','test']
        if phase=='train':self.update_train_probe(data)
        elif phase=='val':self.update_val_probe(data)
        elif phase=='test':self.update_test_probe(data)
    
    def update_train_probe(self,probes,output):
        ...
    
    def update_val_probe(self,probes,output):
        ...
    
    def update_test_probe(self,probes,output):
        ...

    def restore_ckpt(self,ckpt):
        ...
    
    def export_ckpt(self):
        ...
    
    def fit(self)->float|Sequence[float]:
        for epochi in self._helper.range_epoch():
            with self._helper.phase_train() as p:
                