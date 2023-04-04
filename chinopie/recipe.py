from typing import Sequence,Any,Dict,TypeVar,Generic
from abc import ABC,abstractmethod
import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from chinopie.modelhelper import TrainHelper,PhaseHelper


M=TypeVar('M')
O=TypeVar('O')

class ModuleRecipe(ABC,Generic[M,O]):
    def __init__(self):
        pass

    def reg_params(self,helper:TrainHelper):
        pass

    
    def prepare(self,helper:TrainHelper):
        pass
    
    # TODO: this should be removed
    def _set_helper(self,helper:TrainHelper):
        self._helper=helper
    
    
    @property
    def model(self):
        return self._helper._model
    
    @property
    def optimizer(self):
        return self._helper._optimizer
    
    def run_train_phase(self,p:PhaseHelper):
        self.model.train()
        for batchi,data in p.range_data():
            # TODO: check device
            self.run_train_iter(data,p)
        p.end_phase(self.report_score('train'))

    def run_val_phase(self,p:PhaseHelper):
        self.model.eval()
        for batchi,data in p.range_data():
            self.run_val_iter(data,p)
        p.end_phase(self.report_score('val'))

    def run_test_phase(self,p:PhaseHelper):
        self.model.eval()
        for batchi,data in p.range_data():
            self.run_test_iter(data,p)
        pass
        p.end_phase(self.report_score('test'))
    
    def run_train_iter(self,data:Any,p:PhaseHelper):
        output=self.forward(data)
        loss=self.cal_loss(data,output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        p.update_loss(loss.detach().cpu())
        self.update_probe(data,output.detach().cpu(),p)
        self.after_iter(data,output,'train')
    
    def run_val_iter(self,data,p:PhaseHelper):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
            p.update_loss(loss.detach().cpu())
            self.update_probe(data,output.detach().cpu(),p)
        self.after_iter(data,output,'val')
    
    def run_test_iter(self,data,p:PhaseHelper):
        with torch.no_grad():
            output=self.forward(data)
            loss=self.cal_loss(data,output)
            p.update_loss(loss.detach().cpu())
            self.update_probe(data,output.detach().cpu(),p)
        self.after_iter(data,output,'test')
    
    @abstractmethod
    def forward(self,data)->Tensor:
        raise NotImplemented
    
    @abstractmethod
    def cal_loss(self,data,output:Tensor)->Tensor:
        raise NotImplemented
    
    def update_probe(self,data,output,p:PhaseHelper):
        """
        managed custom probe are supposed to be updated here
        """
        ...
    
    def after_iter(self,data,output,phase:str):
        """
        your probes are supposed to be updated here
        """
        ...
    
    @abstractmethod
    def report_score(self,phase:str)->float:
        """
        report the score of the phase
        """
        raise NotImplemented

    def restore_ckpt(self,ckpt:str)->Dict[str,Any]:
        data=torch.load(ckpt,map_location='cpu')
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        return data['extra']
    
    def save_ckpt(self,ckpt:str,extra_state:Any):
        data={
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'extra':extra_state,
        }
        torch.save(data,ckpt)
    
    def before_epoch(self):
        ...

    def after_epoch(self):
        ...
                