import pdb
from typing import Sequence,Any,Dict,TypeVar,Generic,Optional
from abc import ABC,abstractmethod
import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import chinopie
from chinopie import logger
from chinopie.modelhelper import ModelStaff,PhaseHelper,HyperparameterManager


class ModuleRecipe(ABC):
    def __init__(self, clamp_grad:Optional[float]=None):
        self._clamp_grad=clamp_grad
        pass


    def prepare(self,hp_manager:HyperparameterManager,staff:ModelStaff,inherited_states:Dict[str,Any]):
        """
        prepare models and probes here
        """
        pass

    def end(self,helper:ModelStaff)->Dict[str,Any]:
        logger.info("pass empty state to next stage")
        return {}

    @abstractmethod
    def set_optimizers(self,model,hp_manager:HyperparameterManager,staff:ModelStaff)->Optimizer:
        ...

    def set_scheduler(self,optimizer:Optimizer,hp_manager:HyperparameterManager,staff:ModelStaff)->Optional[LRScheduler]:
        logger.info(f"no scheduler set for optimizer `{type(optimizer)}`")
        return None
    
    def _set_staff(self,staff:ModelStaff):
        self._staff=staff
    
    @property
    def model(self):
        return self._staff._model
    
    @property
    def optimizer(self):
        return self._staff._optimizer
    
    @property
    def scheduler(self):
        if hasattr(self._staff,'_scheduler'):
            return self._staff._scheduler
        return None
    
    @property
    def dev(self):
        return self._staff.dev
    
    
    def switch_train(self,model:nn.Module):
        if model.training:
            logger.warning("wrong model mode detected: expected model in eval mode but found train mode.")
        chinopie.set_train(model)
    
    def switch_eval(self,model:nn.Module):
        if not model.training:
            logger.warning("wrong model mode detected: expected model in train mode but found eval mode.")
        chinopie.set_eval(model)
    
    def run_train_phase(self,p:PhaseHelper):
        self.switch_train(self.model)
        for batchi,data in p.range_data():
            self.run_train_iter(data,p)
        p.end_phase(self.report_score('train'))

    def run_val_phase(self,p:PhaseHelper):
        self.switch_eval(self.model)
        for batchi,data in p.range_data():
            self.run_val_iter(data,p)
        p.end_phase(self.report_score('val'))

    def run_test_phase(self,p:PhaseHelper):
        self.model.eval()
        for batchi,data in p.range_data():
            self.run_test_iter(data,p)
        pass
        p.end_phase(self.report_score('test'))
    
    def run_train_iter(self,data,p:PhaseHelper):
        dev_data=chinopie.any_to(data,self.dev)
        output=self.forward_train(dev_data)
        loss=self.cal_loss_train(dev_data,output)
        p.update_loss(loss.detach().cpu())

        self.optimizer.zero_grad()
        loss.backward()
        if self._clamp_grad is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),max_norm=self._clamp_grad)
        self.optimizer.step()
        self.update_probe(data,chinopie.any_to(output,'cpu'),p)
        self.after_iter(dev_data,output,'train')


    def run_val_iter(self,data,p:PhaseHelper):
        self.before_iter(data,'test')
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_val(dev_data)
            loss=self.cal_loss_val(dev_data,output)
            p.update_loss(loss.detach().cpu())
            self.update_probe(data,chinopie.any_to(output,'cpu'),p)
        self.after_iter(dev_data,output,'val')
    
    def run_test_iter(self,data,p:PhaseHelper):
        self.before_iter(data,'test')
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_test(dev_data)
            loss=self.cal_loss_test(dev_data,output)
            p.update_loss(loss.detach().cpu())
            self.update_probe(data,chinopie.any_to(output,'cpu'),p)
        self.after_iter(dev_data,output,'test')
    
    def forward_train(self,data)->Any:
        return self.forward(data)

    def forward_val(self,data)->Any:
        return self.forward(data)

    def forward_test(self,data)->Any:
        return self.forward(data)

    @abstractmethod
    def forward(self,data)->Any:
        raise NotImplemented
    
    def cal_loss_train(self,data,output)->Tensor:
        return self.cal_loss(data,output)

    def cal_loss_val(self,data,output)->Tensor:
        return self.cal_loss(data,output)

    def cal_loss_test(self,data,output)->Tensor:
        return self.cal_loss(data,output)
    
    @abstractmethod
    def cal_loss(self,data,output)->Tensor:
        raise NotImplemented
    
    def update_probe(self,data,output,p:PhaseHelper):
        """
        managed custom probe are supposed to be updated here
        """
        pass

    def before_iter(self,data,phase:str):
        ...
    
    def after_iter(self,data,output,phase:str):
        """
        unmanaged probes are supposed to be updated here
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
        if 'custom' in ckpt:
            self.import_custom_state(data['custom'])
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(data['scheduler'])
        else:
            if 'scheduler' in data:
                logger.warning("found scheduler state in checkpoint but no scheduler is set")
        return data['extra']
    
    def save_ckpt(self,ckpt:str,extra_state:Any):
        data={
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'extra':extra_state,
        }
        if self.scheduler is not None:
            data['scheduler']=self.scheduler.state_dict(),
        custom_state=self.export_custom_state()
        if custom_state is not None:
            data['custom']=custom_state
        torch.save(data,ckpt)

    def export_custom_state(self)->Optional[Dict[str,Any]]:
        return None
    
    def import_custom_state(self,state:Dict[str,Any]):
        ...

    
    def before_epoch(self):
        """
        do schedular task here
        """
        ...

    def after_epoch(self):
        """
        do schedular task here
        """
        if self.scheduler is not None:
            self.scheduler.step()
