import pdb
import warnings
from typing import Sequence,Any,Dict,TypeVar,Generic,Optional
from abc import ABC,abstractmethod
import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import chinopie
from chinopie.modelhelper import ModelStaff,PhaseHelper,HyperparameterManager

from . import logging
_logger=logging.get_logger(__name__)


class ModuleRecipe(ABC):
    def __init__(self, clamp_grad:Optional[float]=None,eval_on_nograd_module:bool=True):
        self._clamp_grad=clamp_grad
        self._eval_on_nograd_module=eval_on_nograd_module

        self._cur_epoch:Optional[int]=None
        self._cur_batch:Optional[int]=None
        self._total_epoch:Optional[int]=None
        self._total_batch:Optional[int]=None
        pass


    def prepare(self,hp:HyperparameterManager,staff:ModelStaff,inherited_states:Dict[str,Any]):
        """
        prepare models and probes here
        """
        pass

    def before_start(self):
        """
        Pre-calculation are done here.
        One can expect the model is correctly broadcasted.
        """
        pass

    def end(self,helper:ModelStaff)->Dict[str,Any]:
        _logger.info("pass empty state to next stage")
        return {}

    @abstractmethod
    def set_optimizers(self,model,hp:HyperparameterManager,staff:ModelStaff)->Optimizer:
        ...

    def set_scheduler(self,optimizer:Optimizer,hp:HyperparameterManager,staff:ModelStaff)->Optional[LRScheduler]:
        _logger.info(f"no scheduler set for optimizer `{type(optimizer)}`")
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
    
    @property
    def cur_epoch(self):
        assert self._cur_epoch is not None, "Epoch is not init. Did you access it before the recipe starts?"
        return self._cur_epoch
    
    @property
    def cur_batch(self):
        assert self._cur_batch is not None, "Batch is not init. Did you access it before the recipe starts?"
        return self._cur_batch
    
    @property
    def total_epoch(self):
        assert self._total_epoch is not None, "Epoch is not init. Did you access it before the recipe starts?"
        return self._total_epoch
    
    @property
    def total_batch(self):
        assert self._total_batch is not None, "Batch is not init. Did you access it before the recipe starts?"
        return self._total_batch

    
    
    def switch_train(self,model:Optional[nn.Module]=None):
        if model is None:model=self.model
        chinopie.set_train(model,self._eval_on_nograd_module)
    
    def switch_eval(self,model:Optional[nn.Module]=None):
        if model is None:model=self.model
        chinopie.set_eval(model)
    
    def run_train_phase(self,p:PhaseHelper):
        self.switch_train(self.model)
        state_before=self.model.training
        self._total_batch=p._batch_len
        for batchi,data in p.range_data():
            self._cur_batch=batchi
            self.run_train_iter(data,p)
        state_after=self.model.training
        if state_before!=state_after:
            _logger.warn("The model's state seems be changed unexpectedly during train phase. Please check your code.")
        if not self.model.training:
            warnings.warn("The model is not set for training during train phase. Please understand what you have done.")
        p.end_phase(self.report_score('train'))

    def run_val_phase(self,p:PhaseHelper):
        self.switch_eval(self.model)
        state_before=self.model.training
        self._total_batch=p._batch_len
        for batchi,data in p.range_data():
            self._cur_batch=batchi
            self.run_val_iter(data,p)
        state_after=self.model.training
        if state_before!=state_after:
            _logger.warn("The model's state seems be changed unexpectedly during train phase. Please check your code.")
        if self.model.training:
            warnings.warn("The model is not set for evaluating during val phase. Please understand what you have done.")
        p.end_phase(self.report_score('val'))

    def run_test_phase(self,p:PhaseHelper):
        self.switch_eval(self.model)
        state_before=self.model.training
        self._total_batch=p._batch_len
        for batchi,data in p.range_data():
            self._cur_batch=batchi
            self.run_test_iter(data,p)
        state_after=self.model.training
        if state_before!=state_after:
            _logger.warn("The model's state seems be changed unexpectedly during train phase. Please check your code.")
        if self.model.training:
            warnings.warn("The model is not set for evaluating during val phase. Please understand what you have done.")
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

        output_cpu=chinopie.any_to(output,'cpu')
        self.update_probe(data,output_cpu,p)
        self.after_iter(data,output_cpu,'train')


    def run_val_iter(self,data,p:PhaseHelper):
        self.before_iter(data,'test')
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_val(dev_data)
            loss=self.cal_loss_val(dev_data,output)
            p.update_loss(loss.detach().cpu())
            output_cpu=chinopie.any_to(output,'cpu')
            self.update_probe(data,output_cpu,p)

        self.after_iter(data,output_cpu,'val')
    
    def run_test_iter(self,data,p:PhaseHelper):
        self.before_iter(data,'test')
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_test(dev_data)
            loss=self.cal_loss_test(dev_data,output)
            p.update_loss(loss.detach().cpu())
            output_cpu=chinopie.any_to(output,'cpu')
            self.update_probe(data,output_cpu,p)
        
        self.after_iter(data,output_cpu,'test')
    
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
        raise NotImplementedError()
    
    def update_probe(self,data,output,p:PhaseHelper):
        """
        update managed custom probe here
        """
        pass

    def before_iter(self,data,phase:str):
        ...
    
    def after_iter(self,data,output,phase:str):
        """
        update unmanaged probes here
        """
        ...
    
    @abstractmethod
    def report_score(self,phase:str)->float:
        """
        report the score of the phase
        """
        raise NotImplemented

    def restore_ckpt(self,ckpt:str)->Dict[str,Any]:
        data=torch.load(ckpt,map_location=self.dev)
        if 'custom' in ckpt:
            self.import_custom_state(data['custom'])
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(data['scheduler'])
        else:
            if 'scheduler' in data:
                _logger.warning("found scheduler state in checkpoint but no scheduler is set")
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


class EvaluationRecipe(ModuleRecipe):
    def run_train_iter(self,data,p:PhaseHelper):
        dev_data=chinopie.any_to(data,self.dev)
        output=self.forward_train(dev_data)
        loss=self.cal_loss_train(dev_data,output)
        p.update_loss(loss.detach().cpu())

        if self._clamp_grad is not None:
            warnings.warn("The clamp_grad is on in evaluation recipe.")

        output_cpu=chinopie.any_to(output,'cpu')
        self.update_probe(data,output_cpu,p)
        self.after_iter(data,output_cpu,'train')
    

    def save_ckpt(self,ckpt:str,extra_state:Any):
        # one can still export its own data
        data={
            'extra':extra_state,
        }
        custom_state=self.export_custom_state()
        if custom_state is not None:
            data['custom']=custom_state
        torch.save(data,ckpt)

class ModelStateKeeper:
    def __init__(self,recipe:ModuleRecipe,model:nn.Module) -> None:
        self._is_training=model.training
        self._model=model
        self._recipe=recipe

    def __enter__(self):
        self._is_training=self._model.training
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_training:
            self._recipe.switch_train(self._model)
        else:
            self._recipe.switch_eval(self._model)