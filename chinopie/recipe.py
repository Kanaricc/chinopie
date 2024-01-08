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
from chinopie.modelhelper import ModelStaff,PhaseEnv,HyperparameterManager

from . import logging
_logger=logging.get_logger(__name__)


class ModuleRecipe(ABC):
    def __init__(self, clamp_grad:Optional[float]=None,eval_on_nograd_module:bool=True,stop_backward:bool=False):
        self._clamp_grad=clamp_grad
        self._eval_on_nograd_module=eval_on_nograd_module
        self._stop_backward=stop_backward

        self._cur_epoch:Optional[int]=None
        self._cur_batch:Optional[int]=None
        self._total_epoch:Optional[int]=None
        self._total_batch:Optional[int]=None
        pass

    def ask_hyperparameter(self,hp:HyperparameterManager):
        pass


    def prepare(self,staff:ModelStaff):
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

    def end(self,helper:ModelStaff):
        ...

    @abstractmethod
    def set_optimizers(self,model)->Optimizer:
        ...

    def set_scheduler(self,optimizer:Optimizer)->Optional[LRScheduler]:
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
    
    def run_train_phase(self,p:PhaseEnv):
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

    def run_val_phase(self,p:PhaseEnv):
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

    def run_test_phase(self,p:PhaseEnv):
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
    
    def run_train_iter(self,data,p:PhaseEnv):
        dev_data=chinopie.any_to(data,self.dev)
        self.before_iter_train(data)

        output=self.forward_train(dev_data)
        loss=self.cal_loss_train(dev_data,output)
        p.update_loss(loss.detach().cpu())

        if not self._stop_backward:
            self.optimizer.zero_grad()
            loss.backward()
            if self._clamp_grad is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),max_norm=self._clamp_grad)
            self.optimizer.step()
        else:
            warnings.warn("backward is stopped")

        output_cpu=chinopie.any_to(output,'cpu')
        self.update_probe_train(data,output_cpu,p)
        self.after_iter_train(data,output_cpu)


    def run_val_iter(self,data,p:PhaseEnv):
        self.before_iter_val(data)
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_val(dev_data)
            loss=self.cal_loss_val(dev_data,output)
            p.update_loss(loss.detach().cpu())
            output_cpu=chinopie.any_to(output,'cpu')
            self.update_probe_val(data,output_cpu,p)

        self.after_iter_val(data,output_cpu)
    
    def run_test_iter(self,data,p:PhaseEnv):
        self.before_iter_test(data)
        with torch.no_grad():
            dev_data=chinopie.any_to(data,self.dev)
            output=self.forward_test(dev_data)
            loss=self.cal_loss_test(dev_data,output)
            p.update_loss(loss.detach().cpu())
            output_cpu=chinopie.any_to(output,'cpu')
            self.update_probe_test(data,output_cpu,p)
        
        self.after_iter_test(data,output_cpu)
    
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
    
    def update_probe_train(self,data,output,p:PhaseEnv):
        self.update_probe(data,output,p)
    
    def update_probe_val(self,data,output,p:PhaseEnv):
        self.update_probe(data,output,p)
    
    def update_probe_test(self,data,output,p:PhaseEnv):
        self.update_probe(data,output,p)
    
    def update_probe(self,data,output,p:PhaseEnv):
        """
        update managed custom probe here
        """
        pass

    def before_iter_train(self,data):
        self.before_iter(data)
    
    def before_iter_val(self,data):
        self.before_iter(data)
    
    def before_iter_test(self,data):
        self.before_iter(data)

    def before_iter(self,data):
        ...
    
    def after_iter_train(self,data,output):
        self.after_iter(data,output,'train')

    def after_iter_val(self,data,output):
        self.after_iter(data,output,'val')

    def after_iter_test(self,data,output):
        self.after_iter(data,output,'test')
    
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
    
    def export_model_state(self):
        return self.model.state_dict()
    
    def import_model_state(self,state):
        self.model.load_state_dict(state)

    def restore_ckpt(self,ckpt:str)->Dict[str,Any]:
        data=torch.load(ckpt,map_location=self.dev)
        if 'custom' in ckpt:
            self.import_custom_state(data['custom'])
        self.import_model_state(data['model']) # this is done after custom state loaded
        self.optimizer.load_state_dict(data['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(data['scheduler'])
        else:
            if 'scheduler' in data:
                _logger.warning("found scheduler state in checkpoint but no scheduler is set")
        return data['extra']
    
    
    def save_ckpt(self,ckpt:str,extra_state:Any):
        data={
            'model':self.export_model_state(),
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

class TrainingRecipe(ModuleRecipe):
    def __init__(self, clamp_grad: float | None = None, eval_on_nograd_module: bool = True, stop_backward: bool = False):
        super().__init__(clamp_grad, eval_on_nograd_module, stop_backward)
    
    def run_val_phase(self, p: PhaseEnv):
        _logger.info("skipping validation phase in training recipe")
        p.end_phase(0)

class EvaluationRecipe(ModuleRecipe):
    def __init__(self):
        super().__init__(clamp_grad=None, eval_on_nograd_module=False, stop_backward=True)

    def run_train_phase(self, p: PhaseEnv):
        _logger.info("skipped training phase in evaluation recipe.")
        p.end_phase(0)

    def run_train_iter(self,data,p:PhaseEnv):
        dev_data=chinopie.any_to(data,self.dev)
        output=self.forward_train(dev_data)
        loss=self.cal_loss_train(dev_data,output)
        p.update_loss(loss.detach().cpu())

        output_cpu=chinopie.any_to(output,'cpu')
        self.update_probe(data,output_cpu,p)
        self.after_iter(data,output_cpu,'train')


    def switch_train(self, model):
        # Set eval during training to avoid potential changes to models.
        # But useless.
        self.switch_eval(model)

    def save_ckpt(self,ckpt:str,extra_state:Any):
        # Avoid saving useless data. One can still export its own data.
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
        _logger.debug("restore model training/evaluating status")