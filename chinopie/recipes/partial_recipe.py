import pdb
import warnings
from typing import Sequence,Any,Dict,TypeVar,Generic,Optional
from abc import ABC,abstractmethod
import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import chinopie
from chinopie.modelhelper import ModelStaff,PhaseEnv,HyperparameterManager

from . import ModuleRecipe
from .. import logging
_logger=logging.get_logger(__name__)




class TrainingRecipe(ModuleRecipe):
    def __init__(self, clamp_grad: float | None = None, eval_on_nograd_module: bool = True, stop_backward: bool = False,autocast:bool=False):
        super().__init__(clamp_grad, eval_on_nograd_module, stop_backward,autocast=autocast)
    
    def run_val_phase(self, p: PhaseEnv):
        _logger.info("skipping validation phase in training recipe")
        p.end_phase(0)

class EvaluationRecipe(ModuleRecipe):
    def __init__(self,autocast:bool=False):
        super().__init__(clamp_grad=None, eval_on_nograd_module=False, stop_backward=True,autocast=autocast)

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


    def switch_train(self, model: nn.Module | None = None):
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

