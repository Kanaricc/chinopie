from datetime import datetime
import logging
import os, sys, shutil,pdb
import argparse
import random
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import warnings


import torch
import torch.backends.mps
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.multiprocessing as mp
import optuna
from optuna.distributions import CategoricalChoiceType
import numpy as np

from .probes.avgmeter import AverageMeter
from .datasets.fakeset import FakeEmptySet
from . import iddp as dist
from .filehelper import GlobalFileHelper,InstanceFileHelper
from .phasehelper import (
    PhaseEnv,
)
from .utils import show_params_in_3cols,create_snapshot,check_gitignore,set_fixed_seed
from .logging import get_logger,set_logger_file,set_verbosity

logger=get_logger(__name__)

class HyperparameterManager:
    def __init__(self) -> None:
        self._param_config:Dict[str,Any]={}
        self._arg_config:Dict[str,Optional[Any]]={}
        self._arg_parser=argparse.ArgumentParser()

        self._fixed=False
    
    def reg_category(self, name: str, value: Optional[CategoricalChoiceType] = None, force:bool=False):
        assert self._fixed==False or force, "cannot reg new parameter after fixed"
        if name not in self._arg_config:
            self._arg_parser.add_argument(f"--{name}",required=False)
            self._arg_config[name] = value

    def reg_int(self, name: str, value: Optional[int] = None, force:bool=False):
        assert self._fixed==False or force, "cannot reg new parameter after fixed"
        if name not in self._arg_config:
            self._arg_parser.add_argument(f"--{name}",type=int,required=False)
            self._arg_config[name] = value

    def reg_float(self, name: str, value: Optional[float] = None, force:bool=False):
        assert self._fixed==False or force, "cannot reg new parameter after fixed"
        if name not in self._arg_config:
            self._arg_parser.add_argument(f"--{name}",type=float,required=False)
            self._arg_config[name] = value
    
    def parse_args(self,raw_args:List[str]):
        args=self.arg_parser.parse_args(raw_args)
        logger.debug(f"hyperparameters in argparser: {args}")
        for k in self._arg_config.keys():
            if getattr(args,k) is not None:
                self._arg_config[k]=getattr(args,k)
                logger.debug(f"flushed `{k}`")
    
    def load_params(self,val:Dict[str,Any]):
        self._param_config|=val
    
    def _set_trial(self,trial:optuna.Trial):
        self._trial=trial
        self._param_config.clear() # clear params for next trial
        self._fixed=True
        
    
    def suggest_category(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        assert name in self._arg_config, f"request for unregisted param `{name}`"
        fixed_val = self._arg_config[name]
        if fixed_val is not None:
            assert fixed_val in choices
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            self._param_config[name] = self._trial.suggest_categorical(name, choices)
            return self._param_config[name]

    def suggest_int(self, name: str, low: int, high: int, step=1, log=False) -> int:
        assert name in self._arg_config, f"request for unregisted param `{name}`"
        fixed_val = self._arg_config[name]
        if fixed_val is not None:
            if fixed_val< low or fixed_val>high:
                logger.warning(f"fixed val {fixed_val} of {name} is out of interval")
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            self._param_config[name] =  self._trial.suggest_int(name, low, high, step=step, log=log)
            return self._param_config[name] # type: ignore

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log=False,
    ) -> float:
        assert name in self._arg_config, f"request for unregisted param `{name}`"
        fixed_val = self._arg_config[name]
        if fixed_val is not None:
            if fixed_val< low or fixed_val>high:
                logger.warning(f"fixed val {fixed_val} of {name} is out of interval [{low}, {high}]")
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            self._param_config[name] =  self._trial.suggest_float(name, low, high, step=step, log=log)
            return self._param_config[name] # type: ignore
    
    @property
    def arg_parser(self):
        return self._arg_parser
    
    @property
    def params(self):
        return self._arg_config|self._param_config
    
    

# ModelStaff has no state
class ModelStaff:
    def __init__(
        self,
        file_helper: InstanceFileHelper,
        prev_file_helpers:Optional[List[InstanceFileHelper]],
        dev: str,
        diagnose:bool,
    ) -> None:
        self.file=file_helper
        self.prev_files=prev_file_helpers
        self._diagnose=diagnose

        logger.info(f"[DDP] current rank is {dist.get_rank()}")
        if dist.is_main_process():
            logger.info(f"[DDP] rank {dist.get_rank()} is the leader with full functions")
        else:
            logger.info(f"[DDP] rank {dist.get_rank()} is the follower with limited functions")
        
        world_size=dist.get_world_size()
        if dev=='cuda':
            assert torch.cuda.is_available(), "cuda is not available"
            if world_size<=torch.cuda.device_count():
                self.dev=f"cuda:{dist.get_rank()}"
            else:
                raise ValueError(f"world_size is larger than the number of devices")
        elif dev=='cpu':
            self.dev=f"cpu"
        elif dev=='mps':
            assert torch.backends.mps.is_available(), "mps is not available"
            assert world_size==1, "mps mode only support single process"
            self.dev='mps'
        else:
            raise ValueError(f"unknown device `{dev}`")
        logger.info(f"[DDP] use `{self.dev}` for this process")

        self._custom_probes = []
        self._flags={}
    
    def _set_flag(self,key:str,val:Any=True):
        self._flags[key]=val
    
    def _get_flag(self,key:str):
        return self._flags[key] if key in self._flags else None

    def reg_probe(self, name: str):
        self._custom_probes.append(name)
        logger.debug(f"register probe `{name}`")

    def reg_dataset(
        self, train: Any, trainloader: DataLoader, val: Any, valloader: DataLoader
    ):
        self._data_train = train
        self._dataloader_train = trainloader
        self._data_val = val
        self._dataloader_val = valloader
        logger.debug("registered train and val set")
        self._set_flag('trainval_data_set')

        # worker_init_fn are used in data iteration
        self._dataloader_train.worker_init_fn=worker_init_fn
        self._dataloader_val.worker_init_fn=worker_init_fn
        
        if dist.get_world_size()>1:
            logger.debug("checking distributed sampler in train and val set")
            assert isinstance(self._dataloader_train.sampler, DistributedSampler), "Please use DistributedSampler when DDP is enabled"
            if isinstance(self._dataloader_val.sampler, DistributedSampler):
                # https://discuss.pytorch.org/t/a-question-about-model-test-in-ddp/140456
                warnings.warn("DistributedSampler is used for valloader, of which the behavior may lead to incorrect metrics when #batch is not divisible by #gpu.")

    def reg_test_dataset(self, test: Any, testloader: DataLoader):
        self._data_test = test
        self._dataloader_test = testloader
        logger.debug("registered test set. enabled test phase.")
        self._set_flag('test_data_set')

        self._dataloader_test.worker_init_fn=worker_init_fn

        logger.debug("checking distributed sampler in test set")
        if isinstance(self._dataloader_test.sampler, DistributedSampler):
            # https://discuss.pytorch.org/t/a-question-about-model-test-in-ddp/140456
            warnings.warn("DistributedSampler is used for testloader, of which the behavior may lead to incorrect metrics when #batch is not divisible by #gpu.")
    
    def reg_model(self,model:nn.Module):
        self._model=model
    
    def prepare(self,rank:int):
        self._model=self._model.to(self.dev)
        self._raw_model=self._model
        if self.dev!='cpu':
            self._model=nn.parallel.DistributedDataParallel(self._model,device_ids=[rank],find_unused_parameters=self._diagnose)
        else:
            self._model=nn.parallel.DistributedDataParallel(self._model,find_unused_parameters=self._diagnose)

    
    def _reg_optimizer(self,optimizer:Optimizer):
        self._optimizer=optimizer
    
    def _reg_scheduler(self,scheduler:LRScheduler):
        self._scheduler=scheduler
    
    def update_tb(self, epochi:int, phase: PhaseEnv, tbwriter:SummaryWriter):
        assert phase._phase_name in ["train", "val", "test"]
        # only log probes in main process
        average_loss=phase.loss_probe.average()
        if dist.is_main_process():
            tbwriter.add_scalar(
                f"loss/{phase._phase_name}", average_loss, epochi
            )
        
        average_score=phase.score()
        if dist.is_main_process():
            tbwriter.add_scalar(f"score/{phase._phase_name}", average_score, epochi) # ????? stuck here
        
        for k in self._custom_probes:
            if phase.custom_probes[k].has_data():
                average_value=phase.custom_probes[k].average()
                if dist.is_main_process():
                    tbwriter.add_scalar(
                        f"{k}/{phase._phase_name}",
                        average_value,
                        epochi,
                    )

# worker seed init in different threads
def worker_init_fn(worker_id):
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
