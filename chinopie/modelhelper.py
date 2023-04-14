from datetime import datetime
import os, sys, shutil
import argparse
import random
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence


import torch
import torch.backends.mps
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import optuna
from optuna.distributions import CategoricalChoiceType
import numpy as np
from loguru import logger

from .probes.avgmeter import AverageMeter
from .datasets.fakeset import FakeEmptySet
from . import iddp as dist
from .filehelper import GlobalFileHelper,InstanceFileHelper
from .phasehelper import (
    PhaseHelper,
)
from .utils import show_params_in_3cols,create_snapshot,check_gitignore,any_to

# LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"


# TrainHelper has no state

class TrainHelper:
    def __init__(
        self,
        trial: optuna.Trial,
        arg_str:Sequence[str],
        file_helper: InstanceFileHelper,
        prev_file_helper:Optional[InstanceFileHelper],
        dev: str,
        global_params:Dict[str,Any],
    ) -> None:
        self.trial = trial
        self._arg_str=arg_str
        self.file=file_helper
        self.prev_file=prev_file_helper

        if dist.is_enabled():
            logger.info(
                f"[DDP] ddp is enabled. current rank is {dist.get_rank()}."
            )
            logger.info(
                f"[DDP] use `{self.dev}` for this process. but you may use other one you want."
            )

            if dist.is_main_process():
                logger.info(
                    f"[DDP] rank {dist.get_rank()} is the leader. methods are fully enabled."
                )
            else:
                logger.info(
                    f"[DDP] rank {dist.get_rank()} is the follower. some methods are disabled."
                )

            world_size = dist.get_world_size()
            assert world_size != -1, "helper must be init after dist"
            if world_size <= torch.cuda.device_count():
                self.dev = f"cuda:{dist.get_rank()}"
            else:
                logger.warning(
                    f"[DDP] world_size is larger than the number of devices. assume use CPU."
                )
                self.dev = f"cpu:{dist.get_rank()}"
        else:
            if dev == "":
                if torch.cuda.is_available():
                    self.dev = "cuda"
                    logger.info("cuda found. use cuda as default device")
                elif torch.backends.mps.is_available():
                    self.dev = "mps"
                    logger.info("mps found. use mps as default device")
                else:
                    self.dev = "cpu"
                    logger.info("use CPU as default device")
            else:
                self.dev = dev
                logger.info(f"use custom device `{dev}`")

        self._custom_probes = []
        self._global_params=global_params
        self._flags={}
    
    def _set_flag(self,key:str,val:Any=True):
        self._flags[key]=val
    
    def _get_flag(self,key:str):
        return self._flags[key] if key in self._flags else None

    def register_probe(self, name: str):
        self._custom_probes.append(name)
        logger.debug(f"register probe `{name}`")


    def register_dataset(
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

        if dist.is_enabled():
            assert isinstance(self._dataloader_train.sampler, DistributedSampler)
            assert not isinstance(self._dataloader_val.sampler, DistributedSampler)
            logger.debug("ddp enabled, checked distributed sampler in train and val set")

    def register_test_dataset(self, test: Any, testloader: DataLoader):
        self._data_test = test
        self._dataloader_test = testloader
        logger.debug("registered test set. enabled test phase.")
        self._set_flag('test_data_set')

        self._dataloader_test.worker_init_fn=worker_init_fn

        if dist.is_enabled():
            assert not isinstance(self._dataloader_test.sampler, DistributedSampler)
            logger.debug("ddp enabled, checked distributed sampler in test set")
    
    def reg_model(self,model:nn.Module):
        self._model=model.to(self.dev)
    
    def _reg_optimizer(self,optimizer:Optimizer):
        self._optimizer=optimizer
    
    def _reg_scheduler(self,scheduler:LRScheduler):
        self._scheduler=scheduler

    def suggest_category(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        assert name in self._global_params, f"request for unregisted param `{name}`"
        fixed_val = self._global_params[name]
        if fixed_val is not None:
            assert fixed_val in choices
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            return self.trial.suggest_categorical(name, choices)

    def suggest_int(self, name: str, low: int, high: int, step=1, log=False) -> int:
        assert name in self._global_params, f"request for unregisted param `{name}`"
        fixed_val = self._global_params[name]
        if fixed_val is not None:
            if fixed_val< low or fixed_val>high:
                logger.warning(f"fixed val {fixed_val} of {name} is out of interval")
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            return self.trial.suggest_int(name, low, high, step, log)

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log=False,
    ) -> float:
        assert name in self._global_params, f"request for unregisted param `{name}`"
        fixed_val = self._global_params[name]
        if fixed_val is not None:
            if fixed_val< low or fixed_val>high:
                logger.warning(f"fixed val {fixed_val} of {name} is out of interval")
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            return self.trial.suggest_float(name, low, high, step=step, log=log)
    
    def update_tb(self, epochi:int, phase: PhaseHelper, tbwriter:SummaryWriter):
        assert phase._phase_name in ["train", "val", "test"]
        if dist.is_enabled():
            phase.loss_probe._sync_dist_nodes()
            phase._score._sync_dist_nodes()
            for k in phase.custom_probes:
                phase.custom_probes[k]._sync_dist_nodes()

        # only log probes in main process
        if dist.is_main_process():
            tbwriter.add_scalar(
                f"loss/{phase._phase_name}", phase.loss_probe.average(), epochi
            )
            tbwriter.add_scalar("score/train", phase.score, epochi)

            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    tbwriter.add_scalar(
                        f"{k}/{phase._phase_name}",
                        phase.custom_probes[k].average(),
                        epochi,
                    )
                    logger.info(
                        f"[{phase._phase_name} probes] {k}: {phase.custom_probes[k].average()}"
                    )

# worker seed init in different threads
def worker_init_fn(worker_id):
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# all block sync should be done in bootstrap. all data sync should be done in helper.

from .recipe import ModuleRecipe
class TrainBootstrap:
    def __init__(
        self,
        disk_root: str,
        epoch_num: int,
        load_checkpoint: bool,
        save_checkpoint: bool,
        comment: Optional[str],
        checkpoint_save_period: int = 1,
        enable_snapshot=False,
        enable_prune=False,
        seed:Optional[Any]=None,
        dev="",
        diagnose=False,
        verbose=False,
    ) -> None:
        argparser=argparse.ArgumentParser(
            prog='ChinoPie'
        )
        argparser.add_argument('-r','--disk_root',type=str,default=disk_root)
        argparser.add_argument('-e','--epoch_num',type=int,default=epoch_num)
        argparser.add_argument('-l','--load_checkpoint',action='store_true',default=load_checkpoint)
        argparser.add_argument('-s','--save_checkpoint',action='store_true',default=save_checkpoint)
        argparser.add_argument('-c','--comment',type=str,default=comment)
        argparser.add_argument('--dev',type=str,default=dev)
        argparser.add_argument('-d','--diagnose',action='store_true',default=diagnose)
        argparser.add_argument('-v','--verbose',action='store_true',default=verbose)
        argparser.add_argument('--clear',action='store_true',default=verbose)
        args,self._extra_arg_str=argparser.parse_known_args()
        self._argparser=argparse.ArgumentParser()
        

        self._disk_root = args.disk_root
        self._epoch_num = args.epoch_num
        if args.comment is not None:
            self._comment = args.comment
        else:
            self._comment = datetime.now().strftime("%Y%m%d%H%M%S")
        self._load_checkpoint:bool = args.load_checkpoint
        self._save_checkpoint:bool = args.save_checkpoint
        self._checkpoint_save_period = checkpoint_save_period
        self._dev = args.dev
        self._diagnose_mode = args.diagnose
        self._enable_prune=enable_prune

        self._init_logger(args.verbose)

        self.file=GlobalFileHelper(disk_root)
        self._custom_params:Dict[str,Any]={}

        # set prune
        if self._enable_prune:
            logger.info('early stop is enabled')

        # set clear
        if args.clear:
            # do clear
            input("are you sure to clear all state files and logs? (press ctrl+c to quit)")
            self.clear()
        
        # check git ignore
        check_gitignore([self._disk_root])

        # set diagnose mode
        if diagnose:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)
            logger.info("diagnose mode enabled")
        
        # set snapshot
        if enable_snapshot:
            if not diagnose:
                create_snapshot(self._comment)
                logger.info("created snapshot")
            else:
                logger.info("snapshot is disabled in diagnose mode")
        
        # set fixed seed
        if seed is not None:
            self.set_fixed_seed(seed)
        
        self._inherit_states:Dict[str,Any]={}
    
    def release(self):
        if self._diagnose_mode:
            self.file.clear_all_instance()
            logger.info("deleted trial files in diagnose mode")
    
    def clear(self):
        if os.path.exists('logs'):shutil.rmtree('logs')
        if os.path.exists('opts'):shutil.rmtree('opts')
    
    def set_fixed_seed(self, seed: Any, ddp_seed=True):
        if not dist.is_enabled() or not ddp_seed:
            set_fixed_seed(seed)
        else:
            set_fixed_seed(seed+dist.get_rank())
            logger.info("ddp detected, use different seed")
    
    def reg_category(self, name: str, value: Optional[CategoricalChoiceType] = None):
        if name not in self._custom_params:
            self._argparser.add_argument(f"--{name}",required=False)
            self._custom_params[name] = value

    def reg_int(self, name: str, value: Optional[int] = None):
        if name not in self._custom_params:
            self._argparser.add_argument(f"--{name}",type=int,required=False)
            self._custom_params[name] = value

    def reg_float(self, name: str, value: Optional[float] = None):
        if name not in self._custom_params:
            self._argparser.add_argument(f"--{name}",type=float,required=False)
            self._custom_params[name] = value
    
    def _flush_params(self):
        args=self._argparser.parse_args(self._extra_arg_str)
        logger.debug(f"hyperparameters in argparser: {args}")
        for k in self._custom_params.keys():
            if getattr(args,k) is not None:
                self._custom_params[k]=getattr(args,k)
                logger.debug(f"flushed `{k}`")
        

    def _init_ddp(self):
        logger.info("initialized ddp")

    def _init_logger(self,verbose:bool):
        stdout_level="INFO"
        file_level="DEBUG"
        if verbose:
            stdout_level="DEBUG"
            file_level="TRACE"
        # logger file
        if not os.path.exists("logs"):
            os.mkdir("logs")
        if dist.is_enabled():
            logger.remove()
            logger.add(sys.stderr, level=stdout_level, format=LOGGER_FORMAT)
            logger.add(
                f"logs/log_{self._comment}_r{dist.get_rank()}.log",
                level=file_level,
                format=LOGGER_FORMAT,
            )
        else:
            logger.remove()
            logger.add(sys.stderr, level=stdout_level, format=LOGGER_FORMAT)
            logger.add(f"logs/log_{self._comment}.log",level=file_level, format=LOGGER_FORMAT)
        logger.info("initialized logger")
    
    def _report_info(self,helper:TrainHelper,board_dir:str):
        dataset_str = f"train({len(helper._data_train)}) val({len(helper._data_val)}) test({len(helper._data_test) if hasattr(helper, '_data_test') else 'not set'})"
        table = show_params_in_3cols(
            params={
                "proper device": self._dev,
                "diagnose": self._diagnose_mode,
                "epoch num": self._epoch_num,
                "early stop": self._enable_prune,
                "dataset": dataset_str,
                "board dir": board_dir,
                "checkpoint load/save": f"{self._load_checkpoint}/{self._save_checkpoint}",
                "custom probes": helper._custom_probes,
            }
        )
        logger.warning(f"[INFO]\n{table}")
        logger.warning(f"[HYPERPARAMETERS]\n{show_params_in_3cols(self._custom_params|helper.trial.params)}")

    def optimize(
        self, recipe:ModuleRecipe,direction:str, n_trials: int, stage:Optional[int]=None,inf_score:float=0,
    ):
        self._flush_params()
                
        if stage is None:
            stage_comment=self._comment
        else:
            stage_comment=f"{self._comment}({stage})"
        # find previous file helper
        if stage and stage>0:
            prev_file_helper=self.file.get_exp_instance(f"{self._comment}({stage-1})")
            logger.debug("found previous file helper")
        else:
            prev_file_helper=None

        self.study_file=self.file.get_exp_instance(stage_comment)
        if not os.path.exists("opts"):
            os.mkdir("opts")
        storage_path = os.path.join("opts", f"{stage_comment}.db")
        # do not save storage in diagnose mode
        if self._diagnose_mode:
            storage_path=None
        else:
            storage_path=f"sqlite:///{storage_path}"
        
        self._inf_score=inf_score
        self._best_trial_score=inf_score
        study = optuna.create_study(study_name=stage_comment,direction=direction,storage=storage_path,load_if_exists=True)

        finished_trials=set()
        for trial in study.trials:
            if trial.state==optuna.trial.TrialState.FAIL:
                logger.info(f"found failed trial {trial._trial_id}, resuming")
                study.enqueue_trial(trial.params,{'trial_id':trial._trial_id})
            elif trial.state==optuna.trial.TrialState.COMPLETE or trial.state==optuna.trial.TrialState.PRUNED:
                if 'trial_id' not in trial.user_attrs:
                    finished_trials.add(trial._trial_id)
                else:
                    finished_trials.add(trial.user_attrs['trial_id'])
        logger.debug(f"found finished trials {finished_trials}. the requested #trials is {n_trials}")
        if len(finished_trials)==n_trials:
            logger.warning(f"this study is already finished")
            logger.warning(
                f"best hyperparameters\n{show_params_in_3cols(study.best_params)}"
            )
            logger.warning(f"best score: {study.best_value}")
            return
        
        # in diagnose mode, run 1 times only
        if self._diagnose_mode:
            n_trials=1

        

        try:
            study.optimize(lambda x: self._wrapper(x,recipe,prev_file_helper,self._inherit_states,stage_comment), n_trials=n_trials, callbacks=[self._hook_trial_end], gc_after_trial=True)
        except optuna.TrialPruned:
            pass
        finally:
            # post process
            best_params = study.best_params
            best_value = study.best_value
            best_trial=study.best_trial
            logger.warning("[BOOPSTRAP] finish optimization")
            logger.warning(
                f"[BOOTSTRAP] best hyperparameters\n{show_params_in_3cols(best_params)}"
            )
            logger.warning(f"[BOOTSTRAP] best score: {best_value}")

            best_file=self.file.get_exp_instance(f"{stage_comment}_trial{best_trial._trial_id}")
            target_helper=self.file.get_exp_instance(stage_comment)
            shutil.copytree(best_file.default_board_dir,target_helper.default_board_dir)
            shutil.copytree(best_file.ckpt_dir,target_helper.ckpt_dir)
            logger.info("copied best trial as the final result")
        
        logger.warning("[BOOTSTRAP] good luck!")

    def _wrapper(self, trial: optuna.Trial, recipe:ModuleRecipe, prev_file_helper:Optional[InstanceFileHelper], inherit_states:Dict[str,Any], comment:str) -> float | Sequence[float]:
        trial_id=trial._trial_id if 'trial_id' not in trial.user_attrs else trial.user_attrs['trial_id']
        logger.info(f"this is trial {trial._trial_id}, real id {trial_id}")
        trial_file=self.file.get_exp_instance(f"{comment}_trial{trial_id}")
        self.helper = TrainHelper(
            trial,
            arg_str=self._extra_arg_str,
            file_helper=trial_file,
            prev_file_helper=prev_file_helper,
            dev=self._dev,
            global_params=self._custom_params,
        )
        recipe._set_helper(self.helper)
        recipe.prepare(self.helper,inherit_states)
        # set optimizer
        self.helper._reg_optimizer(recipe.set_optimizers(self.helper._model,self.helper))
        _scheduler=recipe.set_scheduler(self.helper._optimizer)
        if _scheduler is not None:
            self.helper._reg_scheduler(_scheduler)
        del _scheduler

        best_score=self._inf_score
        # check diagnose mode
        if self._diagnose_mode:
            logger.info("diagnose mode is enabled. run 2 epochs only.")
            self._epoch_num = 2
        
        recovered_epoch=None
        if self._load_checkpoint:
            latest_ckpt_path=trial_file.find_latest_checkpoint()
            if latest_ckpt_path is not None:
                logger.info(f"found latest checkpoint at `{latest_ckpt_path}`")
            else:
                logger.info(f"no checkpoint found")

            if self._load_checkpoint and latest_ckpt_path is not None:
                state=recipe.restore_ckpt(latest_ckpt_path)
                recovered_epoch=state['cur_epochi']
                best_score=state['best_score']
        
        assert self.helper._get_flag('trainval_data_set'), "train or val set not set"
        if not self.helper._get_flag('test_data_set'):
            logger.warning("test set not set. test phase will be skipped.")
        
        # create checkpoint dir
        trial_file.prepare_checkpoint_dir()
        # create board dir before training
        tbwriter = SummaryWriter(trial_file.default_board_dir)
        self._report_info(helper=self.helper,board_dir=tbwriter.log_dir)
        if dist.is_enabled():
            dist.barrier()
        logger.warning("ready to train model")
        for epochi in range(self._epoch_num):
            self._cur_epochi=epochi
            if not dist.is_enabled():
                logger.warning(f"=== START EPOCH {epochi} ===")
            else:
                logger.warning(
                    f"=== RANK {dist.get_rank()} START EPOCH {epochi} ==="
                )
            
            recipe.before_epoch()
            if recovered_epoch is not None and epochi <= recovered_epoch:
                recipe.after_epoch()
                logger.info(f"[HELPER] fast pass epoch {recovered_epoch}")
                continue
            
            self._prepare_dataloader_for_epoch(self.helper._dataloader_train)
            phase_train=PhaseHelper(
                "train",
                self.helper._data_train,
                self.helper._dataloader_train,
                dry_run=self._diagnose_mode,
                custom_probes=self.helper._custom_probes.copy(),
                dev=self.helper.dev
            )
            recipe.run_train_phase(phase_train)
            phase_train._check_update()
            self._end_phase(epochi,phase_train)

            self._prepare_dataloader_for_epoch(self.helper._dataloader_val)
            phase_val=PhaseHelper(
                "val",
                self.helper._data_val,
                self.helper._dataloader_val,
                dry_run=self._diagnose_mode,
                custom_probes=self.helper._custom_probes.copy(),
                dev=self.helper.dev
            )
            recipe.run_val_phase(phase_val)
            phase_val._check_update()
            score=phase_val.score
            self._end_phase(epochi,phase_val)

            if self.helper._get_flag('test_data_set'):
                self._prepare_dataloader_for_epoch(self.helper._dataloader_test)
                phase_test=PhaseHelper(
                    "test",
                    self.helper._data_test,
                    self.helper._dataloader_test,
                    dry_run=self._diagnose_mode,
                    custom_probes=self.helper._custom_probes.copy(),
                    dev=self.helper.dev
                )
                recipe.run_train_phase(phase_test)
                phase_test._check_update()
                score=phase_test.score
                self._end_phase(epochi,phase_test)
            else: phase_test=None
            recipe.after_epoch()
            assert type(score)==float

            # output final result of this epoch
            loss_msg=[
                phase_train.loss_probe.average(),
                phase_val.loss_probe.average(),
                phase_test.loss_probe.average() if phase_test else 0
            ]
            score_msg=[
                phase_train.score,
                phase_val.score,
                phase_test.score if phase_test else 0
            ]
            if not dist.is_enabled():
                logger.warning(
                    f"=== END EPOCH {epochi} - loss {'/'.join(map(lambda x: f'{x:.3f}',loss_msg))}, score {'/'.join(map(lambda x: f'{x:.3f}',score_msg))} ==="
                )
            else:
                logger.warning(
                    f"=== RANK {dist.get_rank()} END EPOCH {epochi} - loss {'/'.join(map(lambda x: f'{x:.3f}',loss_msg))}, score {'/'.join(map(lambda x: f'{x:.3f}',score_msg))} ==="
                )

            # check if ckpt is need
            need_save_period = epochi % self._checkpoint_save_period == 0
            if score >= best_score:
                best_score=score
                need_save_best=True
            else:
                need_save_best=False
            if dist.is_main_process() and self._save_checkpoint:
                # force saving ckpt in diagnose mode
                if self._diagnose_mode:
                    need_save_best=True
                    need_save_period=True
                state={
                    'cur_epochi':epochi,
                    'best_score':best_score,
                }
                if need_save_period:recipe.save_ckpt(trial_file.get_checkpoint_slot(epochi),extra_state=state)
                if need_save_best:recipe.save_ckpt(trial_file.get_best_checkpoint_slot(),extra_state=state)
            if dist.is_enabled():
                dist.barrier()
            
            trial.report(score,epochi)
            # early stop
            if self._enable_prune and trial.should_prune():
                raise optuna.TrialPruned()
            
            if self._check_instant_cmd()=='prune':
                logger.warning("breaking epoch")
                break
        
        self._latest_states=recipe.end(self.helper)
        
        return best_score
    
    def _prepare_dataloader_for_epoch(self,dataloader:DataLoader):
        if isinstance(dataloader.sampler,DistributedSampler):
            dataloader.sampler.set_epoch(self._cur_epochi)
    
    def _hook_trial_end(self,study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.state==optuna.trial.TrialState.PRUNED:
            return
        assert trial.value is not None
        if trial.value>=self._best_trial_score:
            self._best_trial_score=trial.value
            self._inherit_states=self._latest_states
            logger.info("update inherit states")
        del self._latest_states
    
    def _end_phase(self,epochi:int,phase:PhaseHelper):
        if not dist.is_enabled():
            logger.warning(
                f"|| end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )
        else:
            logger.warning(
                f"|| RANK {dist.get_rank()} end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )
    
    def _check_instant_cmd(self):
        logger.debug("checking instant cmd")
        if os.path.exists('instant_cmd'):
            with open('instant_cmd','r') as f:
                full_cmd=f.read().strip()
            os.remove('instant_cmd')
            if full_cmd=="prune":
                logger.warning("received command 'prune', stopping current trial!")
                return 'prune'
            else:
                return None
        else:
            return None

def set_fixed_seed(seed:Any):
    logger.info("fixed seed set for random, torch, and numpy")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
