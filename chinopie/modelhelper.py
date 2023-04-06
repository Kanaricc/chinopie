from datetime import datetime
import os, sys, shutil
import argparse
import random
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence


import torch
from torch import nn
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
import optuna
from optuna.distributions import CategoricalChoiceType
import numpy as np
from loguru import logger

from .probes.avgmeter import AverageMeter
from .datasets.fakeset import FakeEmptySet
from .ddpsession import DdpSession
from .filehelper import FileHelper
from .phasehelper import (
    PhaseHelper,
)
from .utils import show_params_in_3cols,create_snapshot,check_gitignore

# LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"


# TrainHelper has no state

class TrainHelper:
    def __init__(
        self,
        trial: optuna.Trial,
        arg_str:Sequence[str],
        disk_root: str,
        comment: str,
        dev: str,
        ddp_session:Optional[DdpSession],
        global_params:Dict[str,Any],
    ) -> None:
        self.trial = trial
        self._arg_str=arg_str
        self._comment = comment
        self._ddp_session = ddp_session
        self._argparser=argparse.ArgumentParser()
        self._test_phase_enabled = False

        self.file = FileHelper(disk_root, comment, self._ddp_session)

        if self._ddp_session:
            logger.info(
                f"[DDP] ddp is enabled. current rank is {self._ddp_session.get_rank()}."
            )
            logger.info(
                f"[DDP] use `{self.dev}` for this process. but you may use other one you want."
            )

            if self._is_main_process():
                logger.info(
                    f"[DDP] rank {self._ddp_session.get_rank()} is the leader. methods are fully enabled."
                )
            else:
                logger.info(
                    f"[DDP] rank {self._ddp_session.get_rank()} is the follower. some methods are disabled."
                )

            world_size = self._ddp_session.get_world_size()
            assert world_size != -1, "helper must be init after dist"
            if world_size <= torch.cuda.device_count():
                self.dev = f"cuda:{self._ddp_session.get_rank()}"
            else:
                logger.warning(
                    f"[DDP] world_size is larger than the number of devices. assume use CPU."
                )
                self.dev = f"cpu:{self._ddp_session.get_rank()}"
        else:
            if dev == "":
                if torch.cuda.is_available():
                    self.dev = "cuda"
                    logger.info("use cuda as default device")
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

    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()

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

        if self._ddp_session:
            assert isinstance(self._dataloader_train.sampler, DistributedSampler)
            assert not isinstance(self._dataloader_val.sampler, DistributedSampler)
            logger.debug("ddp enabled, checked distributed sampler in train and val set")

    def register_test_dataset(self, test: Any, testloader: DataLoader):
        self._data_test = test
        self._dataloader_test = testloader
        logger.debug("registered test set. enabled test phase.")
        self._set_flag('test_data_set')

        if self._ddp_session:
            assert not isinstance(self._dataloader_test.sampler, DistributedSampler)
            logger.debug("ddp enabled, checked distributed sampler in test set")
    
    def reg_model(self,model:nn.Module):
        self._model=model
        # TODO: move device
    
    def _reg_optimizer(self,optimizer:Optimizer):
        self._optimizer=optimizer

    def set_fixed_seed(self, seed: Any, disable_ddp_seed=False):
        if not self._ddp_session or disable_ddp_seed:
            logger.info("[HELPER] fixed seed set for random and torch")
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)
        else:
            logger.info(
                f"[HELPER|DDP] fixed seed `{seed + self._ddp_session.get_rank()}` set for this process"
            )
            os.environ["PYTHONHASHSEED"] = str(seed + self._ddp_session.get_rank())
            random.seed(seed + self._ddp_session.get_rank())

            torch.manual_seed(seed + self._ddp_session.get_rank())
            torch.cuda.manual_seed(seed + self._ddp_session.get_rank())
            torch.cuda.manual_seed_all(seed + self._ddp_session.get_rank())

            np.random.seed(seed + self._ddp_session.get_rank())

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
            assert fixed_val >= low and fixed_val <= high
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
            assert fixed_val >= low and fixed_val <= high
            logger.debug(f"using fixed param `{name}`")
            return fixed_val
        else:
            logger.debug(f"suggesting dynamic param `{name}`")
            return self.trial.suggest_float(name, low, high, step=step, log=log)
    
    def update_tb(self, epochi:int, phase: PhaseHelper, tbwriter:SummaryWriter):
        assert phase._phase_name in ["train", "val", "test"]
        if self._ddp_session:
            phase.loss_probe._sync_dist_nodes()
            phase._score._sync_dist_nodes()
            for k in phase.custom_probes:
                phase.custom_probes[k]._sync_dist_nodes()

        # only log probes in main process
        if self._is_main_process():
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
        self._enable_ddp = False
        self._enable_prune=enable_prune
        if self._enable_prune:
            logger.info('early stop is enabled')

        self._custom_params:Dict[str,Any]={}

        if self._enable_ddp:
            self._init_ddp()
        else:
            self._ddp_session=None
        self._init_logger(args.verbose)
        check_gitignore([self._disk_root])
        

        if diagnose:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)
            logger.info("diagnose mode enabled")
        if enable_snapshot:
            if not diagnose:
                create_snapshot(self._comment)
                logger.info("created snapshot")
            else:
                logger.info("snapshot is disabled in diagnose mode")
    
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
        self._ddp_session = DdpSession()
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
        if self._ddp_session:
            logger.remove()
            logger.add(sys.stderr, level=stdout_level, format=LOGGER_FORMAT)
            logger.add(
                f"logs/log_{self._comment}_r{self._ddp_session.get_rank()}.log",
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
        logger.warning(f"[HYPERPARAMETERS]\n{show_params_in_3cols(helper.trial.params)}")

    def optimize(
        self, recipe:ModuleRecipe, n_trials: int, stage:Optional[int]=None,inf_score:float=0,
    ):
        self._flush_params()
                
        if stage is None:
            stage_comment=self._comment
        else:
            stage_comment=f"{self._comment}({stage})"
        
        if not os.path.exists("opts"):
            os.mkdir("opts")
        storage_path = os.path.join("opts", f"{stage_comment}.db")
        # do not save storage in diagnose mode
        if self._diagnose_mode:
            storage_path=None
        else:
            storage_path=f"sqlite:///{storage_path}"
        
        self._inf_score=inf_score
        study = optuna.create_study(storage=storage_path)
        
        # in diagnose mode, run 3 times only
        if self._diagnose_mode:
            n_trials=3
        study.optimize(lambda x: self._wrapper(x,recipe,stage_comment), n_trials=n_trials, gc_after_trial=True)

        if self._diagnose_mode:
            # remove checkpoints and boards
            if os.path.exists(self.helper.file.ckpt_dir):
                shutil.rmtree(self.helper.file.ckpt_dir)
            if os.path.exists(self.helper.file.default_board_dir):
                shutil.rmtree(self.helper.file.default_board_dir)
            logger.info("removed ckpt and board")
            logger.warning("========== Diagnose End ==========")
        else:
            best_params = study.best_params
            best_value = study.best_value
            best_trial=study.best_trial
            logger.warning("[BOOPSTRAP] finish optimization")
            logger.warning(
                f"[BOOTSTRAP] best hyperparameters\n{show_params_in_3cols(best_params)}"
            )
            logger.warning(f"[BOOTSTRAP] best score: {best_value}")

            target_helper=FileHelper(self._disk_root,f"{stage_comment}")
            shutil.copytree(self.tbwriter.log_dir,target_helper.default_board_dir)
            shutil.copytree(self.helper.file.ckpt_dir,target_helper.ckpt_dir)
            logger.info("copied best trial as the final result")
        logger.warning("[BOOTSTRAP] good luck!")

    def _wrapper(self, trial: optuna.Trial, recipe:ModuleRecipe, comment:str) -> float | Sequence[float]:
        self.helper = TrainHelper(
            trial,
            arg_str=self._extra_arg_str,
            disk_root=self._disk_root,
            comment=f"{comment}_trial{trial._trial_id}",
            dev=self._dev,
            ddp_session=self._ddp_session,
            global_params=self._custom_params,
        )
        recipe._set_helper(self.helper)
        recipe.prepare(self.helper)
        self.helper._reg_optimizer(recipe.set_optimizers(self.helper._model,self.helper))

        best_score=self._inf_score
        # check diagnose mode
        if self._diagnose_mode:
            logger.info("diagnose mode is enabled. run 2 epochs.")
            self._epoch_num = 2
        
        recovered_epoch=None
        recovered_board_dir=None
        if self._load_checkpoint:
            latest_ckpt_path=self.helper.file.find_latest_checkpoint()
            if latest_ckpt_path is not None:
                logger.info(f"found latest checkpoint at `{latest_ckpt_path}`")
            else:
                logger.info(f"no checkpoint found")

            if self._load_checkpoint and latest_ckpt_path is not None:
                state=recipe.restore_ckpt(latest_ckpt_path)
                recovered_epoch=state['cur_epochi']
                best_score=state['best_score']
                recovered_board_dir=state['board_dir']
        
        assert self.helper._get_flag('trainval_data_set'), "train or val set not set"
        if not self.helper._get_flag('test_data_set'):
            logger.warning("test set not set. test phase will be skipped.")
        
        # create checkpoint dir
        self.helper.file.prepare_checkpoint_dir()
        # create board dir before training
        board_dir=recovered_board_dir if recovered_board_dir is not None else self.helper.file.default_board_dir
        self.tbwriter = SummaryWriter(board_dir)
        self._report_info(helper=self.helper,board_dir=self.tbwriter.log_dir)
        if self._ddp_session:
            self._ddp_session.barrier()
        logger.warning("ready to train model")
        for epochi in range(self._epoch_num):
            if not self._ddp_session:
                logger.warning(f"=== START EPOCH {epochi} ===")
            else:
                logger.warning(
                    f"=== RANK {self._ddp_session.get_rank()} START EPOCH {epochi} ==="
                )
            
            recipe.before_epoch()
            if recovered_epoch is not None and epochi <= recovered_epoch:
                recipe.after_epoch()
                logger.info(f"[HELPER] fast pass epoch {recovered_epoch}")
                continue
            
            phase=PhaseHelper(
                "train",
                self.helper._data_train,
                self.helper._dataloader_train,
                ddp_session=None,
                dry_run=self._diagnose_mode,
                custom_probes=self.helper._custom_probes.copy(),
                dev=self.helper.dev
            )
            recipe.run_train_phase(phase)
            phase._check_update()
            self._end_phase(epochi,phase)

            phase=PhaseHelper(
                "val",
                self.helper._data_val,
                self.helper._dataloader_val,
                ddp_session=None,
                dry_run=self._diagnose_mode,
                custom_probes=self.helper._custom_probes.copy(),
                dev=self.helper.dev
            )
            recipe.run_val_phase(phase)
            phase._check_update()
            score=phase.score
            self._end_phase(epochi,phase)

            if self.helper._test_phase_enabled:
                phase=PhaseHelper(
                    "val",
                    self.helper._data_val,
                    self.helper._dataloader_val,
                    ddp_session=None,
                    dry_run=self._diagnose_mode,
                    custom_probes=self.helper._custom_probes.copy(),
                    dev=self.helper.dev
                )
                recipe.run_train_phase(phase)
                phase._check_update()
                score=phase.score
                self._end_phase(epochi,phase)
            recipe.after_epoch()
            assert type(score)==float

            # check if ckpt is need
            need_save_period = epochi % self._checkpoint_save_period == 0
            if score >= best_score:
                best_score=score
                need_save_best=True
            else:
                need_save_best=False
            if self.helper._is_main_process() and self._save_checkpoint:
                # force saving ckpt in diagnose mode
                if self._diagnose_mode:
                    need_save_best=True
                    need_save_period=True
                state={
                    'cur_epochi':epochi,
                    'best_score':best_score,
                    'board_dir':self.tbwriter.log_dir,
                }
                if need_save_period:recipe.save_ckpt(self.helper.file.get_checkpoint_slot(epochi),extra_state=state)
                if need_save_best:recipe.save_ckpt(self.helper.file.get_best_checkpoint_slot(),extra_state=state)
            if self._ddp_session:
                self._ddp_session.barrier()
            
            trial.report(score,epochi)
            # early stop
            if self._enable_prune and trial.should_prune():
                raise optuna.TrialPruned()

        return best_score    
    
    def _end_phase(self,epochi:int,phase:PhaseHelper):
        if not self._ddp_session:
            logger.warning(
                f"|| end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )
        else:
            logger.warning(
                f"|| RANK {self._ddp_session.get_rank()} end {phase._phase_name} {epochi} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
