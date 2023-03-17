import os, sys,shutil
import argparse
import random
import inspect
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
import numpy as np
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from prettytable import PrettyTable,PLAIN_COLUMNS
from torch.utils.data.distributed import DistributedSampler
from loguru import logger

from .probes.avgmeter import AverageMeter, NumericMeter
from .datasets.fakeset import FakeEmptySet
from .ddpsession import DdpSession
from .filehelper import FileHelper
from .phasehelper import CheckpointLoadSection, CheckpointSaveSection, PhaseHelper,FunctionalSection
from .utils import show_params_in_3cols
from .snapshot import create_snapshot

LOGGER_FORMAT='<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'


class TrainHelper:
    def __init__(
            self,
            disk_root: str,
            epoch_num: int,
            batch_size: int,
            load_checkpoint: bool,
            enable_checkpoint: bool,
            checkpoint_save_period: Optional[int],
            comment: str,
            dev: str = "",
            enable_ddp=False,
            enable_snapshots=False,
            diagnose: bool = False,
    ) -> None:
        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._load_checkpoint_enabled = load_checkpoint
        self._save_checkpoint_enabled = enable_checkpoint
        self._snapshots_enabled=enable_snapshots
        if checkpoint_save_period is not None:
            self._checkpoint_save_period = checkpoint_save_period
        else:
            self._checkpoint_save_period = 1
        self._comment = comment
        self._ddp_session = DdpSession() if enable_ddp else None

        # init diagnose flags
        self._has_checkpoint_load_section=False
        self._has_train_phase=False
        self._has_val_phase=False
        self._has_test_phase=False
        self._has_checkpoint_save_section=False

        if self._ddp_session:
            self._ddp_session.barrier()

        self.file = FileHelper(disk_root, comment, self._ddp_session)
        self._board_dir = self.file.get_default_board_dir()

        if self._ddp_session:
            world_size = self._ddp_session.get_world_size()
            assert world_size != -1, "helper must be init after dist"
            if world_size <= torch.cuda.device_count():
                self.dev = f"cuda:{self._ddp_session.get_rank()}"
            else:
                logger.warning(
                    f"[DDP] world_size is larger than the number of devices. assume use CPU."
                )
                self.dev = f"cpu:{self._ddp_session.get_rank()}"

            # logger file
            logger.remove()
            logger.add(sys.stderr, level="INFO",format=LOGGER_FORMAT)
            logger.add(f"log-{self._comment}({self._ddp_session.get_rank()}).log",format=LOGGER_FORMAT)

            logger.warning(f"[DDP] ddp is enabled. current rank is {self._ddp_session.get_rank()}.")
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
        else:
            # logger file
            logger.remove()
            logger.add(sys.stderr, level="INFO",format=LOGGER_FORMAT)
            logger.add(f"log-{self._comment}.log",format=LOGGER_FORMAT)

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
        self._custom_global_params: Dict[str, Any] = {}
        self._fastforward_handlers:List[Callable[[int],None]] = []
        self._diagnose_mode = diagnose

        if self._diagnose_mode:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)
            logger.info("diagnose mode enabled")

    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()
    
    def section_checkpoint_load(self):
        self._has_checkpoint_load_section=True
        return CheckpointLoadSection(lambda x:self._load_from_checkpoint(x),self._load_checkpoint_enabled)

    def section_checkpoint_save(self):
        self._has_checkpoint_save_section=True
        
        flag_save= self._is_main_process() and self._save_checkpoint_enabled and self._trigger_checkpoint
        flag_best = self._is_main_process() and self._save_checkpoint_enabled and self._trigger_best_score
        return CheckpointSaveSection(self._export_state(),flag_save,flag_best,not (flag_save or flag_best),lambda x:self._merge_section_flags(x))

    def _load_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        load checkpoint file, including model state, epoch index.
        """
        self._board_dir = checkpoint["board_dir"]
        self._recoverd_epoch = checkpoint["cur_epoch"]
        self._best_val_score = checkpoint["best_val_score"]
        logger.info(f"[HELPER] load state from checkpoint")

    def _export_state(self):
        return {
            "board_dir": self._board_dir,
            "cur_epoch": self.cur_epoch,
            "best_val_score": self._best_val_score,
        }

    def register_probe(self, name: str):
        self._custom_probes.append(name)

    def register_global_params(self, name: str, value: Any):
        self._custom_global_params[name] = value
    
    def register_fastforward_handler(self,func:Callable[[int],None]):
        self._fastforward_handlers.append(func)

    def register_dataset(
            self, train: Any, trainloader: DataLoader, val: Any, valloader: DataLoader
    ):
        self._data_train = train
        self._dataloader_train = trainloader
        self._data_val = val
        self._dataloader_val = valloader

        assert (
                self._dataloader_train.batch_size == self._batch_size
        ), f"batch size of dataloader_train does not match"
        if self._dataloader_val.batch_size != self._batch_size:
            logger.error("[HELPER] batch size of dataloader_val does not match")

        if self._ddp_session:
            assert isinstance(self._dataloader_train.sampler, DistributedSampler)
            assert not isinstance(self._dataloader_val.sampler, DistributedSampler)

    def register_test_dataset(self, test: Any, testloader: DataLoader):
        self._data_test = test
        self._dataloader_test = testloader

        assert (
                self._dataloader_test.batch_size == self._batch_size
        ), f"batch size of dataloader_test does not match"

        if self._ddp_session:
            assert not isinstance(self._dataloader_test.sampler, DistributedSampler)

    def set_fixed_seed(self, seed: Any, disable_ddp_seed=False):
        if not self._ddp_session or disable_ddp_seed:
            logger.info("[HELPER] fixed seed is set for random and torch")
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)
        else:
            logger.info(
                f"[HELPER|DDP] fixed seed `{seed + self._ddp_session.get_rank()}` is set for this process"
            )
            os.environ["PYTHONHASHSEED"] = str(seed + self._ddp_session.get_rank())
            random.seed(seed + self._ddp_session.get_rank())

            torch.manual_seed(seed + self._ddp_session.get_rank())
            torch.cuda.manual_seed(seed + self._ddp_session.get_rank())
            torch.cuda.manual_seed_all(seed + self._ddp_session.get_rank())

            np.random.seed(seed + self._ddp_session.get_rank())

    def get(self, name: str) -> Any:
        assert name in self._custom_global_params
        return self._custom_global_params[name]

    def set_dry_run(self):
        self._diagnose_mode = True

    def ready_to_train(self):
        if (
                not hasattr(self, "_data_train")
                or not hasattr(self, "_data_val")
                or not hasattr(self, "_dataloader_train")
                or not hasattr(self, "_dataloader_val")
        ):
            logger.error("[DATASET] dataset not set")
            raise RuntimeError("dataset not set")
        self._test_phase_enabled=hasattr(self, "_dataloader_test")
        if not self._test_phase_enabled:
            logger.warning("[DATASET] test dataset not set. pass test phase.")

        self.file.prepare_checkpoint_dir()
        # make sure all processes waiting till the dir is done
        if self._ddp_session:
            self._ddp_session.barrier()

        # create board dir before training
        self.tbwriter = SummaryWriter(self._board_dir)

        # to avoid flush checkpoint
        if not hasattr(self, "_best_val_score"):
            self._best_val_score = 0.0
        if not hasattr(self, "_best_test_score"):
            self._best_test_score = 0.0

        # section flag
        self._section_flags={}
        self.report_info()
        if self._snapshots_enabled and not self._diagnose_mode and self._is_main_process():
            create_snapshot(self._comment)

        logger.warning("[HELPER] ready to train model")
    
    def _merge_section_flags(self,x:Dict[str,Any]):
        self._section_flags|=x

    def report_info(self):
        dataset_str=f"train({len(self._data_train)}) val({len(self._data_val)}) test({len(self._data_test) if hasattr(self, '_data_test') else 'not set'})"
        table=show_params_in_3cols(params={
            "device":self.dev,
            "diagnose":self._diagnose_mode,
            "epoch num": self._epoch_num,
            "dataset": dataset_str,
            "board dir": self._board_dir,
            "checkpoint load/save": f"{self._load_checkpoint_enabled}/{self._save_checkpoint_enabled}",
            "custom probes": self._custom_probes,
        })
        logger.warning(f"[INFO]\n{table}")
    
    def _diagnose(self):
        logger.warning("==========Diagnose Results==========")
        if not self._has_checkpoint_load_section:
            if self._load_checkpoint_enabled:
                logger.error("checkpoint loading not found")
            else:
                logger.warning("checkpoint loading not found but is disabled")
        if not self._has_train_phase:
            logger.error("train phase not found")
        if not self._has_val_phase:
            logger.error("val phase not found")
        if not self._has_test_phase:
            if self._test_phase_enabled:
                logger.error("test phase not found")
            else:
                logger.warning("test phase not found but is disabled")
        if not self._has_checkpoint_save_section:
            if self._save_checkpoint_enabled:
                logger.error("checkpoint saving not found")
            else:
                logger.warning("checkpoint saving not found but is disabled")
        else:
            if 'checked_helper_state' not in self._section_flags:
                logger.error("helper state not checked")
            if 'checked_save_ckpt' not in self._section_flags:
                logger.error("checkpoint not checked")
            if 'checked_save_best' not in self._section_flags:
                logger.error("best result not checked")
            
        
        # remove checkpoints and boards
        shutil.rmtree(self.file.ckpt_dir)
        shutil.rmtree(self.file.board_dir)


    def range_epoch(self):
        if self._diagnose_mode:
            logger.info("you have enable dry-run mode")
            self._epoch_num = 2
        for i in range(self._epoch_num):
            if hasattr(self, "_recoverd_epoch") and i <= self._recoverd_epoch:
                for item in self._fastforward_handlers:
                    item(i)
                logger.info(f"[HELPER] fast pass epoch {self._recoverd_epoch}")
                continue
            self.cur_epoch = i

            # begin of epoch. set trigger.
            self._trigger_checkpoint = (
                    self._save_checkpoint_enabled
                    and self.cur_epoch % self._checkpoint_save_period == 0
            )
            self._trigger_best_score = False
            self._trigger_state_save = False
            self._trigger_run_test = False

            if not self._ddp_session:
                logger.warning(f"=== START EPOCH {self.cur_epoch} ===")
            else:
                logger.warning(
                    f"=== RANK {self._ddp_session.get_rank()} START EPOCH {self.cur_epoch} ==="
                )

            if self._ddp_session:
                assert isinstance(self._dataloader_train.sampler, DistributedSampler), "DistributedSampler not set for dataloader"
                self._dataloader_train.sampler.set_epoch(i)

            yield i
        if self._diagnose_mode:
            self._diagnose()


    def phase_train(self):
        self._has_train_phase=True

        return PhaseHelper(
            "train",
            self._data_train,
            self._dataloader_train,
            ddp_session=None,
            dry_run=self._diagnose_mode,
            exit_callback=self.end_phase_callback,
            custom_probes=self._custom_probes.copy(),
        )

    def phase_val(self):
        self._has_val_phase=True

        return PhaseHelper(
            "val",
            self._data_val,
            self._dataloader_val,
            ddp_session=None,
            dry_run=self._diagnose_mode,
            exit_callback=self.end_phase_callback,
            custom_probes=self._custom_probes.copy(),
        )

    def phase_test(self):
        self._has_test_phase=True

        need_run = self._test_phase_enabled and self._trigger_run_test
        return PhaseHelper(
            "test",
            self._data_test if hasattr(self, "_data_test") else FakeEmptySet(),
            self._dataloader_test
            if hasattr(self, "_dataloader_test")
            else DataLoader(FakeEmptySet()),
            ddp_session=None,
            dry_run=self._diagnose_mode,
            exit_callback=self.end_phase_callback,
            break_phase=not need_run,
            custom_probes=self._custom_probes.copy(),
        )
    
    def end_phase_callback(self,phase:PhaseHelper):
        assert phase._phase_name in ['train', 'val','test']
        # collect probes
        if self._ddp_session is None:
            output_dist_probes=phase._output_dist_probes
        else:
            gathered_output_dist_probes:List[List[NumericMeter]]=[]
            self._ddp_session.gather_object(phase._output_dist_probes,gathered_output_dist_probes,0)
            output_dist_probes=gathered_output_dist_probes[0]
            for i in gathered_output_dist_probes[1:]:
                for dst,src in zip(output_dist_probes,i):
                    dst.update(src.val)
            del gathered_output_dist_probes
        
        # only log probes in main process
        if self._is_main_process():
            # assume training loss is sync by user
            self.tbwriter.add_scalar(
                f"loss/{phase._phase_name}", phase.loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/train", phase.score, self.cur_epoch)

            for k,v in enumerate(output_dist_probes):
                if v.val.numel()>0:
                    self.tbwriter.add_histogram(f"outdist/{phase._phase_name}/{k}",v.val,self.cur_epoch)

            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        f"{k}/{phase._phase_name}", phase.custom_probes[k].average(), self.cur_epoch
                    )
                    logger.info(
                        f"[{phase._phase_name} probes] {k}: {phase.custom_probes[k].average()}"
                    )
        if phase._phase_name in ['train','val']:
            self._last_train_score=phase.score
            self._last_train_loss=phase.loss_probe.average()
            if not self._ddp_session:
                logger.warning(
                    f"|| end {phase._phase_name} {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
                )
            else:
                logger.warning(
                    f"|| RANK {self._ddp_session.get_rank()} end {phase._phase_name} {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
                )
        if phase._phase_name=='val':
            if phase.score >= self._best_val_score or self._diagnose_mode:
                self._best_val_score = phase.score
                self._trigger_run_test = True
                if self._save_checkpoint_enabled and not self._test_phase_enabled:
                    self._trigger_best_score = True
                    self._trigger_state_save = True
            
            logger.warning(
                f"||| END EPOCH {self.cur_epoch} TRAIN/VAL - loss {self._last_train_loss}/{phase.loss_probe.average()}, score {self._last_train_score}/{phase.score} |||"
            )
        if phase._phase_name=='test':
            if phase.score >=self._best_test_score or self._diagnose_mode:
                self._best_test_score=phase.score
                if self._save_checkpoint_enabled:
                    self._trigger_best_score = True
                    self._trigger_state_save = True


    @staticmethod
    def auto_bind_and_run(func):
        logger.remove()
        logger.add(sys.stderr,format=LOGGER_FORMAT)
        temp = logger.add(f"log.log",format=LOGGER_FORMAT)

        sig = inspect.signature(func)
        params = sig.parameters
        parser = argparse.ArgumentParser(description="TrainHelper")
        for param_name, param in params.items():
            if param.default != inspect.Parameter.empty:
                parser.add_argument(
                    f"--{param_name}", default=param.default, type=param.annotation
                )
            else:
                parser.add_argument(
                    f"--{param_name}", type=param.annotation, required=True
                )

        args = vars(parser.parse_args())

        active_results = {}
        name,val,changed=[],[],[]
        for param_name, param in params.items():
            env_input = args[param_name]
            if env_input != param.default:
                active_results[param_name] = env_input
                name.append(f"*{param_name}")
                val.append(env_input)
            else:
                assert (
                        param.default != inspect.Parameter.empty
                ), f"you did not set parameter `{param_name}`"
                active_results[param_name] = param.default
                name.append(f"{param_name}")
                val.append(param.default)
        
        table=show_params_in_3cols(name=name,val=val)
        logger.warning(f"[HYPERPARAMETERS]\n{table}")
        
        logger.remove(temp)
        func(**active_results)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
