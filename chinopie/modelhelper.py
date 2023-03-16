import os, sys
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
from ddpsession import DdpSession
from filehelper import FileHelper
from phasehelper import PhaseHelper

LOGGER_FORMAT='<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'

class TrainHelper:
    dev: str
    _best_val_score: float
    _dry_run: bool

    _custom_global_params: Dict[str, Any]
    _fastforward_handlers:List[Callable[[int],None]]
    _data_train: Any
    _data_val: Any
    _data_test: Any
    _dataloader_train: DataLoader
    _dataloader_val: DataLoader
    _dataloader_test: DataLoader
    _batch_size: int

    _trigger_best_score: bool
    _trigger_state_save: bool
    _trigger_checkpoint: bool
    _trigger_run_test: bool

    def __init__(
            self,
            disk_root: str,
            epoch_num: int,
            batch_size: int,
            load_checkpoint: bool,
            enable_checkpoint: bool,
            checkpoint_save_period: Optional[int],
            comment: str,
            details: Optional[str] = None,
            dev: str = "",
            enable_ddp=False,
            dry_run: bool = False,
    ) -> None:
        logger.warning("[HELPER] details for this run")
        logger.warning(details)

        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._load_checkpoint = load_checkpoint
        self._enable_checkpoint = enable_checkpoint
        if checkpoint_save_period is not None:
            self._checkpoint_save_period = checkpoint_save_period
        else:
            self._checkpoint_save_period = 1
        self._comment = comment
        self._ddp_session = DdpSession() if enable_ddp else None

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
        self._custom_global_params = {}
        self._fastforward_handlers = []
        self._dry_run = dry_run

        if self._dry_run:
            self._enable_checkpoint = False
            self._load_checkpoint = False
            torch.autograd.anomaly_mode.set_detect_anomaly(True)

    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()

    def if_need_load_checkpoint(self):
        return self._load_checkpoint

    def if_need_save_checkpoint(self):
        t = self._trigger_checkpoint
        self._trigger_checkpoint = False
        if not self._is_main_process():
            return False
        return self._enable_checkpoint and t

    def if_need_save_best_checkpoint(self):
        t = self._trigger_best_score
        self._trigger_best_score = False
        if not self._is_main_process():
            return False
        return self._enable_checkpoint and t

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        load checkpoint file, including model state, epoch index.
        """
        self._board_dir = checkpoint["board_dir"]
        self._recoverd_epoch = checkpoint["cur_epoch"]
        self._best_val_score = checkpoint["best_val_score"]
        logger.info(f"[CHECKPOINT] load state from checkpoint")

    def export_state(self):
        self._trigger_state_save = False
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
            logger.warning("[HELPER] batch size of dataloader_val does not match")

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
                f"[DDP] fixed seed `{seed + self._ddp_session.get_rank()}` is set for this process"
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
        self._dry_run = True

    def ready_to_train(self):
        if (
                not hasattr(self, "_data_train")
                or not hasattr(self, "_data_val")
                or not hasattr(self, "_dataloader_train")
                or not hasattr(self, "_dataloader_val")
        ):
            logger.error("[DATASET] dataset not set")
            raise RuntimeError("dataset not set")
        if not hasattr(self, "_dataloader_test"):
            logger.warning("[DATASET] test dataset not set. pass test phase.")

        self.file.prepare_checkpoint_dir()
        # make sure all processes waiting till the dir is done
        if self._ddp_session:
            self._ddp_session.barrier()

        # create board dir before training
        self.tbwriter = SummaryWriter(self._board_dir)

        if not hasattr(self, "_best_val_score"):
            self._best_val_score = 0.0
        self._trigger_best_score = False
        self._trigger_checkpoint = False
        self._trigger_state_save = False
        self._trigger_run_test = False

        self.report_info()
        logger.warning("[TRAIN] ready to train model")

    def report_info(self):
        logger.warning(f"""[PARAMS]
device: {self.dev}
epoch num: {self._epoch_num}
dataset: train({len(self._data_train)}) val({len(self._data_val)}) test({len(self._data_test) if hasattr(self, '_data_test') else 'not set'})
board dir: {self._board_dir}
checkpoint load/save: {self._load_checkpoint}/{self._enable_checkpoint}
custom probes: {self._custom_probes}
        """)


    def range_epoch(self):
        if self._dry_run:
            logger.info("you have enable dry-run mode")
            self._epoch_num = 2
        for i in range(self._epoch_num):
            if hasattr(self, "_recoverd_epoch") and i <= self._recoverd_epoch:
                for item in self._fastforward_handlers:
                    item(i)
                logger.info(f"[HELPER] fast pass epoch {self._recoverd_epoch}")
                continue
            self.cur_epoch = i

            # begin of epoch
            self._trigger_checkpoint = (
                    self._enable_checkpoint
                    and self.cur_epoch % self._checkpoint_save_period == 0
            )

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

            if self.if_need_save_checkpoint():
                logger.error("[CHECKPOINT] checkpoint not saved")
            if self.if_need_save_best_checkpoint():
                logger.error(
                    "[CHECKPOINT] best score checkpoint not saved"
                )
            if self._trigger_state_save:
                logger.error(
                    "[CHECKPOINT] state of helper not saved"
                )
                self._trigger_state_save = False

    def phase_train(self):
        return PhaseHelper(
            "train",
            self._data_train,
            self._dataloader_train,
            ddp_session=None,
            dry_run=self._dry_run,
            exit_callback=self.end_train,
            custom_probes=self._custom_probes.copy(),
        )

    def phase_val(self):
        return PhaseHelper(
            "val",
            self._data_val,
            self._dataloader_val,
            ddp_session=None,
            dry_run=self._dry_run,
            exit_callback=self.end_val,
            custom_probes=self._custom_probes.copy(),
        )

    def phase_test(self):
        need_run = hasattr(self, "_dataloader_test") and self._trigger_run_test
        self._trigger_run_test = False
        return PhaseHelper(
            "test",
            self._data_test if hasattr(self, "_data_test") else FakeEmptySet(),
            self._dataloader_test
            if hasattr(self, "_dataloader_test")
            else DataLoader(FakeEmptySet()),
            ddp_session=None,
            dry_run=self._dry_run,
            exit_callback=self.end_test,
            break_phase=not need_run,
            custom_probes=self._custom_probes.copy(),
        )

    def end_train(self, phase: PhaseHelper):
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
        
        if self._is_main_process():
            # assume training loss is sync by user
            self.tbwriter.add_scalar(
                "loss/train", phase.loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/train", phase.score, self.cur_epoch)

            for k,v in enumerate(output_dist_probes):
                if v.val.numel()>0:
                    self.tbwriter.add_histogram(f"outdist/train/{k}",v.val,self.cur_epoch)

            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        f"{k}/train", phase.custom_probes[k].average(), self.cur_epoch
                    )
                    logger.info(
                        f"[TRAIN_CPROBES] {k}: {phase.custom_probes[k].average()}"
                    )
        
        self._last_train_score=phase.score
        self._last_train_loss=phase.loss_probe.average()
        if not self._ddp_session:
            logger.warning(
                f"|| END_TRAIN {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )
        else:
            logger.warning(
                f"|| RANK {self._ddp_session.get_rank()} END_TRAIN {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )

    def end_val(self, phase: PhaseHelper):
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

        if self._is_main_process():
            # validation phase is full and run duplicated on every processes, including main process
            self.tbwriter.add_scalar(
                "loss/val", phase.loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/val", phase.score, self.cur_epoch)

            for k,v in enumerate(output_dist_probes):
                if v.val.numel()>0:
                    self.tbwriter.add_histogram(f"outdist/val/{k}",v.val,self.cur_epoch)

            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        f"{k}/val", phase.custom_probes[k].average(), self.cur_epoch
                    )
                    logger.info(
                        f"[VAL_CPROBES] {k}: {phase.custom_probes[k].average()}"
                    )

            logger.warning(
                f"|| END_VAL {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
            )
            logger.warning(
                f"||| END EPOCH {self.cur_epoch} TRAIN/VAL - loss {self._last_train_loss}/{phase.loss_probe.average()}, score {self._last_train_score}/{phase.score} |||"
            )

        if phase.score >= self._best_val_score:
            self._best_val_score = phase.score
            self._trigger_run_test = True
            # TODO: this ?
            if self._enable_checkpoint:
                self._trigger_best_score = True
                self._trigger_state_save = True

    def end_test(self, phase: PhaseHelper):
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

        if self._is_main_process():
            self.tbwriter.add_scalar(
                "loss/test", phase.loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/test", phase.score, self.cur_epoch)

            for k,v in enumerate(output_dist_probes):
                if v.val.numel()>0:
                    self.tbwriter.add_histogram(f"outdist/test/{k}",v.val,self.cur_epoch)

            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        f"{k}/train", phase.custom_probes[k].average(), self.cur_epoch
                    )
                    logger.info(
                        f"[TEST_CPROBES] {k}: {phase.custom_probes[k].average()}"
                    )
            logger.warning(
                f"|| END_TEST {self.cur_epoch} - {phase.loss_probe.average()}, score {phase.score} ||"
            )

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
        
        while len(name)%3!=0:
            name.append('')
            val.append('')
        col_len=len(name)//3
        table=PrettyTable()
        table.set_style(PLAIN_COLUMNS)
        for i in range(3):
            table.add_column("params",name[i*col_len:(i+1)*col_len],"l")
            table.add_column("values",val[i*col_len:(i+1)*col_len],"c")
        logger.warning(f"[CPARAMS]\n{table}")
        
        logger.remove(temp)
        func(**active_results)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
