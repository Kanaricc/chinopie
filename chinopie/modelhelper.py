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
import optuna
from optuna.distributions import CategoricalChoiceType
import numpy as np
from prettytable import PrettyTable, PLAIN_COLUMNS
from loguru import logger

from .probes.avgmeter import AverageMeter, NumericMeter
from .datasets.fakeset import FakeEmptySet
from .ddpsession import DdpSession
from .filehelper import FileHelper
from .phasehelper import (
    CheckpointLoadSection,
    CheckpointSaveSection,
    PhaseHelper,
)
from .utils import show_params_in_3cols,create_snapshot,check_gitignore

LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class TrainHelper:
    def __init__(
        self,
        trial: optuna.Trial,
        disk_root: str,
        epoch_num: int,
        load_checkpoint: bool,
        save_checkpoint: bool,
        checkpoint_save_period: Optional[int],
        comment: str,
        dev: str = "",
        enable_ddp=False,
        enable_diagnose=False,
    ) -> None:
        self.trial = trial
        self._epoch_num = epoch_num
        self._load_checkpoint_enabled = load_checkpoint
        self._save_checkpoint_enabled = save_checkpoint
        if checkpoint_save_period is not None:
            self._checkpoint_save_period = checkpoint_save_period
        else:
            self._checkpoint_save_period = 1
        self._comment = comment
        self._ddp_session = DdpSession() if enable_ddp else None
        self._diagnose_mode=enable_diagnose
        self._argparser=argparse.ArgumentParser()
        self._test_phase_enabled = False


        # init diagnose flags
        self._has_checkpoint_load_section = False
        self._has_train_phase = False
        self._has_val_phase = False
        self._has_test_phase = False
        self._has_checkpoint_save_section = False
        self._has_flushed_params=False

        if self._ddp_session:
            self._ddp_session.barrier()

        self.file = FileHelper(disk_root, comment, self._ddp_session)
        self._board_dir = self.file.get_default_board_dir()

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
        self._custom_category_params: Dict[str, Optional[CategoricalChoiceType]] = {}
        self._custom_int_params: Dict[str, Optional[int]] = {}
        self._custom_float_params: Dict[str, Optional[float]] = {}
        self._fastforward_handlers: List[Callable[[int], None]] = []


    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()

    def section_checkpoint_load(self):
        self._has_checkpoint_load_section = True
        latest_ckpt_path=self.file.find_latest_checkpoint()
        if latest_ckpt_path is not None:
            logger.debug(f"found latest checkpoint at `{latest_ckpt_path}`")
        else:
            logger.debug(f"no checkpoint found")

        break_load_section=not self._load_checkpoint_enabled or latest_ckpt_path is None
        return CheckpointLoadSection(latest_ckpt_path, break_load_section,lambda x:self._load_from_checkpoint(x))

    def section_checkpoint_save(self):
        self._has_checkpoint_save_section = True

        flag_save = (
            self._is_main_process()
            and self._save_checkpoint_enabled
            and self._trigger_checkpoint
        )
        flag_best = (
            self._is_main_process()
            and self._save_checkpoint_enabled
            and self._trigger_best_score
        )

        # force saving ckpt in diagnose mode
        flag_best=True
        flag_save=True
        return CheckpointSaveSection(
            self._export_state(),
            self.file.get_checkpoint_slot(self.cur_epoch),
            self.file.get_best_checkpoint_slot(),
            flag_save,
            flag_best,
            not (flag_save or flag_best),
            lambda x: self._merge_section_flags(x),
        )

    def _load_from_checkpoint(self, flags: Dict[str, Any]):
        """
        load checkpoint file, including model state, epoch index.
        """
        if 'no_checkpoint' in flags:
            logger.debug("no checkpoint found, skipping helper state recovery")
            return
        assert 'checked_helper_state' in flags, "helper state not loaded but checkpoint is found"
        checkpoint=flags['checked_helper_state']

        self._board_dir = checkpoint["board_dir"]
        self._recoverd_epoch = checkpoint["cur_epoch"]
        self._best_val_score = checkpoint["best_val_score"]
        if 'best_test_score' in checkpoint:
            self._best_test_score = checkpoint["best_test_score"]
        logger.info(f"[HELPER] load state from checkpoint")

    def _export_state(self):
        # FIXME: no test score
        return {
            "board_dir": self._board_dir,
            "cur_epoch": self.cur_epoch,
            "best_val_score": self._best_val_score,
        }

    def register_probe(self, name: str):
        self._custom_probes.append(name)

    def reg_category(self, name: str, value: Optional[CategoricalChoiceType] = None):
        self._argparser.add_argument(f"--{name}",required=False)
        self._custom_category_params[name] = value

    def reg_int(self, name: str, value: Optional[int] = None):
        self._argparser.add_argument(f"--{name}",type=int,required=False)
        self._custom_int_params[name] = value

    def reg_float(self, name: str, value: Optional[float] = None):
        self._argparser.add_argument(f"--{name}",type=float,required=False)
        self._custom_float_params[name] = value

    def flush_params(self):
        self._has_flushed_params=True
        args,_=self._argparser.parse_known_args()
        logger.debug(f"hyperparameter in argparser: {args}")
        for k in self._custom_category_params.keys():
            if getattr(args,k) is not None:
                self._custom_category_params[k]=getattr(args,k)
                logger.debug(f"flushed `{k}`")
        for k in self._custom_float_params.keys():
            if getattr(args,k) is not None:
                self._custom_float_params[k]=getattr(args,k)
                logger.debug(f"flushed `{k}`")
        for k in self._custom_int_params.keys():
            if getattr(args,k) is not None:
                self._custom_int_params[k]=getattr(args,k)
                logger.debug(f"flushed `{k}`")


    def register_fastforward_handler(self, func: Callable[[int], None]):
        self._fastforward_handlers.append(func)

    def register_dataset(
        self, train: Any, trainloader: DataLoader, val: Any, valloader: DataLoader
    ):
        self._data_train = train
        self._dataloader_train = trainloader
        self._data_val = val
        self._dataloader_val = valloader

        if self._ddp_session:
            assert isinstance(self._dataloader_train.sampler, DistributedSampler)
            assert not isinstance(self._dataloader_val.sampler, DistributedSampler)

    def register_test_dataset(self, test: Any, testloader: DataLoader):
        self._data_test = test
        self._dataloader_test = testloader
        self._test_phase_enabled = True

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

    def suggest_category(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        assert name in self._custom_category_params, f"request for unregisted param `{name}`"
        fixed_val = self._custom_category_params[name]
        if fixed_val is not None:
            assert fixed_val in choices
            return fixed_val
        else:
            return self.trial.suggest_categorical(name, choices)

    def suggest_int(self, name: str, low: int, high: int, step=1, log=False) -> int:
        assert name in self._custom_int_params, f"request for unregisted param `{name}`"
        fixed_val = self._custom_int_params[name]
        if fixed_val is not None:
            assert fixed_val >= low and fixed_val <= high
            return fixed_val
        else:
            return self.trial.suggest_int(name, low, high, step, log)

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
        log=False,
    ) -> float:
        assert name in self._custom_float_params, f"request for unregisted param `{name}`"
        fixed_val = self._custom_float_params[name]
        if fixed_val is not None:
            assert fixed_val >= low and fixed_val <= high
            return fixed_val
        else:
            return self.trial.suggest_float(name, low, high, step=step, log=log)


    def ready_to_train(self):
        assert hasattr(self, "_data_train"), "train set not set"
        assert hasattr(self, "_data_val"), "val set not set"
        if not self._test_phase_enabled:
            logger.warning("[DATASET] test set not set. test phase will be skipped.")

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
        self._section_flags = {}
        self.report_info()

        logger.warning("[HELPER] ready to train model")

    def _merge_section_flags(self, x: Dict[str, Any]):
        self._section_flags |= x

    def report_info(self):
        dataset_str = f"train({len(self._data_train)}) val({len(self._data_val)}) test({len(self._data_test) if hasattr(self, '_data_test') else 'not set'})"
        table = show_params_in_3cols(
            params={
                "device": self.dev,
                "diagnose": self._diagnose_mode,
                "epoch num": self._epoch_num,
                "dataset": dataset_str,
                "board dir": self._board_dir,
                "checkpoint load/save": f"{self._load_checkpoint_enabled}/{self._save_checkpoint_enabled}",
                "custom probes": self._custom_probes,
            }
        )
        logger.warning(f"[INFO]\n{table}")
        logger.warning(f"[HYPERPARAMETERS]\n{show_params_in_3cols(self.trial.params)}")

    def _diagnose(self):
        logger.warning("==========Diagnose Results==========")
        if not self._has_flushed_params:
            logger.error("args not flushed")
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
            if "checked_ckpt_slot" not in self._section_flags:
                logger.warning("default ckpt slot not checked. ignore this if you understand what happened.")
            if "checked_best_slot" not in self._section_flags:
                logger.warning("default best slot not checked. ignore this if you understand what happened.")
            if "checked_helper_state" not in self._section_flags:
                logger.error("helper state not checked")
            if "checked_save_ckpt" not in self._section_flags:
                logger.error("checkpoint not checked")
            if "checked_save_best" not in self._section_flags:
                logger.error("best result not checked")

        # remove checkpoints and boards
        if os.path.exists(self.file.ckpt_dir):
            shutil.rmtree(self.file.ckpt_dir)
        if os.path.exists(self.file.board_dir):
            shutil.rmtree(self.file.board_dir)

    def range_epoch(self):
        if self._diagnose_mode:
            logger.info("diagnose mode is enabled. run 2 epochs.")
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
            self._trigger_run_test = False

            if not self._ddp_session:
                logger.warning(f"=== START EPOCH {self.cur_epoch} ===")
            else:
                logger.warning(
                    f"=== RANK {self._ddp_session.get_rank()} START EPOCH {self.cur_epoch} ==="
                )

            if self._ddp_session:
                assert isinstance(
                    self._dataloader_train.sampler, DistributedSampler
                ), "DistributedSampler not set for dataloader"
                self._dataloader_train.sampler.set_epoch(i)

            yield i

            # early stop
            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def phase_train(self):
        self._has_train_phase = True
        assert hasattr(self,'cur_epoch'), "the phase should be called in epoch"
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
        self._has_val_phase = True
        assert hasattr(self,'cur_epoch'), "the phase should be called in epoch"
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
        self._has_test_phase = True
        assert hasattr(self,'cur_epoch'), "the phase should be called in epoch"

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

    def end_phase_callback(self, phase: PhaseHelper):
        assert phase._phase_name in ["train", "val", "test"]
        # collect probes
        if self._ddp_session is None:
            output_dist_probes = phase._output_dist_probes
        else:
            gathered_output_dist_probes: List[List[NumericMeter]] = []
            self._ddp_session.gather_object(
                phase._output_dist_probes, gathered_output_dist_probes, 0
            )
            output_dist_probes = gathered_output_dist_probes[0]
            for i in gathered_output_dist_probes[1:]:
                for dst, src in zip(output_dist_probes, i):
                    dst.update(src.val)
            del gathered_output_dist_probes

        # only log probes in main process
        if self._is_main_process():
            # assume training loss is sync by user
            self.tbwriter.add_scalar(
                f"loss/{phase._phase_name}", phase.loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/train", phase.score, self.cur_epoch)

            for k, v in enumerate(output_dist_probes):
                if v.val.numel() > 0:
                    self.tbwriter.add_histogram(
                        f"outdist/{phase._phase_name}/{k}", v.val, self.cur_epoch
                    )

            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            for k in self._custom_probes:
                if phase.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        f"{k}/{phase._phase_name}",
                        phase.custom_probes[k].average(),
                        self.cur_epoch,
                    )
                    logger.info(
                        f"[{phase._phase_name} probes] {k}: {phase.custom_probes[k].average()}"
                    )
        if phase._phase_name in ["train", "val"]:
            self._last_train_score = phase.score
            self._last_train_loss = phase.loss_probe.average()
            if not self._ddp_session:
                logger.warning(
                    f"|| end {phase._phase_name} {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
                )
            else:
                logger.warning(
                    f"|| RANK {self._ddp_session.get_rank()} end {phase._phase_name} {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score} ||"
                )
        if phase._phase_name == "val":
            if phase.score >= self._best_val_score or self._diagnose_mode:
                self._best_val_score = phase.score
                self._trigger_run_test = True
                # if having no test phase, store best result according to val score
                if self._save_checkpoint_enabled and not self._test_phase_enabled:
                    self._trigger_best_score = True

            # report intermediate score if having no test phase
            if not self._test_phase_enabled:
                self.trial.report(phase.score, self.cur_epoch)
            logger.warning(
                f"=== END EPOCH {self.cur_epoch} TRAIN/VAL - loss {self._last_train_loss}/{phase.loss_probe.average()}, score {self._last_train_score}/{phase.score} ==="
            )

        if phase._phase_name == "test":
            if phase.score >= self._best_test_score or self._diagnose_mode:
                self._best_test_score = phase.score
                if self._save_checkpoint_enabled:
                    self._trigger_best_score = True

            # report intermediate score
            self.trial.report(phase.score, self.cur_epoch)


class TrainBootstrap:
    def __init__(
        self,
        disk_root: str,
        epoch_num: int,
        load_checkpoint: bool,
        save_checkpoint: bool,
        comment: Optional[str],
        checkpoint_save_period: Optional[int] = 1,
        enable_snapshot=False,
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
        args,_=argparser.parse_known_args()
        

        self._disk_root = args.disk_root
        self._epoch_num = args.epoch_num
        if args.comment is not None:
            self._comment = args.comment
        else:
            self._comment = datetime.now().strftime("%Y%m%d%H%M%S")
        self._load_checkpoint = args.load_checkpoint
        self._save_checkpoint = args.save_checkpoint
        self._checkpoint_save_period = checkpoint_save_period
        self._dev = args.dev
        self._diagnose_mode = args.diagnose
        self._enable_ddp = False

        self._init_logger(args.verbose)
        check_gitignore([self._disk_root])
        if self._enable_ddp:
            self._init_ddp()
        if diagnose:
            torch.autograd.anomaly_mode.set_detect_anomaly(True)
            logger.info("diagnose mode enabled")
        if enable_snapshot:
            if not diagnose:
                create_snapshot(self._comment)
                logger.info("created snapshot")
            else:
                logger.info("snapshot is disabled in diagnose mode")
        

    def _init_ddp(self):
        self._ddp_session = DdpSession()
        logger.info("initialized ddp")

    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()

    def _init_logger(self,verbose:bool):
        stdout_level="INFO"
        file_level="DEBUG"
        if verbose:
            stdout_level="DEBUG"
            file_level="TRACE"
        # logger file
        if not os.path.exists("logs"):
            os.mkdir("logs")
        if self._enable_ddp:
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

    def optimize(
        self, func: Callable[[TrainHelper], float | Sequence[float]], n_trials: int
    ):
        if not os.path.exists("opts"):
            os.mkdir("opts")
        self._func = func
        storage_path = os.path.join("opts", f"{self._comment}.db")
        # do not save storage in diagnose mode
        if self._diagnose_mode:
            storage_path=None
        else:
            storage_path=f"sqlite:///{storage_path}"
        study = optuna.create_study(storage=storage_path)
        
        # in diagnose mode, run 3 times only
        if self._diagnose_mode:
            n_trials=3
        study.optimize(self._wrapper, n_trials=n_trials, gc_after_trial=True)

        if self._diagnose_mode:
            self.helper._diagnose()
        else:
            best_params = study.best_params
            best_value = study.best_value
            logger.warning("[BOOPSTRAP] finish optimization")
            logger.warning(
                f"[BOOTSTRAP] best hyperparameters\n{show_params_in_3cols(best_params)}"
            )
            logger.warning(f"[BOOTSTRAP] best score: {best_value}")
        logger.warning("[BOOTSTRAP] good luck!")

    def _wrapper(self, trial: optuna.Trial) -> float | Sequence[float]:
        self.helper = TrainHelper(
            trial,
            disk_root=self._disk_root,
            epoch_num=self._epoch_num,
            load_checkpoint=self._load_checkpoint,
            save_checkpoint=self._save_checkpoint,
            checkpoint_save_period=self._checkpoint_save_period,
            comment=f"{self._comment}_trial{trial._trial_id}",
            dev=self._dev,
            enable_ddp=self._enable_ddp,
            enable_diagnose=self._diagnose_mode
        )
        return self._func(self.helper)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
