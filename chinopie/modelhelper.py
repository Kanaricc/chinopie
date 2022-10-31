import os, sys
import argparse
import random
import inspect
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Self
import torch
from torch import nn
import numpy as np
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist

from datetime import datetime

from loguru import logger

from .probes.avgmeter import AverageMeter, NumericMeter
from .datasets.fakeset import FakeEmptySet

from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

DIR_STATE_DICT = "state"
DIR_CHECKPOINTS = "checkpoints"
DIR_DATASET = "data"
DIR_TENSORBOARD = "boards"


class DdpSession:
    def __init__(self):
        pass

    @staticmethod
    def is_main_process():
        return dist.get_rank() == 0


class FileHelper:
    def __init__(
            self, disk_root: str, comment: str, ddp_session: Optional[DdpSession] = None
    ):
        self.disk_root = disk_root
        self.ddp_session = ddp_session
        self.comment = comment

        if self._is_main_process():
            if not os.path.exists(os.path.join(self.disk_root, DIR_CHECKPOINTS)):
                os.mkdir(os.path.join(self.disk_root, DIR_CHECKPOINTS))
            if not os.path.exists(os.path.join(self.disk_root, DIR_TENSORBOARD)):
                os.mkdir(os.path.join(self.disk_root, DIR_TENSORBOARD))
            if not os.path.exists(os.path.join(self.disk_root, DIR_DATASET)):
                os.mkdir(os.path.join(self.disk_root, DIR_DATASET))
            if not os.path.exists(os.path.join(self.disk_root, DIR_STATE_DICT)):
                os.mkdir(os.path.join(self.disk_root, DIR_STATE_DICT))

        if self.ddp_session:
            dist.barrier()

        self.cp_dir = os.path.join(self.disk_root, DIR_CHECKPOINTS, comment)
        self.board_dir = os.path.join(
            self.disk_root,
            DIR_TENSORBOARD,
            f"{self.comment}-{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}",
        )

    def prepare_checkpoint_dir(self):
        if not os.path.exists(self.cp_dir):
            if self._is_main_process():
                os.mkdir(self.cp_dir)

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        find the latest checkpoint file at checkpoint dir.
        """

        if not os.path.exists(self.cp_dir):
            return None

        checkpoint_files: List[str] = []
        for (dirpath, dirnames, filenames) in os.walk(self.cp_dir):
            checkpoint_files.extend(filenames)

        latest_checkpoint_path = None
        latest_checkpoint_epoch = -1

        for file in checkpoint_files:
            if file.find("best") != -1:
                continue
            if file.find("init") != -1:
                continue
            if file.find("checkpoint") == -1:
                continue
            epoch = int(file.split(".")[0].split("-")[1])
            if epoch > latest_checkpoint_epoch:
                latest_checkpoint_epoch = epoch
                latest_checkpoint_path = file

        if latest_checkpoint_path:
            return os.path.join(self.cp_dir, latest_checkpoint_path)

    def get_initparams_slot(self) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        logger.warning("[HELPER] you have ask for initialization slot")
        filename = f"init.pth"
        return os.path.join(self.cp_dir, filename)

    def get_checkpoint_slot(self, cur_epoch: int) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        filename = f"checkpoint-{cur_epoch}.pth"
        return os.path.join(self.cp_dir, filename)

    def get_dataset_slot(self, dataset_id: str) -> str:
        return os.path.join(self.disk_root, DIR_DATASET, dataset_id)

    def get_best_checkpoint_slot(self) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        return os.path.join(self.cp_dir, "best.pth")

    def get_default_board_dir(self) -> str:
        return self.board_dir

    def _is_main_process(self):
        return self.ddp_session is None or self.ddp_session.is_main_process()


class PhaseHelper:
    class JumpPhaseException(Exception):
        pass

    _phase_name: str
    _loss_probe: AverageMeter
    _output_dist_probes: List[NumericMeter]
    _custom_probe_name: List[str]
    _custom_probes: Dict[str, AverageMeter]
    _dataset: Any
    _dataloader: DataLoader
    _score: float
    _ddp_session: Optional[DdpSession]
    _dry_run: bool

    _loss_updated: bool
    _score_updated: bool

    _exit_callback: Callable[[Self], None]
    _break_phase: bool

    def __init__(
            self,
            phase_name: str,
            dataset: Any,
            dataloader: DataLoader,
            ddp_session: Optional[DdpSession] = None,
            dry_run: bool = False,
            custom_probes: List[str] = [],
            exit_callback: Callable[[Self], None] = lambda x: None,
            break_phase: bool = False,
    ) -> None:
        self._output_updated = True
        self._phase_name = phase_name
        self._dry_run = dry_run
        self._ddp_session = ddp_session
        self._dataset = dataset
        self._dataloader = dataloader

        self._custom_probe_name = custom_probes
        self._exit_callback = exit_callback
        self._break_phase = break_phase

    def get_data_sample(self):
        for data in self._dataloader:
            return data

    def range_data(self):
        if self._break_phase:
            self._loss_updated = True
            self._score_updated = True
            raise self.JumpPhaseException

        batch_len = len(self._dataloader)
        if self._is_main_process():
            with tqdm(total=batch_len) as progressbar:
                for batchi, data in enumerate(self._dataloader):
                    yield batchi, data
                    progressbar.update()
                    if self._dry_run and batchi>=2:
                        break
        else:
            for batchi, data in enumerate(self._dataloader):
                yield batchi, data
                if self._dry_run and batchi>=2:
                    break

    def __enter__(self):
        self._score = 0.0
        self._loss_probe = AverageMeter("")
        self._output_dist_probes = []
        self._custom_probes = dict(
            [(x, AverageMeter(x)) for x in self._custom_probe_name]
        )

        self._loss_updated = False
        self._score_updated = False
        self._output_updated = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == self.JumpPhaseException:
            return True

        if not self._score_updated:
            logger.error(f"no score updated during phase {self._phase_name}")
        if not self._loss_updated:
            logger.error(f"no loss updated during phase {self._phase_name}")
        if not self._output_updated:
            logger.error(f"no output updated during phase {self._phase_name}")

        for name in self._custom_probe_name:
            logger.error(f"{name} not updated during phase {self._phase_name}")

        self._exit_callback(self)

    def update_probe(self, name: str, value: float, n: int = 1):
        if name in self._custom_probe_name:
            self._custom_probe_name.remove(name)
        self._custom_probes[name].update(value, n)

    @staticmethod
    def validate_loss(loss: Tensor, panic: bool = True) -> bool:
        hasnan = loss.isnan().any().item()
        hasinf = loss.isinf().any().item()
        hasneg = (loss < 0).any().item()
        if panic:
            assert not hasnan, f"loss function returns invalid value `nan`: {loss}"
            assert not hasinf, f"loss function returns invalid value `inf`: {loss}"
            assert not hasneg, f"loss function returns negative value: {loss}"
        return not hasnan and not hasinf and not hasneg

    @staticmethod
    def validate_tensor(t: Tensor, panic: bool = True, msg: str = "") -> bool:
        hasnan = t.isnan().any().item()
        hasinf = t.isinf().any().item()

        if panic:
            assert not hasnan, f"tensor has invalid value `nan`: {t} ({msg})"
            assert not hasinf, f"tensor has invalid value `inf`: {t} ({msg})"

        return not hasnan and not hasinf

    def update_loss(self, loss: Tensor, n: int = 1):
        self._loss_updated = True
        self.validate_loss(loss)
        self._loss_probe.update(loss.item(), n)

    def update_output(self, *outputs):
        self._output_updated=True
        for k, v in enumerate(outputs):
            assert type(v) == Tensor
            self.validate_tensor(v)
            if len(self._output_dist_probes) - 1 < k:
                self._output_dist_probes.append(NumericMeter(f"{k}"))

            self._output_dist_probes[k].update(v.cpu().detach())

    def end_phase(self, score: float):
        self._score_updated = True

        self._score = score

    def _is_main_process(self):
        return self._ddp_session is None or self._ddp_session.is_main_process()

    @property
    def loss_probe(self):
        return self._loss_probe

    @property
    def score(self):
        return self._score

    @property
    def custom_probes(self):
        return self._custom_probes


class TrainHelper:
    dev: str
    _best_val_score: float
    _dry_run: bool

    _custom_global_params: Dict[str, Any]
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
            dist.barrier()

        self.file = FileHelper(disk_root, comment, self._ddp_session)
        self._board_dir = self.file.get_default_board_dir()

        if self._ddp_session:
            world_size = dist.get_world_size()
            assert world_size != -1, "helper must be init after dist"
            if world_size <= torch.cuda.device_count():
                self.dev = f"cuda:{dist.get_rank()}"
            else:
                logger.warning(
                    f"world_size is larger than the number of devices. assume use CPU."
                )
                self.dev = f"cpu:{dist.get_rank()}"

            # logger file
            logger.remove()
            logger.add(sys.stderr, level="WARNING")
            logger.add(f"log-{dist.get_rank()}.log")

            logger.warning(f"[DDP] ddp is enabled. current rank is {dist.get_rank()}.")
            logger.warning(
                f"[DDP] use `{self.dev}` for this process. but you may use other one you want."
            )

            if self._is_main_process():
                logger.warning(
                    f"[DDP] rank {dist.get_rank()} as main process, methods are enabled for it."
                )
            else:
                logger.warning(
                    f"[DDP] rank {dist.get_rank()} as follower process, methods are disabled for it."
                )

        else:
            # logger file
            logger.remove()
            logger.add(sys.stderr, level="WARNING")
            logger.add(f"log.log")

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
        logger.warning(f"[CHECKPOINT] load state from checkpoint")

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
        assert (
                self._dataloader_val.batch_size == self._batch_size
        ), f"batch size of dataloader_val does not match"

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
            logger.warning("[HELPER] fixed seed is set for random and torch")
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)
        else:
            logger.warning(
                f"[DDP] fixed seed `{seed + dist.get_rank()}` is set for this process"
            )
            os.environ["PYTHONHASHSEED"] = str(seed + dist.get_rank())
            random.seed(seed + dist.get_rank())

            torch.manual_seed(seed + dist.get_rank())
            torch.cuda.manual_seed(seed + dist.get_rank())
            torch.cuda.manual_seed_all(seed + dist.get_rank())

            np.random.seed(seed + dist.get_rank())

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
            dist.barrier()

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
        logger.warning(f"[PARAMS] device: {self.dev}")
        logger.warning(f"[PARAMS] epoch num: {self._epoch_num}")
        logger.warning(
            f"[PARAMS] dataset: train({len(self._data_train)}) val({len(self._data_val)}) test({len(self._data_test) if hasattr(self, '_data_test') else 'not set'})"
        )
        logger.warning(f"[PARAMS] board dir: {self._board_dir}")
        logger.warning(f"[PARAMS] checkpoint load: {self._load_checkpoint}")
        logger.warning(f"[PARAMS] checkpoint save: {self._enable_checkpoint}")
        logger.warning(f"[PARAMS] custom probes: {self._custom_probes}")

    def range_epoch(self):
        if self._dry_run:
            self._epoch_num = 2
        for i in range(self._epoch_num):
            if hasattr(self, "_recoverd_epoch") and i < self._recoverd_epoch:
                logger.warning(f"[HELPER] fast forward to epoch {self._recoverd_epoch}")
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
                    f"=== RANK {dist.get_rank()} START EPOCH {self.cur_epoch} ==="
                )

            if self._ddp_session:
                assert isinstance(self._dataloader_train.sampler, DistributedSampler)
                self._dataloader_train.sampler.set_epoch(i)

            yield i

            if self.if_need_save_checkpoint():
                logger.error("[CHECKPOINT] checkpoint not saved")
            if self.if_need_save_best_checkpoint():
                logger.error(
                    "[CHECKPOINT] best score checkpoint is not saved in previous epoch"
                )
            if self._trigger_state_save:
                logger.error(
                    "[CHECKPOINT] state of helper is not saved in previous epoch"
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
            dist.gather_object(phase._output_dist_probes,gathered_output_dist_probes,0)
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
        
        self._last_test_score=phase.score
        self._last_test_loss=phase.loss_probe.average()
        if not self._ddp_session:
            logger.warning(
                f"|| END_TRAIN {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score}"
            )
        else:
            logger.warning(
                f"|| RANK {dist.get_rank()} END_TRAIN {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score}"
            )

    def end_val(self, phase: PhaseHelper):
        if self._ddp_session is None:
            output_dist_probes=phase._output_dist_probes
        else:
            gathered_output_dist_probes:List[List[NumericMeter]]=[]
            dist.gather_object(phase._output_dist_probes,gathered_output_dist_probes,0)
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
                f"|| END_VAL {self.cur_epoch} - loss {phase.loss_probe.average()}, score {phase.score}"
            )
            logger.warning(
                f"||| END EPOCH {self.cur_epoch} TRAIN/VAL - loss {self._last_test_loss}/{phase.loss_probe.average()}, score {self._last_test_score}/{phase.score} |||"
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
            dist.gather_object(phase._output_dist_probes,gathered_output_dist_probes,0)
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
                f"|| END_TEST {self.cur_epoch} - {phase.loss_probe.average()}, score {phase.score}"
            )

    @staticmethod
    def auto_bind_and_run(func):
        temp = logger.add(f"log.log")

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
        for param_name, param in params.items():
            env_input = args[param_name]
            if env_input != param.default:
                active_results[param_name] = env_input
                logger.warning(f"[CPARAMS] {param_name}: {env_input} (changed)")
            else:
                assert (
                        param.default != inspect.Parameter.empty
                ), f"you did not set parameter `{param_name}`"
                active_results[param_name] = param.default
                logger.warning(f"[CPARAMS] {param_name}: {param.default}")
        logger.remove(temp)
        func(**active_results)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
