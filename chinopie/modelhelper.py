import os, sys
import argparse
import random
import inspect
from typing import Any, Dict, List, Optional
import torch
from torch import nn
import numpy as np
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.distributed as dist

from datetime import datetime

from loguru import logger

from .probes.avgmeter import AverageMeter

from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

DIR_STATE_DICT = "state"
DIR_CHECKPOINTS = "checkpoints"
DIR_DATASET = "data"
DIR_TENSORBOARD = "board"

class FileHelper:
    def __init__(self,disk_root:str,comment:str,enable_ddp:bool=False) -> None:
        self.disk_root=disk_root
        self.enable_ddp=enable_ddp
        self.comment=comment

        if self._is_main_process():
            if not os.path.exists(os.path.join(self.disk_root, DIR_CHECKPOINTS)):
                os.mkdir(os.path.join(self.disk_root, DIR_CHECKPOINTS))
            if not os.path.exists(os.path.join(self.disk_root, DIR_TENSORBOARD)):
                os.mkdir(os.path.join(self.disk_root, DIR_TENSORBOARD))
            if not os.path.exists(os.path.join(self.disk_root, DIR_DATASET)):
                os.mkdir(os.path.join(self.disk_root, DIR_DATASET))
            if not os.path.exists(os.path.join(self.disk_root, DIR_STATE_DICT)):
                os.mkdir(os.path.join(self.disk_root, DIR_STATE_DICT))
        
        if self.enable_ddp:
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
        self._trigger_checkpoint = False
        return os.path.join(self.cp_dir, filename)

    def get_dataset_slot(self, dataset_id: str) -> str:
        return os.path.join(self.disk_root, DIR_DATASET, dataset_id)

    def get_best_checkpoint_slot(self) -> str:
        if not self._is_main_process():
            logger.warning("[DDP] try to get checkpoint slot on follower")
        self._trigger_best_score = False
        return os.path.join(self.cp_dir, "best.pth")
    
    def get_default_board_dir(self)->str:
        return self.board_dir
    
    def _is_main_process(self):
        return self.enable_ddp == False or dist.get_rank() == 0


class TrainHelper:
    dev: str
    train_loss_probe: AverageMeter
    validate_loss_probe: AverageMeter
    test_loss_probe: AverageMeter
    custom_probes: Dict[str, AverageMeter]
    best_val_score: float
    is_dry_run: bool

    custom_global_params: Dict[str, Any]
    data_train: Any
    data_val: Any
    data_test: Any
    dataloader_train: DataLoader
    dataloader_val: DataLoader
    dataloader_test: DataLoader
    batch_size: int

    _trigger_epoch_start: bool
    _trigger_trainval_start: bool
    _trigger_best_score: bool
    _trigger_state_save: bool
    _trigger_checkpoint: bool
    _trigger_loss_check: bool
    _trigger_run_test: bool
    _trigger_test_start: bool

    def __init__(
        self,
        disk_root: str,
        epoch_num: int,
        batch_size: int,
        auto_load_checkpoint: bool,
        enable_checkpoint: bool,
        checkpoint_save_period: Optional[int],
        comment: str,
        details:Optional[str]=None,
        dev: str = "",
        enable_ddp=False,
        dry_run: bool = False,
    ) -> None:
        logger.warning("[HELPER] details for this run")
        logger.warning(details)
        
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.auto_load_checkpoint = auto_load_checkpoint
        self.enable_checkpoint = enable_checkpoint
        if checkpoint_save_period != None:
            self.checkpoint_save_period = checkpoint_save_period
        else:
            self.checkpoint_save_period = 1
        self.comment = comment
        self.enable_ddp = enable_ddp

        
        if self.enable_ddp:
            dist.barrier()
        
        self.file=FileHelper(disk_root,comment,enable_ddp)
        self.board_dir=self.file.get_default_board_dir()

        if self.enable_ddp:
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

        self.custom_probes = {}
        self.custom_global_params = {}
        self.is_dry_run = dry_run
        
        if self.is_dry_run:
            self.enable_checkpoint=False
            self.auto_load_checkpoint=False

    def _is_main_process(self):
        return self.enable_ddp == False or dist.get_rank() == 0

    def if_need_load_checkpoint(self):
        return self.auto_load_checkpoint

    def if_need_save_checkpoint(self):
        t = self._trigger_checkpoint
        self._trigger_checkpoint = False
        if not self._is_main_process():
            return False
        return self.enable_checkpoint and t

    def if_need_save_best_checkpoint(self):
        t = self._trigger_best_score
        self._trigger_best_score = False
        if not self._is_main_process():
            return False
        return self.enable_checkpoint and t

    def if_need_run_test_phase(self):
        t = self._trigger_run_test
        self._trigger_run_test = False
        return hasattr(self, "dataloader_test") and t
    

    def load_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        load checkpoint file, including model state, epoch index.
        """
        self.board_dir = checkpoint["board_dir"]
        self._recoverd_epoch = checkpoint["cur_epoch"]
        self.best_val_score = checkpoint["best_val_score"]
        logger.warning(f"[CHECKPOINT] load state from checkpoint")

    def export_state(self):
        self._trigger_state_save = False
        return {
            "board_dir": self.board_dir,
            "cur_epoch": self.cur_epoch,
            "best_val_score": self.best_val_score,
        }

    

    def register_probe(self, name: str):
        self.custom_probes[name] = AverageMeter(name)

    def register_global_params(self, name: str, value: Any):
        self.custom_global_params[name] = value

    def register_dataset(
        self, train: Any, trainloader: DataLoader, val: Any, valloader: DataLoader
    ):
        self.data_train = train
        self.dataloader_train = trainloader
        self.data_val = val
        self.dataloader_val = valloader

        assert (
            self.dataloader_train.batch_size == self.batch_size
        ), f"batch size of dataloader_train does not match"
        assert (
            self.dataloader_val.batch_size == self.batch_size
        ), f"batch size of dataloader_val does not match"

        if self.enable_ddp:
            assert isinstance(self.dataloader_train.sampler, DistributedSampler)
            assert not isinstance(self.dataloader_val.sampler, DistributedSampler)

    def register_test_dataset(self, test: Any, testloader: DataLoader):
        self.data_test = test
        self.dataloader_test = testloader

        assert (
            self.dataloader_test.batch_size == self.batch_size
        ), f"batch size of dataloader_test does not match"

        if self.enable_ddp:
            assert not isinstance(self.dataloader_test.sampler, DistributedSampler)

    def set_fixed_seed(self, seed: Any, disable_ddp_seed=False):
        if not self.enable_ddp or disable_ddp_seed:
            logger.warning("[HELPER] fixed seed is set for random and torch")
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)
        else:
            logger.warning(
                f"[DDP] fixed seed `{seed+dist.get_rank()}` is set for this process"
            )
            os.environ["PYTHONHASHSEED"] = str(seed + dist.get_rank())
            random.seed(seed + dist.get_rank())

            torch.manual_seed(seed + dist.get_rank())
            torch.cuda.manual_seed(seed + dist.get_rank())
            torch.cuda.manual_seed_all(seed + dist.get_rank())

            np.random.seed(seed + dist.get_rank())

    def get(self, name: str) -> Any:
        assert name in self.custom_global_params
        return self.custom_global_params[name]

    def set_dry_run(self):
        self.is_dry_run = True

    def ready_to_train(self):
        if (
            not hasattr(self, "data_train")
            or not hasattr(self, "data_val")
            or not hasattr(self, "dataloader_train")
            or not hasattr(self, "dataloader_val")
        ):
            logger.error("[DATASET] dataset not set")
            raise Exception("dataset not set")
        if not hasattr(self, "dataloader_test"):
            logger.warning(
                "[DATASET] test dataset not set. test phase will be ignored."
            )
        
        self.file.prepare_checkpoint_dir()
        # make sure all processes waiting till the dir is done
        if self.enable_ddp:
            dist.barrier()

        # create board dir before training
        self.tbwriter = SummaryWriter(self.board_dir)
        self.train_loss_probe = AverageMeter("train_loss")
        self.validate_loss_probe = AverageMeter("val_loss")
        self.test_loss_probe = AverageMeter("test_loss")

        if not hasattr(self, "best_val_score"):
            self.best_val_score = 0.0
        self._trigger_epoch_start = False
        self._trigger_trainval_start = False
        self._trigger_best_score = False
        self._trigger_checkpoint = False
        self._trigger_loss_check = False
        self._trigger_state_save = False
        self._trigger_run_test = False
        self._trigger_test_start = False

        self.report_info()
        logger.warning("[TRAIN] ready to train model")

    def report_info(self):
        logger.warning(
            f"""[SUMMARY]
\t\tdevice: {self.dev}
\t\tepoch num: {self.epoch_num}
\t\tdataset: train({len(self.data_train)}) val({len(self.data_val)}) test({len(self.data_test) if hasattr(self,'data_test') else 'not set'})
\t\tboard dir: {self.board_dir}
\t\tcheckpoint load: {self.auto_load_checkpoint}
\t\tcheckpoint save: {self.enable_checkpoint}
\t\tcustom probes: {self.custom_probes.keys()}
        """
        )

    def range_epoch(self):
        if self.is_dry_run:
            self.epoch_num = 2
        for i in range(self.epoch_num):
            if hasattr(self, "_recoverd_epoch") and i < self._recoverd_epoch:
                logger.warning(f"[HELPER] fast forward to epoch {self._recoverd_epoch}")
                continue
            self.cur_epoch = i

            # begin of epoch
            if self._trigger_epoch_start:
                logger.error(
                    "[HELPER] epoch ended unexpectedly without `end_epoch()` being called"
                )
                raise Exception(
                    "epoch ended unexpectedly without `end_epoch()` being called"
                )
                # self._trigger_epoch_start = False # this is a critical exception
            if self._trigger_test_start:
                logger.error(
                    "[HELPER] epoch ended unexpectedly without test phase result"
                )
                self._trigger_test_start = False
            if self.if_need_validate_loss():
                logger.error("[LOSS] no loss checked in previous epoch")
            if self.if_need_save_checkpoint():
                logger.error("[CHECKPOINT] checkpoint is not saved in previous epoch")
            if self.if_need_save_best_checkpoint():
                logger.error(
                    "[CHECKPOINT] best score checkpoint is not saved in previous epoch"
                )
            if self._trigger_state_save:
                logger.error(
                    "[CHECKPOINT] state of helper is not saved in previous epoch"
                )
                self._trigger_state_save = False
            if self.if_need_run_test_phase():
                logger.error("[HELPER] test phase should be run in previous epoch")

            self.cur_epoch = self.cur_epoch
            self.reset_all_probe()

            self._trigger_checkpoint = (
                self.enable_checkpoint
                and self.cur_epoch % self.checkpoint_save_period == 0
            )
            self._trigger_loss_check = True

            if not self.enable_ddp:
                logger.warning(f"=== START EPOCH {self.cur_epoch} ===")
            else:
                logger.warning(f"=== RANK {dist.get_rank()} START EPOCH {self.cur_epoch} ===")


            self._trigger_epoch_start = True
            self._trigger_trainval_start = True

            if self.enable_ddp:
                assert isinstance(self.dataloader_train.sampler, DistributedSampler)
                self.dataloader_train.sampler.set_epoch(i)

            yield i
    
    def _range_common(self,dataloader):
        batch_len = len(dataloader)
        if self._is_main_process():
            with tqdm(total=batch_len) as progressbar:
                for batchi, data in enumerate(dataloader):
                    if self.cur_epoch == 0 and batchi == 0:
                        logger.info(f"data sample: {data}")
                    yield batchi, data
                    progressbar.update()
                    if self.is_dry_run:
                        break
        else:
            for batchi, data in enumerate(dataloader):
                if self.cur_epoch == 0 and batchi == 0:
                    logger.info(f"data sample: {data}")
                yield batchi, data
                if self.is_dry_run:
                    break

    def range_train(self):
        logger.info("start train phase")
        return self._range_common(self.dataloader_train)

    def range_val(self):
        logger.info("start val phase")
        return self._range_common(self.dataloader_val)

    def range_test(self):
        assert self.dataloader_test != None
        logger.info("start test phase")
        self._trigger_test_start = True
        return self._range_common(self.dataloader_test)

    def reset_loss_probe(self):
        self.train_loss_probe.reset()
        self.validate_loss_probe.reset()
        self.test_loss_probe.reset()

    def reset_custom_probe(self):
        for k in self.custom_probes:
            self.custom_probes[k].reset()

    def reset_all_probe(self):
        self.reset_loss_probe()
        self.reset_custom_probe()

    def update_loss_probe(self, phase: str, loss: Tensor, batch_size: int):
        if phase == "train":
            if (
                self.enable_ddp
                and batch_size != dist.get_world_size() * self.batch_size
            ):
                logger.warning(
                    f"[DDP] the batch_size seems not proper during training loss update: {batch_size}!={dist.get_world_size()*self.batch_size}. ignore this if you understand what you have done."
                )
            self.train_loss_probe.update(loss.item(), batch_size)
        elif phase == "val":
            self.validate_loss_probe.update(loss.item(), batch_size)
        elif phase == "test":
            self.test_loss_probe.update(loss.item(), batch_size)
        else:
            assert False, "phase should be either `train`, `val`, or 'test'"

    def update_probe(self, name: str, value: Tensor, times: int = 1):
        assert name in self.custom_probes
        # we cannot check if the times is proper for DDP

        self.custom_probes[name].update(value.item(), times)

    def end_trainval(self, train_score: float, val_score: float):
        if self._is_main_process():
            # assume training loss is sync by user
            self.tbwriter.add_scalar(
                "loss/train", self.train_loss_probe.average(), self.cur_epoch
            )
            # validation phase is full and run duplicated on every processes, including main process
            self.tbwriter.add_scalar(
                "loss/val", self.validate_loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/train", train_score, self.cur_epoch)
            self.tbwriter.add_scalar("score/val", val_score, self.cur_epoch)

        if not self.enable_ddp:
            logger.warning(
                f"=== END EPOCH {self.cur_epoch} - {self.train_loss_probe}, {self.validate_loss_probe}, train/val score {train_score}/{val_score} ==="
            )
        else:
            logger.warning(
                f"=== RANK {dist.get_rank()} END EPOCH {self.cur_epoch} - {self.train_loss_probe}, {self.validate_loss_probe}, train/val score {train_score}/{val_score} ==="
            )
        if val_score >= self.best_val_score:
            self.best_val_score = val_score
            self._trigger_run_test = True
            if self.enable_checkpoint:
                self._trigger_best_score = True
                self._trigger_state_save = True

        self._trigger_trainval_start = False

    def end_epoch(self):
        if self._trigger_trainval_start == True:
            logger.error(
                f"[HELPER] trainval phase not done. call `end_trainval()` before end a epoch."
            )
            assert (
                self._trigger_trainval_start != True
            ), "trainval phase not done. call `end_trainval()` before end a epoch."
        if self._is_main_process():
            # sync of custom probes is done by users
            # TODO: but this can be done by us if necessary
            custom_probes_reports = ""
            for k in self.custom_probes:
                if self.custom_probes[k].has_data():
                    self.tbwriter.add_scalar(
                        k, self.custom_probes[k].average(), self.cur_epoch
                    )
                    custom_probes_reports += (
                        f"\t\t{k}: {self.custom_probes[k].average()}\n"
                    )
            logger.info("[CPROBES] custom probes reports:\n" + custom_probes_reports)

        self._trigger_epoch_start = False

    def update_test_score(self, score: float):
        self._trigger_test_start = False
        if self._is_main_process():
            self.tbwriter.add_scalar(
                "loss/test", self.test_loss_probe.average(), self.cur_epoch
            )
            self.tbwriter.add_scalar("score/test", score, self.cur_epoch)

        logger.warning(
            f">>> END EPOCH {self.cur_epoch} TEST: test score: {score}, {self.test_loss_probe} <<<"
        )

    def validate_loss(self, loss: Tensor, panic: bool = True) -> bool:
        hasnan = loss.isnan().any().item()
        hasinf = loss.isinf().any().item()
        hasneg = (loss < 0).any().item()
        if panic:
            assert not hasnan, f"loss function returns invalid value `nan`: {loss}"
            assert not hasinf, f"loss function returns invalid value `inf`: {loss}"
            assert not hasneg, f"loss function returns negative value: {loss}"

        self._trigger_loss_check = False

        return not hasnan and not hasinf and not hasneg

    def validate_tensor(self, t: Tensor, panic: bool = True, msg: str = "") -> bool:
        hasnan = t.isnan().any().item()
        hasinf = t.isinf().any().item()

        if panic:
            assert not hasnan, f"tensor has invalid value `nan`: {t} ({msg})"
            assert not hasinf, f"tensor has invalid value `inf`: {t} ({msg})"

        return not hasnan and not hasinf

    def if_need_validate_loss(self) -> bool:
        t = self._trigger_loss_check
        self._trigger_loss_check = False
        return t

    @staticmethod
    def auto_bind_and_run(func):
        temp=logger.add(f"log.log")

        sig = inspect.signature(func)
        params = sig.parameters
        parser=argparse.ArgumentParser(description="TrainHelper")
        for param_name, param in params.items():
            if param.default!=inspect.Parameter.empty:
                parser.add_argument(f"--{param_name}",default=param.default,type=param.annotation)
            else:
                parser.add_argument(f"--{param_name}",type=param.annotation,required=True)
        
        args=vars(parser.parse_args())

        active_results = {}
        reports = ""
        for param_name, param in params.items():
            env_input = args[param_name]
            if env_input != param.default:
                active_results[param_name] = env_input
                reports += (
                    f"\t\t{param_name}: {env_input} (changed)\n"
                )
            else:
                assert (
                    param.default != inspect.Parameter.empty
                ), f"you did not set parameter `{param_name}`"
                active_results[param_name] = param.default
                reports += f"\t\t{param_name}: {param.default}\n"

        logger.warning("[SUMMARY] custom training parameters:\n" + reports)

        logger.remove(temp)

        func(**active_results)


def gettype(name):
    t = getattr(__builtins__, name)
    if isinstance(t, type):
        return t
    raise ValueError(name)
