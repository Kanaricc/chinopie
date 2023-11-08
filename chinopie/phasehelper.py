import sys
import inspect
from typing import List,Dict,Any,Optional,Callable,Sequence
from typing_extensions import Self

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from .probes import AverageMeter,SmoothMeanMeter
from . import iddp as dist
from .utils import any_to,validate_loss,validate_tensor
from . import logging
_logger=logging.get_logger(__name__)

class PhaseEnv:
    def __init__(
            self,
            phase_name: str,
            dataset: Any,
            dataloader: DataLoader,
            dev:Any,
            custom_probes: List[str] = [],
            dry_run: bool = False,
            log_sample:bool=False,
            check_loss:bool=True,
            check_score:bool=True,
    ) -> None:
        self._phase_name = phase_name
        self._dry_run = dry_run
        self._do_log_sample=log_sample
        self._do_check_loss=check_loss
        self._do_check_score=check_score
        self._dataset = dataset
        self._dataloader = dataloader
        self._dev=dev
        self._batch_len=len(self._dataloader)

        self._custom_probe_name = custom_probes

        self._score = AverageMeter("",dev)
        self._loss_probe = AverageMeter("",dev)
        self._realtime_loss_probe=SmoothMeanMeter(len(self._dataloader))
        self._custom_probes:Dict[str,AverageMeter] = dict(
            [(x, AverageMeter(x,dev)) for x in self._custom_probe_name]
        )

        self._loss_updated = False
        self._score_updated = False

    def get_data_sample(self):
        for data in self._dataloader:
            return data

    def range_data(self):
        batch_len = len(self._dataloader)
        one_percent_len=max(1,(batch_len+25-1)//25)
        if dist.is_main_process():
            if self._do_log_sample:
                _logger.info("data preview can be found in log")
            with tqdm(total=batch_len,dynamic_ncols=True,ascii=' >=') as progressbar:
                for batchi, data in enumerate(self._dataloader):
                    if self._do_log_sample:
                        # torch.set_printoptions(profile='full')
                        _logger.debug(data)
                        # torch.set_printoptions(profile='default')
                    yield batchi, data
                    progressbar.update()
                    postfix={'loss':str(self._realtime_loss_probe)}
                    progressbar.set_postfix(postfix)
                    if batchi%one_percent_len==0:
                        _logger.debug(f"progress {batchi}/{batch_len}: {postfix}")
                    if self._dry_run and batchi>=2:
                        break
        else:
            for batchi, data in enumerate(self._dataloader):
                yield batchi, data
                if self._dry_run and batchi>=2:
                    break
    
    def _check_update(self):
        if self._do_check_score and not self._score_updated:
            _logger.error(f"no score updated during phase {self._phase_name}")
        if self._do_check_loss and not self._loss_updated:
            _logger.error(f"no loss updated during phase {self._phase_name}")

        for name in self._custom_probe_name:
            _logger.error(f"{name} not updated during phase {self._phase_name}")

    def update_probe(self, name: str, value: float, n: int = 1):
        if name in self._custom_probe_name:
            self._custom_probe_name.remove(name)
        self._custom_probes[name].update(value, n)

    @staticmethod
    def validate_loss(loss: Tensor, panic: bool = True) -> bool:
        return validate_loss(loss,panic)

    @staticmethod
    def validate_tensor(t: Tensor, panic: bool = True, msg: str = "") -> bool:
        return validate_tensor(t,panic,msg)

    def update_loss(self, loss: Tensor, n: int = 1):
        self._loss_updated = True
        self.validate_loss(loss)
        self._loss_probe.update(loss.item(), n)
        self._realtime_loss_probe.add(loss.item())

    def end_phase(self, score: float):
        self._score_updated = True
        self._score.update(score)

    @property
    def loss_probe(self):
        return self._loss_probe

    def score(self):
        return self._score.average()

    @property
    def custom_probes(self):
        return self._custom_probes