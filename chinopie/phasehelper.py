import sys
import inspect
from typing import List,Dict,Any,Optional,Callable
from typing_extensions import Self

from torch.utils.data import DataLoader
from torch import Tensor
from loguru import logger
from tqdm import tqdm

from .probes import AverageMeter,SmoothMeanMeter
from .ddpsession import DdpSession

class FunctionalSection:
    class JumpSectionException(Exception):
        pass

    def __init__(self,break_phase:bool,report_cb:Optional[Callable[[Dict[str,Any]],None]]=None) -> None:
        self._break_phase=break_phase
        self._state:Dict[str,Any]={}
        self._report_cb=report_cb

    def set(self,key:str,val:Any=True):
        self._state[key]=val

    def __enter__(self):
        if self._break_phase:
            sys.settrace(lambda *args,**keys: None)
            frame=sys._getframe(1)
            frame.f_trace=self.trace
        return self
    
    def trace(self,frame,event,arg):
        raise self.JumpSectionException()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type,self.JumpSectionException):
            return True
        
        if self._report_cb:
            self._report_cb(self._state)


class PhaseHelper:
    def __init__(
            self,
            phase_name: str,
            dataset: Any,
            dataloader: DataLoader,
            ddp_session: Optional[DdpSession] = None,
            dry_run: bool = False,
            custom_probes: List[str] = [],
            break_phase: bool = False,
    ) -> None:
        self._phase_name = phase_name
        self._dry_run = dry_run
        self._ddp_session = ddp_session
        self._dataset = dataset
        self._dataloader = dataloader

        self._custom_probe_name = custom_probes
        self._break_phase = break_phase

    def get_data_sample(self):
        for data in self._dataloader:
            return data

    def range_data(self):
        batch_len = len(self._dataloader)
        if self._is_main_process():
            with tqdm(total=batch_len,ncols=64) as progressbar:
                for batchi, data in enumerate(self._dataloader):
                    if self._dry_run:
                        logger.info("data preview")
                        logger.info(data)
                    yield batchi, data
                    progressbar.update()
                    progressbar.set_postfix({'loss':self._realtime_loss_probe})
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
        self._realtime_loss_probe=SmoothMeanMeter()
        self._custom_probes = dict(
            [(x, AverageMeter(x)) for x in self._custom_probe_name]
        )

        self._loss_updated = False
        self._score_updated = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._score_updated:
            logger.error(f"no score updated during phase {self._phase_name}")
        if not self._loss_updated:
            logger.error(f"no loss updated during phase {self._phase_name}")

        for name in self._custom_probe_name:
            logger.error(f"{name} not updated during phase {self._phase_name}")

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
        self._realtime_loss_probe.add(loss.item(),n)

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