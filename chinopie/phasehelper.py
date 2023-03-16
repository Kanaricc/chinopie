from typing import List,Dict,Any,Optional,Callable
from typing_extensions import Self

from torch.utils.data import DataLoader
from torch import Tensor
from loguru import logger
from tqdm import tqdm

from probes import AverageMeter,NumericMeter
from ddpsession import DdpSession

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
            with tqdm(total=batch_len,ncols=64) as progressbar:
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