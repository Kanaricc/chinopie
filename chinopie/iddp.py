import torch.distributed as dist
from torch.distributed import * # type:ignore
from torch.utils.data.distributed import DistributedSampler

_ddp_enabled=False

def enable_ddp():
    global _ddp_enabled
    _ddp_enabled=True

def is_preferred():
    return _ddp_enabled

def is_main_process():
    return not dist.is_initialized() or dist.get_rank()==0