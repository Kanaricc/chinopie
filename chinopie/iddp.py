import torch.distributed as dist
from torch.distributed import * # type:ignore
from torch.utils.data.distributed import DistributedSampler

_ddp_preferred=False

def prefer_ddp():
    global _ddp_preferred
    _ddp_preferred=True

def is_preferred():
    return _ddp_preferred

def is_main_process():
    return not dist.is_initialized() or dist.get_rank()==0