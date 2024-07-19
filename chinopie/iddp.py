import torch.distributed as dist
from torch.distributed import * # type:ignore
from torch.utils.data.distributed import DistributedSampler


def is_main_process():
    return not dist.is_initialized() or dist.get_rank()==0