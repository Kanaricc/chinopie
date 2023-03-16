import torch.distributed as dist

class DdpSession:
    def __init__(self):
        pass

    @staticmethod
    def is_main_process():
        return dist.get_rank() == 0
    
    @staticmethod
    def barrier():
        dist.barrier()
    
    def get_world_size(self):
        return dist.get_world_size()
    
    def get_rank(self):
        return dist.get_rank()
    
    def gather_object(self, obj, object_gather_list=None, dst=0, group=None):
        return dist.gather_object(obj,object_gather_list,dst,group)