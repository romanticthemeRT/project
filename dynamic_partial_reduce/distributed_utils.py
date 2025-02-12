import torch.distributed as dist

def synchronize():
    """
    同步所有分布式进程。
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()