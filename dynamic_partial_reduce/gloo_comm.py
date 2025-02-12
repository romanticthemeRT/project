import torch
import torch.distributed as dist
from .base_comm import BaseComm

class GlooComm(BaseComm):
    def __init__(self, rank: int, world_size: int, device: str = 'cpu'):
        super().__init__(rank, world_size)
        self.device = device

    def _backend_all_reduce(self, tensor: torch.Tensor, reduce_op):
        if tensor.device.type != self.device:
            raise RuntimeError(f"Gloo后端需要{self.device}张量, 检测到{tensor.device.type}张量!")
        dist.all_reduce(tensor, op=reduce_op)


