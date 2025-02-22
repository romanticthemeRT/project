from typing import List, Callable

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
        self.logger.info(f"Backend all_reduce with tensor shape: {tensor.shape}, device: {tensor.device}")
        dist.all_reduce(tensor, op=reduce_op)

    def get_participation_count(self, indices: torch.Tensor) -> torch.Tensor:
        """获取参与通信的进程数量"""
        counts = torch.ones_like(indices, device=indices.device)
        self.logger.info(f"Participation count with tensor shape: {counts.shape}, device: {counts.device}")
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return counts

    def batched_all_reduce(
            self,
            tensor: torch.Tensor,
            indices: torch.Tensor,
            reduce_op,
            batch_size: int
    ) -> torch.Tensor:
        for i in range(0, indices.size(0), batch_size):
            batch_idx = indices[i:i + batch_size]

            # 确保索引合法
            if batch_idx.numel() == 0:
                break

            # 确保索引不会超过 tensor 的维度
            if batch_idx.max() >= tensor.numel():
                self.logger.error(f"索引 {batch_idx.max()} 超出 tensor 维度 {tensor.numel()}")
                raise IndexError(f"索引 {batch_idx.max()} 超出 tensor 维度 {tensor.shape}")

            batch_data = tensor.view(-1)[batch_idx]

            self.logger.info(f"Batch data shape: {batch_data.shape}, device: {batch_data.device}")
            self.logger.info(f"Batch idx shape: {batch_idx.shape}, device: {batch_idx.device}")

            # 通信核心操作
            self._backend_all_reduce(batch_data, reduce_op)
            counts = self.get_participation_count(batch_idx)

            self.logger.info(f"Counts shape: {counts.shape}, device: {counts.device}")

            # 使用安全除法更新
            if batch_data.dim() == 1:
                tensor.view(-1)[batch_idx] = batch_data / counts.float().clamp(min=1.0)  # 修正更新逻辑
            else:
                tensor.view(-1)[batch_idx] = batch_data / counts.unsqueeze(-1).float().clamp(min=1.0)  # 修正更新逻辑

            # 提前释放内存
            del batch_data, counts

        return tensor

    def dynamic_all_reduce(
            self,
            tensor: torch.Tensor,
            condition_fns: List[Callable[[torch.Tensor], torch.Tensor]],
            reduce_op=dist.ReduceOp.SUM,
            batch_size: int = 1024
    ) -> torch.Tensor:
        if tensor.device.type != self.device:
            raise RuntimeError(f"Gloo后端需要{self.device}张量, 检测到{tensor.device.type}张量!")
        return super().dynamic_all_reduce(tensor, condition_fns, reduce_op, batch_size)













