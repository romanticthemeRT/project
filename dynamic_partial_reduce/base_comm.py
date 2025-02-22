import binascii
import sys
from abc import ABC
import torch
import torch.distributed as dist
import logging
from typing import List, Callable, Dict, Tuple
from .dynamic_conditions import ConditionBuilder

class BaseComm(ABC):
    # 索引缓存优化
    _index_cache: Dict[Tuple[torch.Size, torch.device], torch.Tensor] = {}

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger(f"rank_{rank}")
        self._tensor_backup = None

    def dynamic_all_reduce(
            self,
            tensor: torch.Tensor,
            condition_fns: List[Callable[[torch.Tensor], torch.Tensor]],
            reduce_op=dist.ReduceOp.SUM,
            batch_size: int = 1024
    ) -> torch.Tensor:
        try:
            # 备份与条件检测
            self._tensor_backup = tensor.detach().clone()
            mask = self.generate_global_mask(tensor, condition_fns)

            # 索引缓存机制
            selected_indices = self._get_cached_indices(mask)
            if selected_indices.numel() == 0:
                return tensor

            # 动态分批次处理
            tensor = self.batched_all_reduce(tensor, selected_indices, reduce_op, batch_size)

            # 数据完整性校验
            if self.rank == 0 and not self.validate_checksum(tensor):
                self.logger.warning("Checksum验证失败，触发全局恢复")
                self.global_rollback(tensor)

            return tensor
        except Exception as e:
            self.logger.error(f"触发本地回滚 (错误类型: {type(e).__name__})")
            if self._tensor_backup is not None:
                tensor = self._tensor_backup.clone()  # 避免原地操作
                self.logger.warning("已从备份恢复张量数据")
            else:
                self.logger.critical("无可用备份数据！")
            raise

    def batched_all_reduce(
            self,
            tensor: torch.Tensor,
            indices: torch.Tensor,
            reduce_op,
            batch_size: int
    ) -> torch.Tensor:
        for i in range(0, indices.size(0), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_data = tensor[batch_idx]

            # 通信核心操作
            self._backend_all_reduce(batch_data, reduce_op)
            counts = self.get_participation_count(batch_idx)

            # 使用安全除法更新
            tensor[batch_idx] = batch_data / counts.float().clamp(min=1.0)

            # 提前释放内存
            del batch_data, counts

        return tensor

    def generate_global_mask(self, tensor, condition_fns):
        numels = torch.tensor([tensor.numel()], device=tensor.device)
        gathered = [torch.empty_like(numels) for _ in range(self.world_size)]
        dist.all_gather(gathered, numels)

        if not all(n == gathered[0] for n in gathered):
            raise ValueError("所有节点的张量维度必须一致")

        local_mask = ConditionBuilder.vectorized_condition(condition_fns)(tensor)
        # 合并元数据传输
        metadata = torch.tensor([
            local_mask.sum().item(),
            tensor.median().item(),
            tensor.numel()
        ], device=tensor.device)

        gathered_meta = [torch.empty_like(metadata) for _ in range(self.world_size)]
        dist.all_gather(gathered_meta, metadata)

        # 计算全局阈值
        total_selected = sum(t[0] for t in gathered_meta)
        global_median = torch.median(torch.stack([t[1] for t in gathered_meta]))
        mask = (tensor > global_median) & (local_mask) & (total_selected > 0)

        return mask

    def _get_cached_indices(self, mask: torch.Tensor) -> torch.Tensor:
        key = (mask.shape, mask.device)
        required_size = mask.sum().item()

        with torch.no_grad():
            # 启用CUDA异步加速
            device = mask.device
            if key not in self._index_cache or self._index_cache[key].size(0) < required_size:
                new_size = max(int(required_size * 1.5), 1024)  # 按1.5倍扩容
                self._index_cache[key] = torch.empty(
                    new_size, dtype=torch.long, device=device
                )
                self.logger.info(f"扩展索引缓存: {key} -> {new_size}")

        cache = self._index_cache[key]
        indices = torch.nonzero(mask.flatten(), as_tuple=False)
        cache[:indices.size(0)] = indices.squeeze(1)

        return cache[:indices.size(0)].clone()

    def validate_checksum(self, tensor: torch.Tensor) -> bool:
        header_bytes = tensor.detach().cpu().numpy()[:256].tobytes()
        crc = binascii.crc32(header_bytes) & 0xFFFFFFFF  # 确保unsigned
        checksum_tensor = torch.tensor([crc], dtype=torch.int64, device=tensor.device)

        gathered = [torch.empty_like(checksum_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, checksum_tensor)

        return all(t == gathered[0] for t in gathered)

    def global_rollback(self, tensor: torch.Tensor):
        dist.broadcast(self._tensor_backup, src=0)
        tensor.copy_(self._tensor_backup)

    def rollback_procedure(self, tensor: torch.Tensor):
        """回滚保护机制"""
        self.logger.error(f"触发本地回滚 (错误类型: {type(sys.exc_info()[1]).__name__})")
        if self._tensor_backup is not None:
            tensor = self._tensor_backup.clone()  # 避免原地操作
            self.logger.warning("已从备份恢复张量数据")
        else:
            self.logger.critical("无可用备份数据！")




