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
            self.logger.info(f"Backup tensor shape: {self._tensor_backup.shape}, device: {self._tensor_backup.device}")
            mask = self.generate_global_mask(tensor, condition_fns)

            # 检查 mask 的合法性
            if mask.numel() == 0 or mask.sum() == 0:
                self.logger.warning("mask 为空或无元素满足条件")
                return tensor

            # 索引缓存机制
            selected_indices = self._get_cached_indices(mask)
            self.logger.info(f"Selected indices shape: {selected_indices.shape}, device: {selected_indices.device}")
            assert selected_indices.max() < tensor.numel(), f"索引 {selected_indices.max()} 超出 tensor 维度 {tensor.shape}"

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
            self.logger.info(f"Backup tensor shape: {self._tensor_backup.shape}, device: {self._tensor_backup.device}")
            mask = self.generate_global_mask(tensor, condition_fns)

            # 检查 mask 的合法性
            if mask.numel() == 0 or mask.sum() == 0:
                self.logger.warning("mask 为空或无元素满足条件")
                return tensor

            # 索引缓存机制
            selected_indices = self._get_cached_indices(mask)
            assert selected_indices.max() < tensor.numel(), f"索引 {selected_indices.max()} 超出 tensor 维度 {tensor.shape}"

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

    def generate_global_mask(self, tensor, condition_fns):
        numels = torch.tensor([tensor.numel()], device=tensor.device)
        gathered = [torch.empty_like(numels) for _ in range(self.world_size)]
        dist.all_gather(gathered, numels)

        if not all(n == gathered[0] for n in gathered):
            raise ValueError("所有节点的张量维度必须一致")

        local_mask = ConditionBuilder.vectorized_condition(condition_fns)(tensor)
        self.logger.info(f"Local mask shape: {local_mask.shape}, device: {local_mask.device}")

        # 合并元数据传输
        metadata = torch.tensor([
            local_mask.sum().item(),
            tensor.median().item(),
            tensor.numel()
        ], device=tensor.device)

        self.logger.info(f"Metadata shape: {metadata.shape}, device: {metadata.device}")

        gathered_meta = [torch.empty_like(metadata) for _ in range(self.world_size)]
        dist.all_gather(gathered_meta, metadata)

        self.logger.info(f"Gathered metadata shape: {gathered_meta[0].shape}, device: {gathered_meta[0].device}")

        # 计算全局阈值
        total_selected = sum(t[0] for t in gathered_meta)
        global_median = torch.median(torch.stack([t[1] for t in gathered_meta]))
        mask = (tensor > global_median) & (local_mask) & (total_selected > 0)

        self.logger.info(f"Global mask shape: {mask.shape}, device: {mask.device}")

        return mask

    def _get_cached_indices(self, mask: torch.Tensor) -> torch.Tensor:
        key = (mask.shape, mask.device)
        required_size = mask.sum().item()

        with torch.no_grad():
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

        self.logger.info(f"Checksum tensor shape: {checksum_tensor.shape}, device: {checksum_tensor.device}")

        gathered = [torch.empty_like(checksum_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, checksum_tensor)

        self.logger.info(f"Gathered checksum tensors: {[t.item() for t in gathered]}")

        return all(t == gathered[0] for t in gathered)

    def global_rollback(self, tensor: torch.Tensor):
        self.logger.info(f"Starting global rollback with tensor shape: {self._tensor_backup.shape}, device: {self._tensor_backup.device}")
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

    def get_participation_count(self, indices: torch.Tensor) -> torch.Tensor:
        """获取参与通信的进程数量"""
        counts = torch.ones_like(indices, device=indices.device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return counts







