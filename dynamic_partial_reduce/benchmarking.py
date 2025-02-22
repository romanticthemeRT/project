import time

import pytest
import torch
from .base_comm import BaseComm
from .dynamic_conditions import ConditionBuilder

class ReduceBenchmarker:
    def __init__(self, comm: BaseComm, tensor_size=(4096, 4096), device: str = 'cpu'):
        self.comm = comm
        self.tensor = torch.randn(tensor_size, device=device, requires_grad=True)
        self.device = device

    def run_test(self, rounds=5, batch_sizes=[256, 1024, 4096]):
        results = {}
        for bs in batch_sizes:
            self.comm.logger.info(f"测试批次大小: {bs}")
            t_total = 0.0
            for _ in range(rounds):
                test_tensor = self.tensor.clone()
                self.comm.logger.info(f"Test tensor shape: {test_tensor.shape}, device: {test_tensor.device}")

                # 检查 indices 的生成逻辑
                indices = test_tensor.nonzero().flatten()
                assert indices.max() < test_tensor.numel(), f"索引 {indices.max()} 超出 tensor 维度 {test_tensor.shape}"

                try:
                    start = time.perf_counter()
                    reduced_tensor = self.comm.dynamic_all_reduce(
                        test_tensor,
                        condition_fns=[ConditionBuilder.max_value(0.5)],
                        batch_size=bs
                    )
                    t_total += time.perf_counter() - start

                    self.comm.logger.info(f"Reduced tensor shape: {reduced_tensor.shape}, device: {reduced_tensor.device}")
                    assert reduced_tensor.device == torch.device(self.device)
                    assert reduced_tensor.shape == test_tensor.shape
                except RuntimeError as e:
                    self.comm.logger.error(f"dynamic_all_reduce failed: {e}")
                    pytest.fail(f"dynamic_all_reduce failed: {e}")

            results[bs] = t_total / rounds
            self.comm.logger.info(f"Batch size: {bs}, Average time: {results[bs]:.4f} seconds")

        return results









