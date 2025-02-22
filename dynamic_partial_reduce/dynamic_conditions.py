import torch
from typing import List, TypeVar, Callable
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)

class ConditionBuilder:
    @staticmethod
    def vectorized_condition(condition_fns: List[T]) -> Callable:
        """
        创建一个JIT编译的条件链，该链将所有条件合并成一个布尔掩码。

        Args:
            condition_fns (List[T]): 一个包含条件函数的列表。

        Returns:
            Callable: 一个JIT编译的条件链函数。
        """
        class ConditionChain(torch.jit.ScriptModule):
            def __init__(self, condition_fns: List[T]):
                super().__init__()
                self.condition_fns = torch.nn.ModuleList(condition_fns)

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mask = torch.ones_like(x, dtype=torch.bool)
                for fn in self.condition_fns:
                    mask &= fn(x)
                return mask

        return ConditionChain(condition_fns)

    @staticmethod
    def max_value(threshold: float) -> T:
        """
        创建一个JIT编译的条件函数，用于检查张量元素是否超过给定阈值。

        Args:
            threshold (float): 阈值。

        Returns:
            T: 一个JIT编译的条件函数。
        """
        class ThresholdCheck(torch.jit.ScriptModule):
            __constants__ = ['threshold']

            def __init__(self, threshold: float):
                super().__init__()
                self.threshold = threshold

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x > self.threshold

        return ThresholdCheck(threshold)

    @staticmethod
    def min_gradient(grad_thresh: float) -> T:
        """
        创建一个JIT编译的条件函数，用于检查张量梯度的绝对值是否低于给定阈值。

        Args:
            grad_thresh (float): 梯度阈值。

        Returns:
            T: 一个JIT编译的条件函数。
        """
        class GradCheck(torch.jit.ScriptModule):
            __constants__ = ['grad_thresh']

            def __init__(self, grad_thresh: float):
                super().__init__()
                self.grad_thresh = grad_thresh

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                grad = self._safe_get_grad(x)
                return torch.abs(grad) > self.grad_thresh

            @torch.jit.script_method
            def _safe_get_grad(self, x: torch.Tensor) -> torch.Tensor:
                # 确保 x.grad 始终是一个张量
                if not torch.is_grad_enabled() or x.grad is None:
                    return torch.zeros_like(x, memory_format=torch.preserve_format)
                return x.grad.detach()

        return GradCheck(grad_thresh)



