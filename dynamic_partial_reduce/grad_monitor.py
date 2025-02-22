import torch
from typing import Dict
import logging

class GradientProfiler:
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        初始化梯度检测器。

        Args:
            model (torch.nn.Module): 待检测的模型。
            device (str): 模型和梯度所在的设备。
        """
        self.hooks = []
        self.grad_data: Dict[str, torch.Tensor] = {}
        self.device = device

        for name, param in model.named_parameters():
            def make_hook(n):
                return lambda grad: self._capture_grad(n, grad)

            hook = param.register_hook(make_hook(name))
            self.hooks.append(hook)

    def _capture_grad(self, name: str, grad: torch.Tensor):
        """
        捕获并记录梯度数据。

        Args:
            name (str): 参数名称。
            grad (torch.Tensor): 参数的梯度。
        """
        if grad is not None:
            self.grad_data[name] = grad.clone().to(self.device)
            self.logger().info(f"Captured gradient for parameter {name}")

    def release(self):
        """
        释放所有梯度钩子。
        """
        for hook in self.hooks:
            hook.remove()
        self.grad_data.clear()
        self.logger().info("所有梯度钩子已释放")

    def logger(self):
        """
        获取日志记录器。

        Returns:
            logging.Logger: 日志记录器。
        """
        return logging.getLogger("GradientProfiler")