import torch
import torch.distributed as dist
from datetime import timedelta
import threading

from . import setup_logger
from .base_comm import BaseComm
from .gloo_comm import GlooComm

class CommFactory:
    _init_lock = threading.Lock()
    _initialized = False

    @classmethod
    def reset(cls):
        with cls._init_lock:
            cls._initialized = False

    @classmethod
    def create(
            cls,
            backend: str,
            rank: int,
            world_size: int,
            master_addr: str,
            master_port: int,
            auto_device: bool = False
    ) -> BaseComm:
        with cls._init_lock:
            if dist.is_initialized():
                dist.destroy_process_group()
                cls._initialized = False
                logger = setup_logger(rank)
                logger.info("Process group destroyed and reset")

            if not cls._initialized:
                logger = setup_logger(rank)
                logger.info(f"Initializing distributed process group with {backend} for rank {rank}")
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size,
                    init_method=f"tcp://{master_addr}:{master_port}",
                    timeout=timedelta(seconds=10)
                )
                cls._initialized = True
                logger.info("Distributed process group initialized")

        if backend == 'gloo':
            device = 'cpu'  
            comm = GlooComm(rank, world_size, device=device)
            logger.info(f"Created GlooComm with device {device}")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return comm






