import logging
import os

def setup_logger(rank: int, log_dir: str = "logs", log_level: int = logging.INFO):
    """
    设置日志记录器。

    Args:
        rank (int): 当前进程的排名。
        log_dir (str): 日志文件的保存目录。
        log_level (int): 日志记录级别。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(log_level)
    handler = logging.FileHandler(f"{log_dir}/rank_{rank}.log")
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

