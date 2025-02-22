import time
import torch
import pytest
import torch.distributed as dist
from datetime import timedelta
from dynamic_partial_reduce import CommFactory, ConditionBuilder, ReduceBenchmarker, setup_logger

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.fixture(scope="function")
def distributed_backend():
    """配置分布式环境的初始化和销毁"""
    rank = 0
    world_size = 1  # 单节点测试

    cleanup_distributed()  # 确保在每次测试前销毁已存在的进程组
    CommFactory.reset()  # 重置 CommFactory 的初始化状态
    setup_distributed_backend(rank, world_size)
    backend = "gloo"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()  # 清空 CUDA 缓存
    comm = CommFactory.create(backend, rank, world_size, "127.0.0.1", "29500", auto_device=True)
    yield comm
    cleanup_distributed()

def setup_distributed_backend(rank: int, world_size: int, backend: str = 'gloo'):
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method='tcp://127.0.0.1:29500',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10)
        )
        logger = setup_logger(rank)
        logger.info(f"Distributed backend initialized with {backend} for rank {rank}")

def test_gloo_cuda_error(distributed_backend):
    """验证Gloo后端拒绝CUDA张量"""
    if not torch.cuda.is_available():  # 环境检查
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_gloo_cuda_error")

    test_data = torch.randn(10).cuda()

    with pytest.raises(RuntimeError, match="Gloo后端需要cpu张量, 检测到cuda张量!"):
        distributed_backend.dynamic_all_reduce(test_data, [])

    logger.info("Test_gloo_cuda_error passed")

def test_jit_condition(distributed_backend):
    """验证JIT条件链正常运作"""
    logger = setup_logger(0)
    logger.info("Starting test_jit_condition")

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]
    jit_cond = ConditionBuilder.vectorized_condition(conds)

    test_tensor = torch.tensor([0.3, 0.6, 0.8], requires_grad=True)
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0])

    mask = jit_cond.forward(test_tensor)
    assert mask.tolist() == [False, False, False]

    logger.info("Test_jit_condition passed")

def test_min_gradient_with_none(distributed_backend):
    """验证无梯度时的处理逻辑"""
    logger = setup_logger(0)
    logger.info("Starting test_min_gradient_with_none")

    x = torch.tensor([1.0], requires_grad=True)

    with torch.enable_grad():
        y = x * 2
        y.backward(torch.ones_like(y))  # 生成初始梯度

    with torch.no_grad():
        x.grad = torch.zeros_like(x, memory_format=torch.preserve_format)

    cond_fn = ConditionBuilder.min_gradient(0.1)
    mask = cond_fn.forward(x)
    assert not mask.any()

    logger.info("Test_min_gradient_with_none passed")

def test_min_gradient_with_no_grad_enabled(distributed_backend):
    """验证梯度计算未启用时的处理逻辑"""
    logger = setup_logger(0)
    logger.info("Starting test_min_gradient_with_no_grad_enabled")

    x = torch.tensor([1.0], requires_grad=True)

    with torch.enable_grad():
        y = x * 2
        y.backward(torch.ones_like(y))

    with torch.no_grad():
        cond_fn = ConditionBuilder.min_gradient(0.1)
        mask = cond_fn.forward(x)
        assert not mask.any()

    logger.info("Test_min_gradient_with_no_grad_enabled passed")

def test_min_gradient_with_nonexistent_grad(distributed_backend):
    """验证张量没有梯度时的处理逻辑"""
    logger = setup_logger(0)
    logger.info("Starting test_min_gradient_with_nonexistent_grad")

    x = torch.tensor([1.0], requires_grad=True)

    with torch.no_grad():
        cond_fn = ConditionBuilder.min_gradient(0.1)
        mask = cond_fn.forward(x)
        assert not mask.any()

    logger.info("Test_min_gradient_with_nonexistent_grad passed")

def test_repeated_registration(distributed_backend):
    """验证重复条件创建不会报错"""
    logger = setup_logger(0)
    logger.info("Starting test_repeated_registration")

    conds = [
        ConditionBuilder.min_gradient(0.1),
        ConditionBuilder.min_gradient(0.2)  # 多次创建
    ]

    assert all(isinstance(f, torch.jit.ScriptModule) for f in conds)

    logger.info("Test_repeated_registration passed")

def test_device_mismatch_fallback(distributed_backend):
    """验证设备不匹配时的自动修正"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_device_mismatch_fallback")

    x = torch.tensor([1.0], device='cuda:0', requires_grad=True)
    y = x * 2
    y.backward(torch.tensor([0.1], device='cuda:0'))  # 生成梯度，确保梯度值小于阈值
    cond_fn = ConditionBuilder.min_gradient(0.2)

    # 将张量从 GPU 复制到 CPU
    x_cpu = x.detach().cpu().clone()
    x_cpu.grad = x.grad.cpu()  # 确保梯度也从 GPU 复制到 CPU

    mask_cpu = cond_fn.forward(x_cpu)
    mask = mask_cpu.cuda()

    assert mask.device == x.device
    assert not mask.any()

    logger.info("Test_device_mismatch_fallback passed")

def test_dynamic_all_reduce_cuda_cpu(distributed_backend):
    """验证dynamic_all_reduce在CUDA上的正常运作（CPU模拟）"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_dynamic_all_reduce_cuda_cpu")

    test_tensor = torch.tensor([0.3, 0.6, 0.8], device='cuda:0', requires_grad=True)
    y = test_tensor * 2
    y.backward(torch.ones_like(y))  # 生成梯度
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0], device='cuda:0')

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    # 将张量从 GPU 复制到 CPU
    test_tensor_cpu = test_tensor.detach().cpu().clone()
    test_tensor_grad_cpu = test_tensor.grad.cpu()
    test_tensor_cpu.grad = test_tensor_grad_cpu  # 确保梯度也从 GPU 复制到 CPU

    reduced_tensor_cpu = distributed_backend.dynamic_all_reduce(test_tensor_cpu, conds)

    # 将结果复制回 GPU
    reduced_tensor = reduced_tensor_cpu.cuda()

    assert torch.allclose(reduced_tensor, torch.tensor([0.3, 0.6, 0.8], device='cuda:0', requires_grad=True), atol=1e-6)

    logger.info("Test_dynamic_all_reduce_cuda_cpu passed")

def test_large_tensor_performance(distributed_backend):
    """验证大张量在dynamic_all_reduce中的性能"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_large_tensor_performance")

    test_tensor = torch.randn(100, device='cuda:0', requires_grad=True)
    y = test_tensor * 2
    y.backward(torch.ones_like(y))  # 生成梯度
    test_tensor.grad = torch.randn_like(test_tensor)  # 重新生成梯度

    logger.info(f"Test tensor shape: {test_tensor.shape}, device: {test_tensor.device}")
    logger.info(f"Test tensor grad shape: {test_tensor.grad.shape}, device: {test_tensor.grad.device}")

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    # 将张量从 GPU 复制到 CPU
    test_tensor_cpu = test_tensor.detach().cpu().clone()
    test_tensor_grad_cpu = test_tensor.grad.cpu().clone()
    test_tensor_cpu.grad = test_tensor_grad_cpu  # 确保梯度也从 GPU 复制到 CPU

    logger.info(f"Test tensor CPU shape: {test_tensor_cpu.shape}, device: {test_tensor_cpu.device}")
    logger.info(f"Test tensor CPU grad shape: {test_tensor_grad_cpu.shape}, device: {test_tensor_grad_cpu.device}")

    start = time.perf_counter()
    try:
        reduced_tensor_cpu = distributed_backend.dynamic_all_reduce(test_tensor_cpu, conds, batch_size=10)
        logger.info(f"Reduced tensor CPU shape: {reduced_tensor_cpu.shape}, device: {reduced_tensor_cpu.device}")
    except RuntimeError as e:
        logger.error(f"dynamic_all_reduce failed: {e}")
        pytest.fail(f"dynamic_all_reduce failed: {e}")

    end = time.perf_counter()

    # 将结果复制回 GPU
    reduced_tensor = reduced_tensor_cpu.cuda()

    assert reduced_tensor.device == torch.device('cuda:0')
    assert reduced_tensor.shape == test_tensor.shape
    print(f"大张量性能测试: Time taken: {end - start:.4f} seconds")

    logger.info("Test_large_tensor_performance passed")

def test_benchmark(distributed_backend):
    """验证条件链在实际应用中的性能"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_benchmark")

    device = 'cpu'  # 确保使用 CPU 设备
    benchmarker = ReduceBenchmarker(distributed_backend, tensor_size=(100, 100), device=device)

    logger.info(f" Benchmarker device: {device}")

    # 检查 indices 的生成逻辑
    indices = benchmarker.tensor.nonzero().flatten()
    logger.info(f" Indices: {indices}")
    assert indices.max() < benchmarker.tensor.numel(), f"索引 {indices.max()} 超出 tensor 维度 {benchmarker.tensor.shape}"

    results = benchmarker.run_test(rounds=5, batch_sizes=[10, 10])
    for batch_size, avg_time in results.items():
        print(f"Batch size: {batch_size}, Average time: {avg_time:.4f} seconds")

    logger.info("Test_benchmark passed")

def test_exception_handling(distributed_backend):
    """验证异常处理机制"""
    logger = setup_logger(0)
    logger.info("Starting test_exception_handling")

    test_tensor = torch.tensor([0.3, 0.6, 0.8], device='cpu', requires_grad=True)
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0], device='cpu')

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    try:
        # 故意触发错误
        distributed_backend.dynamic_all_reduce(test_tensor, conds, reduce_op=None)
    except TypeError as e:
        assert str(e) == "expected ReduceOp, but got NoneType"
        assert torch.allclose(test_tensor, torch.tensor([0.3, 0.6, 0.8], device='cpu', requires_grad=True), atol=1e-6)
    except RuntimeError as e:
        pytest.fail(f"Unexpected RuntimeError: {e}")

    logger.info("Test_exception_handling passed")


try:
    check = ConditionBuilder.min_gradient(0.1)
    scripted = torch.jit.script(check)
    print("TorchScript代码结构验证通过：")
    print(scripted.code)
except RuntimeError as e:
    print("编译失败:", e)
    raise

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")








