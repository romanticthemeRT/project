import time
import pytest
import torch
from datetime import timedelta
from dynamic_partial_reduce import CommFactory, ConditionBuilder, ReduceBenchmarker, setup_logger

def cleanup_distributed():
    import torch.distributed as dist
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
    comm = CommFactory.create("gloo", rank, world_size, "127.0.0.1", "29500")
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

    with pytest.raises(RuntimeError, match="Gloo后端需要CPU张量"):
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

try:
    check = ConditionBuilder.min_gradient(0.1)
    scripted = torch.jit.script(check)
    print("TorchScript代码结构验证通过：")
    print(scripted.code)
except RuntimeError as e:
    print("编译失败:", e)
    raise

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

    cond_fns = [
        ConditionBuilder.min_gradient(0.1),
        ConditionBuilder.min_gradient(0.2)  # 多次创建
    ]

    assert all(isinstance(f, torch.jit.ScriptModule) for f in cond_fns)

    logger.info("Test_repeated_registration passed")

def test_device_mismatch_fallback(distributed_backend):
    """验证设备不匹配时的自动修正"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_device_mismatch_fallback")

    x = torch.tensor([1.0], device='cuda:0', requires_grad=True)
    cond_fn = ConditionBuilder.min_gradient(0.1)

    mask = cond_fn.forward(x)
    assert mask.device == x.device
    assert not mask.any()

    logger.info("Test_device_mismatch_fallback passed")

def test_dynamic_all_reduce_cpu(distributed_backend):
    """验证dynamic_all_reduce在CPU上的正常运作"""
    logger = setup_logger(0)
    logger.info("Starting test_dynamic_all_reduce_cpu")

    test_tensor = torch.tensor([0.3, 0.6, 0.8], requires_grad=True)
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0])

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    reduced_tensor = distributed_backend.dynamic_all_reduce(test_tensor, conds)
    assert torch.allclose(reduced_tensor, torch.tensor([0.3, 0.6, 0.8], requires_grad=True), atol=1e-6)

    logger.info("Test_dynamic_all_reduce_cpu passed")

def test_dynamic_all_reduce_cuda_cpu(distributed_backend):
    """验证dynamic_all_reduce在CUDA上的正常运作（CPU模拟）"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_dynamic_all_reduce_cuda_cpu")

    test_tensor = torch.tensor([0.3, 0.6, 0.8], device='cuda:0', requires_grad=True)
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0], device='cuda:0')

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    reduced_tensor = distributed_backend.dynamic_all_reduce(test_tensor, conds)
    assert torch.allclose(reduced_tensor, torch.tensor([0.3, 0.6, 0.8], device='cuda:0', requires_grad=True), atol=1e-6)

    logger.info("Test_dynamic_all_reduce_cuda_cpu passed")

def test_benchmark(distributed_backend):
    """验证条件链在实际应用中的性能"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_benchmark")

    benchmarker = ReduceBenchmarker(distributed_backend, tensor_size=(4096, 4096), device='cuda:0')

    results = benchmarker.run_test(rounds=5, batch_sizes=[256, 1024, 4096])
    for batch_size, avg_time in results.items():
        print(f"Batch size: {batch_size}, Average time: {avg_time:.4f} seconds")

    logger.info("Test_benchmark passed")

def test_large_tensor_performance(distributed_backend):
    """验证大张量在dynamic_all_reduce中的性能"""
    if not torch.cuda.is_available():
        pytest.skip("当前环境无CUDA支持")
    logger = setup_logger(0)
    logger.info("Starting test_large_tensor_performance")

    test_tensor = torch.randn(1024 * 1024, device='cuda:0', requires_grad=True)
    test_tensor.grad = torch.randn_like(test_tensor)

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    start = time.perf_counter()
    reduced_tensor = distributed_backend.dynamic_all_reduce(test_tensor, conds, batch_size=1024)
    end = time.perf_counter()

    assert reduced_tensor.device == torch.device('cuda:0')
    assert reduced_tensor.shape == test_tensor.shape
    print(f"大张量性能测试: Time taken: {end - start:.4f} seconds")

    logger.info("Test_large_tensor_performance passed")

def test_exception_handling(distributed_backend):
    """验证异常处理机制"""
    logger = setup_logger(0)
    logger.info("Starting test_exception_handling")

    test_tensor = torch.tensor([0.3, 0.6, 0.8], requires_grad=True)
    test_tensor.grad = torch.tensor([0.0, 1e-5, 0.0])

    conds = [
        ConditionBuilder.max_value(0.5),
        ConditionBuilder.min_gradient(1e-4)
    ]

    try:
        # 故意触发错误
        distributed_backend.dynamic_all_reduce(test_tensor, conds, reduce_op=None)
    except TypeError as e:
        assert str(e) == "expected ReduceOp, but got NoneType"
        assert torch.allclose(test_tensor, torch.tensor([0.3, 0.6, 0.8], requires_grad=True), atol=1e-6)

    logger.info("Test_exception_handling passed")







