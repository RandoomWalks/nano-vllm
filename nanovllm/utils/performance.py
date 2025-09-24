"""
Performance optimization utilities for nano-vLLM.

This module provides utilities for optimizing memory usage, tensor operations,
and overall performance of the LLM inference engine.
"""

import torch
import gc
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
import functools


class MemoryPool:
    """Simple memory pool for tensor reuse to reduce allocation overhead."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.pools: Dict[Tuple[torch.dtype, torch.device, Tuple[int, ...]], list] = {}
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Get a tensor from the pool or create a new one."""
        key = (dtype, device, shape)
        pool = self.pools.get(key, [])
        
        if pool:
            return pool.pop().zero_()
        else:
            return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse."""
        key = (tensor.dtype, tensor.device, tuple(tensor.shape))
        pool = self.pools.setdefault(key, [])
        
        if len(pool) < self.max_size:
            pool.append(tensor.detach())
    
    def clear(self) -> None:
        """Clear all tensors from the pool."""
        self.pools.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global memory pool instance
_memory_pool = MemoryPool()


def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance."""
    return _memory_pool


@contextmanager
def torch_inference_mode():
    """Context manager for inference mode with additional optimizations."""
    with torch.inference_mode():
        # Disable gradient computation
        torch.set_grad_enabled(False)
        try:
            yield
        finally:
            torch.set_grad_enabled(True)


def optimize_tensor_creation(func):
    """Decorator to optimize tensor creation by reusing memory when possible."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Store original tensor creation functions
        original_zeros = torch.zeros
        original_empty = torch.empty
        original_full = torch.full
        
        pool = get_memory_pool()
        
        def pooled_zeros(size, *, dtype=None, device=None, **other_kwargs):
            if dtype is None:
                dtype = torch.get_default_dtype()
            if device is None:
                device = torch.device('cpu')
            if isinstance(device, str):
                device = torch.device(device)
            
            if isinstance(size, int):
                size = (size,)
            elif not isinstance(size, tuple):
                size = tuple(size)
            
            return pool.get_tensor(size, dtype, device)
        
        # Monkey patch tensor creation functions
        torch.zeros = pooled_zeros
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore original functions
            torch.zeros = original_zeros
            torch.empty = original_empty
            torch.full = original_full
    
    return wrapper


def optimize_cuda_operations():
    """Apply CUDA-specific optimizations."""
    if torch.cuda.is_available():
        # Enable tensor core usage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable optimized attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.9)


def profile_memory_usage():
    """Profile current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        print(f"GPU Memory - Allocated: {allocated / 1e9:.2f}GB, Cached: {cached / 1e9:.2f}GB")
    
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"CPU Memory - RSS: {memory_info.rss / 1e9:.2f}GB, VMS: {memory_info.vms / 1e9:.2f}GB")


def cleanup_memory():
    """Clean up memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear memory pool
    get_memory_pool().clear()


class PerformanceProfiler:
    """Simple performance profiler for tracking bottlenecks."""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    @contextmanager
    def profile(self, name: str):
        """Profile a code block."""
        import time
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            if name not in self.timings:
                self.timings[name] = 0
                self.counts[name] = 0
            self.timings[name] += elapsed
            self.counts[name] += 1
    
    def report(self):
        """Print performance report."""
        print("\nPerformance Report:")
        print("-" * 50)
        for name in sorted(self.timings.keys()):
            total_time = self.timings[name]
            count = self.counts[name]
            avg_time = total_time / count
            print(f"{name:30} | Total: {total_time:8.3f}s | Avg: {avg_time*1000:8.3f}ms | Count: {count:6d}")
    
    def clear(self):
        """Clear all timing data."""
        self.timings.clear()
        self.counts.clear()


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _profiler


def apply_compile_optimizations(model: torch.nn.Module) -> torch.nn.Module:
    """Apply torch.compile optimizations to a model."""
    if hasattr(torch, 'compile'):
        try:
            # Use different backends based on availability
            backends = ['inductor', 'aot_eager', 'eager']
            for backend in backends:
                try:
                    compiled_model = torch.compile(model, backend=backend, mode='max-autotune')
                    print(f"Successfully compiled model with backend: {backend}")
                    return compiled_model
                except Exception as e:
                    print(f"Failed to compile with {backend}: {e}")
                    continue
        except Exception as e:
            print(f"Compilation failed: {e}")
    
    return model


def get_optimal_batch_size(model_size_gb: float, available_memory_gb: float) -> int:
    """Estimate optimal batch size based on model and available memory."""
    # Simple heuristic: use 70% of available memory, account for model size
    usable_memory = available_memory_gb * 0.7 - model_size_gb
    
    # Rough estimate: each token uses ~4 bytes per parameter
    # For a 0.6B parameter model, each token uses ~2.4GB in full precision
    # But with optimizations, this can be much lower
    bytes_per_token = model_size_gb * 0.1  # Conservative estimate
    
    if bytes_per_token > 0:
        optimal_batch_size = max(1, int(usable_memory / bytes_per_token))
    else:
        optimal_batch_size = 32  # Default fallback
    
    return min(optimal_batch_size, 512)  # Cap at reasonable maximum


def warm_up_cuda():
    """Warm up CUDA kernels for better performance."""
    if torch.cuda.is_available():
        # Create dummy tensors to initialize CUDA context
        dummy = torch.randn(1024, 1024, device='cuda')
        _ = torch.matmul(dummy, dummy)
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()