import os
from dataclasses import dataclass
from transformers import AutoConfig
from nanovllm.utils.performance import optimize_cuda_operations, get_optimal_batch_size


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        
        # Apply CUDA optimizations
        optimize_cuda_operations()
        
        # Auto-optimize batch size if not explicitly set
        if hasattr(self, '_auto_optimize') and self._auto_optimize:
            import torch
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                model_size = self._estimate_model_size()
                optimal_batch = get_optimal_batch_size(model_size, available_memory)
                self.max_num_seqs = min(self.max_num_seqs, optimal_batch)
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB based on parameters."""
        if hasattr(self.hf_config, 'num_parameters'):
            return self.hf_config.num_parameters * 2 / 1e9  # 2 bytes per param (fp16)
        
        # Rough estimation for transformer models
        vocab_size = getattr(self.hf_config, 'vocab_size', 32000)
        hidden_size = getattr(self.hf_config, 'hidden_size', 1024)
        num_layers = getattr(self.hf_config, 'num_hidden_layers', 12)
        intermediate_size = getattr(self.hf_config, 'intermediate_size', hidden_size * 4)
        
        # Estimate parameters: embedding + transformer layers + head
        embedding_params = vocab_size * hidden_size
        layer_params = num_layers * (
            hidden_size * hidden_size * 3 +  # QKV projection
            hidden_size * hidden_size +      # Output projection
            intermediate_size * hidden_size * 2 +  # MLP
            hidden_size * 2                  # LayerNorms
        )
        head_params = vocab_size * hidden_size
        
        total_params = embedding_params + layer_params + head_params
        return total_params * 2 / 1e9  # 2 bytes per param (fp16)
    
    @classmethod
    def create_optimized(cls, model: str, **kwargs):
        """Create a config with automatic performance optimizations."""
        config = cls(model=model, **kwargs)
        config._auto_optimize = True
        config.__post_init__()
        return config
