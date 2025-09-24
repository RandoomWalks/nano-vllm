import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from functools import lru_cache
import hashlib


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


@lru_cache(maxsize=32)
def _get_model_file_hash(file_path: str) -> str:
    """Get hash of model file for caching purposes."""
    stat = os.stat(file_path)
    return f"{stat.st_mtime}_{stat.st_size}"


def load_model(model: nn.Module, path: str):
    """Load model weights with optimized tensor loading."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    safetensors_files = glob(os.path.join(path, "*.safetensors"))
    
    # Sort files for consistent loading order
    safetensors_files.sort()
    
    # Pre-compute parameter lookup for faster access
    param_cache = {}
    
    print(f"Loading {len(safetensors_files)} safetensors files...")
    
    for i, file in enumerate(safetensors_files):
        print(f"Loading file {i+1}/{len(safetensors_files)}: {os.path.basename(file)}")
        
        with safe_open(file, "pt", "cpu") as f:
            # Get all weight names first for better memory access patterns
            weight_names = list(f.keys())
            
            for weight_name in weight_names:
                # Handle packed modules
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        
                        # Use cached parameter lookup
                        if param_name not in param_cache:
                            param_cache[param_name] = model.get_parameter(param_name)
                        param = param_cache[param_name]
                        
                        weight_loader = getattr(param, "weight_loader")
                        loaded_weight = f.get_tensor(weight_name)
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    # Handle regular parameters
                    if weight_name not in param_cache:
                        param_cache[weight_name] = model.get_parameter(weight_name)
                    param = param_cache[weight_name]
                    
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    loaded_weight = f.get_tensor(weight_name)
                    weight_loader(param, loaded_weight)
    
    print("Model loading completed successfully.")
