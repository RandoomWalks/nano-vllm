import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        
        # Avoid dtype conversion if already in float32
        if orig_dtype == torch.float32:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x.mul(self.weight).div(torch.sqrt(var + self.eps))
        
        # Standard path with dtype conversion
        x_float = x.to(torch.float32)
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float.mul_(torch.rsqrt(var + self.eps))
        return x_float.to(orig_dtype).mul_(self.weight)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        
        # Optimize for float32 case
        if orig_dtype == torch.float32:
            x.add_(residual)
            residual = x.clone()
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x.mul_(self.weight).div_(torch.sqrt(var + self.eps))
            return x, residual
        
        # Standard path with dtype conversion
        x_float = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x_float.to(orig_dtype)
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float.mul_(torch.rsqrt(var + self.eps))
        x_norm = x_float.to(orig_dtype).mul_(self.weight)
        return x_norm, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
