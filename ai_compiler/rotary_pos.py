from typing import Union
import torch

import triton
import triton.language as tl


class RotaryPosEmbTriton(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        return t

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        return grad_input, None, None, None, None


