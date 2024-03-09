from inspect import signature
from transformer_engine.pytorch.attention import apply_rotary_pos_emb
import torch
from rotary_pos import RotaryPosEmbTriton


def test_rotary_pos_fw():
    in_tensor = torch.randn(8, 1, 12, 64)
    freqs = torch.randn(8, 1, 1, 64)
    print(signature(apply_rotary_pos_emb))
    cuda_rotary = apply_rotary_pos_emb(in_tensor, freqs)
    our_rotary_func = RotaryPosEmbTriton.apply
    our_rotary = our_rotary_func(in_tensor, freqs)

    torch.testing.assert_close(cuda_rotary, our_rotary)
