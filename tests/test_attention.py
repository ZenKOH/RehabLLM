import torch

from rehab_minillm.attention import CausalSelfAttention
from rehab_minillm.config import ModelConfig


def test_causal_attention_cannot_see_future_tokens():
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=8,
        n_layers=1,
        n_heads=2,
        d_model=8,
        d_ff=32,
        dropout=0.0,
    )
    attn = CausalSelfAttention(cfg).eval()

    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(1, 8) * 100

    y1 = attn(x1)
    y2 = attn(x2)

    assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5)
    assert not torch.allclose(y1[:, 3], y2[:, 3])
