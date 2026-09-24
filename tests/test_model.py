import torch

from rehab_minillm.config import ModelConfig
from rehab_minillm.model import RehabMiniLLM


def small_config():
    return ModelConfig(
        vocab_size=128,
        context_length=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=128,
        dropout=0.0,
    )


def test_model_forward_shape_and_loss():
    model = RehabMiniLLM(small_config())
    x = torch.randint(0, 128, (2, 12))
    y = torch.randint(0, 128, (2, 12))
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, 128)
    assert loss is not None
    assert torch.isfinite(loss)


def test_embedding_weights_are_tied():
    model = RehabMiniLLM(small_config())
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_generation_appends_tokens():
    torch.manual_seed(1)
    model = RehabMiniLLM(small_config()).eval()
    x = torch.tensor([[2, 5, 9]], dtype=torch.long)
    out = model.generate(x, max_new_tokens=4, temperature=1.0, top_k=10, top_p=0.9)
    assert out.shape == (1, 7)
