from pathlib import Path

from rehab_minillm.config import load_config


def test_small_config_loads():
    path = Path(__file__).parents[1] / "configs" / "small.yaml"
    cfg = load_config(path)
    assert cfg.model.vocab_size == 8000
    assert cfg.model.d_model % cfg.model.n_heads == 0
