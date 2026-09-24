from pathlib import Path

import pytest

from rehab_minillm.train import latest_checkpoint, resolve_resume_checkpoint


def test_latest_checkpoint_selects_highest_step(tmp_path: Path):
    (tmp_path / "step_0000500.pt").write_bytes(b"a")
    (tmp_path / "step_0001500.pt").write_bytes(b"b")
    (tmp_path / "step_invalid.pt").write_bytes(b"c")
    assert latest_checkpoint(tmp_path).name == "step_0001500.pt"


def test_latest_checkpoint_returns_none_when_empty(tmp_path: Path):
    assert latest_checkpoint(tmp_path) is None
    assert resolve_resume_checkpoint("latest", tmp_path) is None


def test_explicit_resume_checkpoint_must_exist(tmp_path: Path):
    checkpoint = tmp_path / "step_0000500.pt"
    checkpoint.write_bytes(b"x")
    assert resolve_resume_checkpoint(checkpoint, tmp_path) == checkpoint

    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path / "missing.pt", tmp_path)
