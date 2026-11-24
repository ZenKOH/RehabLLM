import json
import os
import subprocess
import sys
from pathlib import Path

from rehab_minillm.instruction import load_instruction_jsonl, validate_instruction_splits


def test_preparation_keeps_alternate_answers_together_and_preserves_provenance(tmp_path):
    root = Path(__file__).parents[1]
    source = tmp_path / "raw.jsonl"
    rows = [
        {"instruction": "Explain research.", "response": "Answer one.", "source_ids": ["fixture"]},
        {"instruction": " explain  RESEARCH. ", "response": "Answer two."},
        {"instruction": "Compare evidence.", "response": "Answer three."},
        {"instruction": "Review sources.", "response": "Answer four."},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "prepared"
    subprocess.run([
        sys.executable, "scripts/prepare_instructions.py", "--input", str(source),
        "--out-dir", str(output),
    ], cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")}, check=True,
       capture_output=True, text=True)
    train, val = [load_instruction_jsonl(output / f"{split}.jsonl") for split in ("train", "val")]
    validate_instruction_splits(train, val)
    assert len(train) + len(val) == 4
    assert any(len([e for e in split if "research" in e.instruction.lower()]) == 2
               for split in (train, val))
    prepared = [json.loads(line) for split in ("train", "val")
                for line in (output / f"{split}.jsonl").read_text().splitlines()]
    assert next(r for r in prepared if r["response"] == "Answer one.")["source_ids"] == ["fixture"]
