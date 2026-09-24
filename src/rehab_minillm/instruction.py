from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenizer import SentencePieceTokenizer


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    response: str
    context: str = ""
    id: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> InstructionExample:
        instruction = str(row.get("instruction") or "").strip()
        response = str(row.get("response") or "").strip()
        context = str(row.get("context") or "").strip()
        identifier = str(row.get("id") or "").strip()
        if not instruction:
            raise ValueError("Instruction example is missing 'instruction'")
        if not response:
            raise ValueError("Instruction example is missing 'response'")
        return cls(
            instruction=instruction,
            response=response,
            context=context,
            id=identifier,
        )


def format_instruction_prompt(example: InstructionExample) -> str:
    parts = ["### Instruction:\n", example.instruction.strip(), "\n\n"]
    if example.context:
        parts.extend(["### Context:\n", example.context.strip(), "\n\n"])
    parts.append("### Response:\n")
    return "".join(parts)


def example_fingerprint(example: InstructionExample) -> str:
    payload = (
        f"{example.instruction.strip()}\n{example.context.strip()}\n{example.response.strip()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_instruction_jsonl(path: str | Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(InstructionExample.from_dict(json.loads(line)))
    return examples


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """SFT dataset with prompt tokens masked from the language-model loss."""

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer: SentencePieceTokenizer,
        context_length: int,
    ) -> None:
        if not examples:
            raise ValueError("InstructionDataset requires at least one example")
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        prompt = format_instruction_prompt(example)
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        response_ids = self.tokenizer.encode(example.response, add_eos=True)

        max_total = self.context_length + 1
        if len(prompt_ids) >= max_total:
            prompt_ids = prompt_ids[: max_total - 1]
        remaining = max_total - len(prompt_ids)
        response_ids = response_ids[:remaining]

        full = prompt_ids + response_ids
        if len(full) < 2:
            raise ValueError("Instruction example produced fewer than two tokens")

        x = full[:-1]
        y = full[1:]
        prompt_target_positions = max(0, len(prompt_ids) - 1)
        labels = [-100] * prompt_target_positions + y[prompt_target_positions:]

        pad_id = self.tokenizer.pad_id
        pad_len = self.context_length - len(x)
        if pad_len > 0:
            x = x + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
        else:
            x = x[: self.context_length]
            labels = labels[: self.context_length]

        return torch.tensor(x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
