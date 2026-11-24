from __future__ import annotations

import hashlib
import json
import unicodedata
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
        if not isinstance(row, dict):
            raise TypeError("Instruction example must be an object")
        for field in ("instruction", "response", "context", "id"):
            if row.get(field) is not None and not isinstance(row[field], str):
                raise TypeError(f"Instruction field '{field}' must be text")
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


def prompt_fingerprint(example: InstructionExample) -> str:
    """Group alternate answers to the same normalised question/context together."""
    fields = [example.instruction, example.context]
    payload = json.dumps([
        " ".join(unicodedata.normalize("NFKC", value).casefold().split()) for value in fields
    ], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_instruction_splits(
    train: list[InstructionExample], val: list[InstructionExample]
) -> None:
    overlap = {prompt_fingerprint(e) for e in train} & {prompt_fingerprint(e) for e in val}
    if overlap:
        raise ValueError(f"Instruction train/validation prompt leakage: {len(overlap)} shared groups")


def load_instruction_jsonl(path: str | Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    examples.append(InstructionExample.from_dict(json.loads(line)))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return examples


class InstructionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Prompt-masked SFT, rejecting overflow before training instead of truncating answers."""

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer: SentencePieceTokenizer,
        context_length: int,
    ) -> None:
        if not examples:
            raise ValueError("InstructionDataset requires at least one example")
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if tokenizer.pad_id < 0 or tokenizer.eos_id < 0:
            raise ValueError("SFT requires a tokenizer with padding and end-of-sequence tokens")
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_length = context_length
        self._encoded: list[tuple[list[int], list[int]]] = []
        for index, example in enumerate(examples):
            prompt_ids = tokenizer.encode(format_instruction_prompt(example), add_bos=True)
            response_ids = tokenizer.encode(example.response, add_eos=True)
            if not prompt_ids or len(response_ids) < 2:
                raise ValueError(f"Example {example.id or index} has no usable prompt/response tokens")
            total = len(prompt_ids) + len(response_ids)
            if total > context_length + 1:
                raise ValueError(
                    f"Example {example.id or index} needs {total - 1} tokens; "
                    f"context_length={context_length}. Shorten context/response or split the example; "
                    "silent truncation is disabled."
                )
            self._encoded.append((prompt_ids, response_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids, response_ids = self._encoded[index]

        full = prompt_ids + response_ids
        x = full[:-1]
        y = full[1:]
        prompt_target_positions = max(0, len(prompt_ids) - 1)
        labels = [-100] * prompt_target_positions + y[prompt_target_positions:]

        pad_id = self.tokenizer.pad_id
        pad_len = self.context_length - len(x)
        if pad_len > 0:
            x = x + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
