import pytest
import torch

from rehab_minillm.instruction import (
    InstructionDataset,
    InstructionExample,
    format_instruction_prompt,
    validate_instruction_splits,
)


class FakeTokenizer:
    pad_id = 0
    eos_id = 3

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [10 + i for i, _ in enumerate(text.split())]
        if add_bos:
            ids.insert(0, 2)
        if add_eos:
            ids.append(3)
        return ids


def test_instruction_prompt_has_explicit_response_boundary():
    example = InstructionExample(
        instruction="Explain rehabilitation robotics.",
        context="A short context.",
        response="A short answer.",
    )
    prompt = format_instruction_prompt(example)
    assert "### Instruction:" in prompt
    assert "### Context:" in prompt
    assert prompt.endswith("### Response:\n")


def test_instruction_dataset_masks_prompt_tokens():
    example = InstructionExample(
        instruction="Explain rehabilitation robotics.",
        response="Robots can support structured movement practice.",
    )
    dataset = InstructionDataset([example], FakeTokenizer(), context_length=32)
    x, y = dataset[0]
    assert x.shape == y.shape == torch.Size([32])
    assert (y == -100).any()
    assert (y != -100).any()
    first_supervised = int((y != -100).nonzero()[0])
    assert first_supervised > 0


@pytest.mark.parametrize("field", ["instruction", "response", "context"])
def test_sft_rejects_overflow_instead_of_silently_cutting_answer(field):
    values = {"instruction": "Explain.", "response": "A complete answer.", "context": ""}
    values[field] = "long " * 100
    with pytest.raises(ValueError, match="silent truncation is disabled"):
        InstructionDataset([InstructionExample(**values)], FakeTokenizer(), context_length=32)


def test_split_overlap_catches_alternate_answers_and_whitespace():
    train = [InstructionExample("Explain ROBOTICS", "First answer.")]
    val = [InstructionExample(" explain  robotics ", "Different answer.")]
    with pytest.raises(ValueError, match="leakage"):
        validate_instruction_splits(train, val)


def test_response_mask_includes_full_response_and_eos_at_exact_context_limit():
    example = InstructionExample("Explain.", "A complete answer.")
    tok = FakeTokenizer()
    prompt = tok.encode(format_instruction_prompt(example), add_bos=True)
    response = tok.encode(example.response, add_eos=True)
    dataset = InstructionDataset([example], tok, context_length=len(prompt) + len(response) - 1)
    x, y = dataset[0]
    assert y[y != -100].tolist() == response
    assert x[:len(prompt)].tolist() == prompt
