import torch

from rehab_minillm.instruction import (
    InstructionDataset,
    InstructionExample,
    format_instruction_prompt,
)


class FakeTokenizer:
    pad_id = 0

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
