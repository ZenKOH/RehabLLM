import pytest
import torch

from rehab_minillm.config import ModelConfig
from rehab_minillm.inference import generate_response
from rehab_minillm.instruction import InstructionDataset, InstructionExample
from rehab_minillm.model import RehabMiniLLM


def _model(tokenizer, context_length=128):
    return RehabMiniLLM(ModelConfig(
        vocab_size=tokenizer.vocab_size, context_length=context_length,
        n_layers=1, n_heads=2, d_model=16, d_ff=32, dropout=0.2,
    ))


def test_real_tokenizer_padding_and_complete_sft_answer(real_tokenizer):
    tok = real_tokenizer
    example = InstructionExample("Explain research.", "Complete answers require evidence.")
    dataset = InstructionDataset([example], tok, 128)
    x, labels = dataset[0]
    answer = tok.encode(example.response, add_eos=True)
    assert tok.pad_id == 0
    assert labels[labels != -100].tolist() == answer
    assert (x == tok.pad_id).any()


def test_generation_is_repeatable_completion_only_and_restores_training(real_tokenizer):
    model = _model(real_tokenizer).train()
    before = torch.get_rng_state().clone()
    a = generate_response(model, real_tokenizer, "Explain research.", max_new_tokens=5)
    b = generate_response(model, real_tokenizer, "Explain research.", max_new_tokens=5)
    assert a == b
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert "### Instruction:" not in a["output"]


def test_controlled_inference_rejects_prompt_overflow(real_tokenizer):
    with pytest.raises(ValueError, match="silent prompt truncation"):
        generate_response(_model(real_tokenizer, 16), real_tokenizer, "Explain " * 50)


def test_controlled_inference_reports_context_budget(real_tokenizer):
    prompt = "Research"
    count = len(real_tokenizer.encode(prompt, add_bos=True))
    model = _model(real_tokenizer, count)
    # Force a non-EOS token so the budget is the cause of stopping.
    model.generate = lambda x, **kwargs: torch.cat((x, x[:, -1:]), dim=1)
    result = generate_response(model, real_tokenizer, prompt, prompt_format="continuation",
                               max_new_tokens=10)
    assert result["generated_tokens"] == 1
    assert result["stop_reason"] == "context_limit"


def test_raw_continuation_accepts_context_and_includes_it_in_token_budget(real_tokenizer):
    model = _model(real_tokenizer)
    result = generate_response(model, real_tokenizer, "Research", context="Evidence",
                               prompt_format="continuation", max_new_tokens=1)
    assert result["prompt_tokens"] == len(real_tokenizer.encode("Evidence\n\nResearch", add_bos=True))
