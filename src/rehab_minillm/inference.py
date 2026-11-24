"""Controlled single-prompt generation shared by the chat CLI and evaluator."""
from __future__ import annotations

import torch

from .instruction import InstructionExample, format_instruction_prompt
from .model import RehabMiniLLM
from .tokenizer import SentencePieceTokenizer


def generate_response(
    model: RehabMiniLLM,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    *,
    context: str = "",
    prompt_format: str = "instruction",
    max_new_tokens: int = 160,
    seed: int = 42,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
) -> dict:
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the checkpoint")
    if prompt_format == "instruction":
        formatted = format_instruction_prompt(InstructionExample(prompt, "", context))
    elif prompt_format == "continuation":
        formatted = f"{context}\n\n{prompt}" if context else prompt
    else:
        raise ValueError("prompt_format must be instruction or continuation")
    ids = tokenizer.encode(formatted, add_bos=True)
    if not ids or len(ids) > model.config.context_length:
        raise ValueError(
            f"Prompt needs {len(ids)} tokens; context_length={model.config.context_length}. "
            "Shorten the prompt/context; silent prompt truncation is disabled."
        )
    # Keep the whole prompt visible during every generation step.
    budget = min(max_new_tokens, model.config.context_length - len(ids) + 1)
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        out = model.generate(
            x, max_new_tokens=budget, temperature=temperature, top_k=top_k, top_p=top_p,
            eos_id=tokenizer.eos_id if tokenizer.eos_id >= 0 else None, generator=rng,
        )
    finally:
        model.train(was_training)
    generated = out[0, len(ids):].tolist()
    ended = bool(generated and generated[-1] == tokenizer.eos_id)
    return {
        "output": tokenizer.decode(generated[:-1] if ended else generated),
        "prompt_tokens": len(ids),
        "generated_tokens": len(generated),
        "stop_reason": "eos" if ended else "context_limit" if budget < max_new_tokens
        else "max_new_tokens",
    }
