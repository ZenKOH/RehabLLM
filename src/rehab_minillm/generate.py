from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import RehabMiniLLM
from .tokenizer import SentencePieceTokenizer


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> RehabMiniLLM:
    payload = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**payload["model_config"])
    model = RehabMiniLLM(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with RehabMiniLLM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    model = load_checkpoint(args.checkpoint, device)

    ids = tokenizer.encode(args.prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
