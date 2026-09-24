from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids.insert(0, self.bos_id)
        if add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode(list(ids))


def train_sentencepiece(
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> None:
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
        byte_fallback=False,
        normalization_rule_name="nmt_nfkc",
    )
