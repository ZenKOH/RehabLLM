from pathlib import Path

import pytest

from rehab_minillm.tokenizer import SentencePieceTokenizer, train_sentencepiece


@pytest.fixture(scope="session")
def real_tokenizer(tmp_path_factory):
    root: Path = tmp_path_factory.mktemp("real-tokenizer")
    corpus = root / "corpus.txt"
    corpus.write_text(
        ("### Instruction: Explain research. ### Context: A synthetic example. "
         "### Response: Complete answers require relevant evidence and human review. "
         "Compare the source dates and study limitations. Rehabilitation robotics "
         "instruction dataset training validation tokenizer sample question.\n") * 40,
        encoding="utf-8",
    )
    train_sentencepiece(corpus, root / "sp", vocab_size=128)
    return SentencePieceTokenizer(root / "sp.model")
