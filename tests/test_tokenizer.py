from pathlib import Path

from rehab_minillm.tokenizer import SentencePieceTokenizer, train_sentencepiece


def test_sentencepiece_roundtrip(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("Rehabilitation robotics supports repetitive task practice and measurement.\n" * 50)
        + ("Stroke rehabilitation may include gait, balance, arm and hand training.\n" * 50),
        encoding="utf-8",
    )
    prefix = tmp_path / "rehab_test"
    train_sentencepiece(corpus, prefix, vocab_size=64, model_type="bpe")
    tok = SentencePieceTokenizer(str(prefix) + ".model")
    ids = tok.encode("rehabilitation robotics", add_bos=True, add_eos=True)
    text = tok.decode(ids)
    assert tok.vocab_size > 10
    assert "rehabilitation" in text.lower()
