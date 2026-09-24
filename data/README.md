# Data

This directory intentionally does not ship copyrighted training text or trained model weights.

## v0.2 pipeline

1. `scripts/fetch_pmc.py` writes `raw/pmc_rehab.jsonl` with article text, topic provenance and article-level OAI-PMH rights metadata.
2. `scripts/curate_corpus.py` filters and deduplicates records, then writes deterministic document-level splits under `curated/`.
3. `scripts/train_tokenizer.py` trains SentencePiece on `curated/train.txt` only.
4. `scripts/prepare_data.py` independently tokenises train, validation and test text into `processed/*.bin`.

## Default rights policy

The default fetch pipeline accepts only article-level rights that map to:

- CC0
- CC BY
- CC BY-SA

CC BY-NC, CC BY-ND, unknown and custom rights are excluded by default. This is a conservative project policy, not legal advice.

## Locally generated files

- `raw/pmc_rehab.jsonl` — retrieved article records and provenance
- `curated/articles.jsonl` — accepted records with quality metrics, split and hashes
- `curated/rejected.jsonl` — rejected records with reasons
- `curated/stats.json` — corpus statistics
- `curated/manifest.json` — curation config plus input hash
- `curated/train.txt`, `val.txt`, `test.txt` — document-level text splits
- `processed/rehab_sp.model` / `.vocab` — tokenizer
- `processed/train.bin`, `val.bin`, `test.bin` — int32 token streams

These generated outputs are ignored by Git except `.gitkeep` placeholders.

## Provenance rule

Any future data source should record at least:

- source name
- stable source URL/identifier
- retrieval date
- licence/rights status
- inclusion rationale
- processing steps
- split assignment
- content hash

Do not infer reuse rights from “free to read”.
