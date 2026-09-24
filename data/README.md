# Data

This directory intentionally does not ship copyrighted training text.

## Default corpus policy

`fetch_pmc.py` searches the PMC Open Access Subset through NCBI E-Utilities and retrieves full text through PMC's BioC API. The default query plan uses only:

- CC0
- CC BY
- CC BY-SA

The code excludes CC BY-NC and CC BY-ND from the default training corpus to keep the initial corpus comparatively permissive. This is a conservative project policy, not legal advice.

## Files created locally

- `raw/pmc_rehab.jsonl` — article text plus source/provenance fields
- `raw/pmc_rehab.txt` — plain text used to train SentencePiece/model
- `processed/rehab_sp.model` — tokenizer
- `processed/rehab_sp.vocab` — tokenizer vocabulary
- `processed/train.bin` — int32 training tokens
- `processed/val.bin` — int32 validation tokens

These outputs are ignored by Git.

## Provenance rule

Any future data source should record at least:

- source name
- stable source URL/identifier
- retrieval date
- licence/rights status
- inclusion rationale
- processing steps

Do not infer reuse rights from “free to read”.
