# RehabLLM

**A small decoder-only language model for rehabilitation and rehabilitation robotics, built from scratch in PyTorch.**

RehabLLM is an educational/research project: a compact GPT-style model whose core attention, Transformer blocks, training loop and generation code are implemented directly rather than imported as a ready-made GPT architecture.

> **Status: v0.2 corpus-quality phase.** The Transformer is working; v0.2 adds article-level rights verification, quality filtering, near-deduplication, document-level train/validation/test splits, reproducible corpus manifests and checkpoint evaluation.

## Why rehabilitation?

Rehabilitation is a broad health strategy focused on optimising functioning and reducing disability. The domain spans neurological, musculoskeletal, cardiopulmonary, developmental and other conditions, while rehabilitation robotics adds exoskeletons, end-effector robots, powered gait systems, upper-limb/hand devices, FES, sensing, BCIs and related technologies.

This project is intentionally designed **not** to assume that more technology is automatically better rehabilitation. Evidence and guidelines can conflict by population, device, outcome, dose and comparator. Evaluation therefore includes uncertainty, clinical-context and guideline-conflict checks.

## Model architecture

The default `small.yaml` configuration is a 17,437,440-parameter model:

- 8,000-token SentencePiece vocabulary
- 512-token context
- 8 Transformer layers
- 6 attention heads
- 384-dimensional embeddings
- 1,536-dimensional feed-forward layer
- decoder-only causal Transformer
- tied token/output embeddings
- pre-LayerNorm residual blocks
- AdamW + warm-up + cosine decay
- mixed-precision training on CUDA

The `tiny.yaml` configuration has 5,263,872 parameters and is intended for CPU/smoke testing.

## v0.2 data pipeline

```text
PMC discovery
  ↓
OAI-PMH article-level rights verification
  ↓
BioC full-text retrieval
  ↓
quality filters
  ↓
exact SHA-256 deduplication
  ↓
5-word-shingle MinHash/LSH near-deduplication
  ↓
document-level deterministic train/val/test split
  ↓
SentencePiece trained on train only
  ↓
train.bin / val.bin / test.bin
  ↓
training run manifest + held-out evaluation
```

See `docs/V0.2_DATA_PIPELINE.md` for the design rationale.

## Repository layout

```text
.
├── configs/
│   ├── curation.yaml
│   ├── tiny.yaml
│   └── small.yaml
├── data/
│   ├── eval_prompts.jsonl
│   ├── raw/
│   ├── curated/
│   └── processed/
├── docs/
│   ├── CORPUS_STRATEGY.md
│   ├── EVALUATION.md
│   ├── RESEARCH_BASELINE.md
│   ├── SAFETY.md
│   └── V0.2_DATA_PIPELINE.md
├── scripts/
│   ├── curate_corpus.py
│   ├── evaluate_model.py
│   ├── fetch_pmc.py
│   ├── generate_text.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── train_tokenizer.py
├── src/rehab_minillm/
│   ├── attention.py
│   ├── config.py
│   ├── corpus.py
│   ├── curation.py
│   ├── data.py
│   ├── evaluate.py
│   ├── experiment.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer.py
├── tests/
├── MODEL_CARD.md
└── pyproject.toml
```

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Fetch PMC Open Access material

Set a contact email as requested by NCBI E-Utilities:

```bash
export NCBI_EMAIL="you@example.com"
python scripts/fetch_pmc.py --retmax-per-query 250 --max-articles 2500
```

The fetcher uses NCBI E-Utilities for discovery, PMC OAI-PMH `dc:rights` metadata for article-level rights verification and PMC BioC for full text. The default allow-list is CC0, CC BY and CC BY-SA. It also excludes retraction notices, retracted articles, expressions of concern and correction notices from discovery.

### 3. Curate, deduplicate and split

```bash
python scripts/curate_corpus.py \
  --input data/raw/pmc_rehab.jsonl \
  --config configs/curation.yaml \
  --out-dir data/curated
```

Outputs include accepted/rejected JSONL records, corpus statistics, a build manifest and document-level `train.txt`, `val.txt` and `test.txt` files.

### 4. Train the tokenizer on training text only

```bash
python scripts/train_tokenizer.py \
  --input data/curated/train.txt \
  --prefix data/processed/rehab_sp \
  --vocab-size 8000
```

### 5. Tokenise train, validation and test sets

```bash
python scripts/prepare_data.py \
  --train-text data/curated/train.txt \
  --val-text data/curated/val.txt \
  --test-text data/curated/test.txt \
  --tokenizer data/processed/rehab_sp.model
```

### 6. Smoke-test training

```bash
python scripts/train_model.py \
  --config configs/tiny.yaml \
  --train data/processed/train.bin \
  --val data/processed/val.bin \
  --out checkpoints/tiny
```

Each run writes `run_manifest.json`, including model/training configuration, source Git commit, environment metadata and SHA-256 hashes of the training/validation token files.

### 7. Train the ~17M model

```bash
python scripts/train_model.py \
  --config configs/small.yaml \
  --train data/processed/train.bin \
  --val data/processed/val.bin \
  --out checkpoints/small
```

### 8. Evaluate on held-out text and domain prompts

```bash
python scripts/evaluate_model.py \
  --checkpoint checkpoints/small/final.pt \
  --tokenizer data/processed/rehab_sp.model \
  --test data/processed/test.bin \
  --out eval/small_v0.2.json
```

This records held-out loss/perplexity plus generated responses for the rehabilitation benchmark prompts and leaves fields for human review.

### 9. Generate text

```bash
python scripts/generate_text.py \
  --checkpoint checkpoints/small/final.pt \
  --tokenizer data/processed/rehab_sp.model \
  --prompt "Robot-assisted rehabilitation after stroke" \
  --max-new-tokens 120
```

## Corpus design

A small language model trained from scratch needs enough language diversity to learn syntax as well as domain concepts. The long-term corpus should therefore be a **mixture**, not only papers containing the word “robot”. The proposed mix is:

1. **Core rehabilitation** — functioning, assessment, therapy, outcomes, participation, service delivery.
2. **Rehabilitation robotics** — upper/lower limb, gait, hand, exoskeleton and end-effector systems.
3. **Neurotechnology** — FES, BCI, sensing, motor learning, human-machine interaction.
4. **Clinical populations** — stroke, SCI, TBI, Parkinson disease, cerebral palsy, amputation and musculoskeletal rehabilitation.
5. **Methods/evidence** — RCTs, systematic reviews, measurement, health economics, implementation and safety.

See `docs/CORPUS_STRATEGY.md`.

## Research guardrails

The source strategy uses official PMC/NCBI retrieval interfaces and records source rights and provenance. “Free to read” is not treated as equivalent to “licensed for reuse”. The default pipeline is deliberately conservative about licensing and research-integrity flags.

Key references are listed in `docs/RESEARCH_BASELINE.md` and `docs/V0.2_DATA_PIPELINE.md`.

The code and source documents are not medical advice. See `docs/SAFETY.md` and `MODEL_CARD.md`.

## What v0.2 is — and is not

**It is:**

- a real causal Transformer trained from scratch;
- inspectable enough to learn how LLMs work;
- domain-oriented and licence/provenance-aware;
- equipped with reproducible curation and evaluation scaffolding;
- suitable for experimentation, ablations and later domain adaptation.

**It is not:**

- a clinically validated model;
- a diagnostic or treatment system;
- expected to match modern billion-parameter models;
- ready for patient-facing deployment.

## Roadmap

- **v0.1** — core Transformer, tokenizer, corpus builder, training, generation, tests. ✅
- **v0.2** — rights verification, data cleaning/deduplication, document splits, experiment tracking and evaluation. ✅
- **v0.3** — build the first substantial mixed corpus and train/compare 17M and 50–100M models.
- **v0.4** — supervised instruction tuning for rehabilitation Q&A.
- **v0.5** — retrieval-augmented generation over curated guidelines/papers with citations.
- **v0.6** — tool use for literature search, calculations and evidence retrieval.

## Large files

Do not commit raw corpora or checkpoints to ordinary Git history. Keep GitHub for source, configs, tests and documentation. Store larger datasets/model artefacts in dedicated model/data storage such as Hugging Face Hub or object storage.

## Licence

Code: Apache-2.0. Training data retain their own licences; the code licence does **not** override source-data terms.
