# RehabMiniLLM

**A small decoder-only language model for rehabilitation and rehabilitation robotics, built from scratch in PyTorch.**

RehabMiniLLM is an educational/research project: a compact GPT-style model whose core attention, Transformer blocks, training loop and generation code are implemented directly rather than imported as a ready-made GPT architecture.

> **Status: v0.1 scaffold.** The code trains and generates text. A useful domain model still requires a properly licensed corpus, GPU training, evaluation and later instruction/RAG stages.

## Why rehabilitation?

Rehabilitation is a broad health strategy focused on optimising functioning and reducing disability. WHO estimates that about 2.4 billion people live with a condition that may benefit from rehabilitation. The domain spans neurological, musculoskeletal, cardiopulmonary, developmental and other conditions, while rehabilitation robotics adds exoskeletons, end-effector robots, powered gait systems, upper-limb/hand devices, FES, sensing, BCIs and related technologies.

This project is intentionally designed **not** to assume that more technology is automatically better rehabilitation. Evidence and guidelines can conflict by population, device, outcome, dose and comparator. The evaluation set therefore includes uncertainty, clinical-context and guideline-conflict checks.

## v0.1 architecture

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

## Repository layout

```text
.
├── configs/
│   ├── tiny.yaml
│   └── small.yaml
├── data/
│   ├── README.md
│   ├── eval_prompts.jsonl
│   ├── raw/
│   └── processed/
├── docs/
│   ├── CORPUS_STRATEGY.md
│   ├── EVALUATION.md
│   ├── RESEARCH_BASELINE.md
│   └── SAFETY.md
├── notebooks/
│   ├── 01_tokenisation.ipynb
│   ├── 02_attention.ipynb
│   └── 03_training.ipynb
├── scripts/
│   ├── fetch_pmc.py
│   ├── generate_text.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── train_tokenizer.py
├── src/rehab_minillm/
│   ├── attention.py
│   ├── config.py
│   ├── corpus.py
│   ├── data.py
│   ├── evaluate.py
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
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

### 2. Build a permissively licensed rehabilitation corpus from PMC

Set a contact email as requested by NCBI E-Utilities:

```bash
export NCBI_EMAIL="you@example.com"
python scripts/fetch_pmc.py --retmax-per-query 250 --max-articles 2500
```

The fetcher uses:

- NCBI E-Utilities for discovery
- PMC's BioC API for full text
- only CC0, CC BY and CC BY-SA filters by default
- a rehabilitation/robotics topic plan
- deduplication by PMCID

It deliberately excludes CC BY-NC and CC BY-ND materials from the default corpus. Always review article-level licence obligations before publishing a dataset or trained model.

### 3. Train the tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/raw/pmc_rehab.txt \
  --prefix data/processed/rehab_sp \
  --vocab-size 8000
```

### 4. Tokenise the corpus

```bash
python scripts/prepare_data.py \
  --input data/raw/pmc_rehab.txt \
  --tokenizer data/processed/rehab_sp.model
```

### 5. Smoke-test training

```bash
python scripts/train_model.py \
  --config configs/tiny.yaml \
  --train data/processed/train.bin \
  --val data/processed/val.bin \
  --out checkpoints/tiny
```

### 6. Train the ~17M model

```bash
python scripts/train_model.py \
  --config configs/small.yaml \
  --train data/processed/train.bin \
  --val data/processed/val.bin \
  --out checkpoints/small
```

### 7. Generate

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

The source strategy was informed by current authoritative material:

- WHO rehabilitation overview: https://www.who.int/news-room/fact-sheets/detail/rehabilitation
- WHO Package of Interventions for Rehabilitation: https://www.who.int/teams/noncommunicable-diseases/sensory-functions-disability-and-rehabilitation/rehabilitation/service-delivery/package-of-interventions-for-rehabilitation
- PMC Open Access Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- PMC licence filters: https://pmc.ncbi.nlm.nih.gov/about/userguide/
- PMC BioC API: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
- NCBI E-Utilities: https://www.ncbi.nlm.nih.gov/books/NBK25499/
- NICE stroke rehabilitation guideline: https://www.nice.org.uk/guidance/ng236/chapter/Recommendations

The code and source documents are not medical advice. See `docs/SAFETY.md` and `MODEL_CARD.md`.

## What v0.1 is — and is not

**It is:**

- a real causal Transformer trained from scratch;
- inspectable enough to learn how LLMs work;
- domain-oriented and licence-aware;
- suitable for experimentation, ablations and later domain adaptation.

**It is not:**

- a clinically validated model;
- a diagnostic or treatment system;
- expected to match modern billion-parameter models;
- ready for patient-facing deployment.

## Roadmap

- **v0.1** — core Transformer, tokenizer, corpus builder, training, generation, tests.
- **v0.2** — stronger data cleaning/deduplication, experiment tracking, better evaluation.
- **v0.3** — scale to 50–100M parameters with a mixed general-biomedical + rehabilitation corpus.
- **v0.4** — supervised instruction tuning for rehabilitation Q&A.
- **v0.5** — retrieval-augmented generation over curated guidelines/papers with citations.
- **v0.6** — tool use for literature search, calculations and evidence retrieval.

## Large files

Do not commit raw corpora or checkpoints to ordinary Git history. Keep GitHub for source, configs, tests and documentation. Store larger datasets/model artefacts in dedicated model/data storage such as Hugging Face Hub or object storage.

## Licence

Code: Apache-2.0. Training data retain their own licences; the code licence does **not** override source-data terms.
