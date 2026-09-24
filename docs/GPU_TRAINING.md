# Substantial GPU training

The v0.3 target is a real from-scratch training run of the 17.4M-parameter RehabLLM on a rehabilitation-focused subset of the filtered/deduplicated Common Pile PubMed corpus.

## Default run

- Hardware target: 1× NVIDIA L4 (24 GB VRAM)
- Model: 17,437,440 parameters
- Context: 512 tokens
- Micro-batch: 24 sequences
- Gradient accumulation: 3
- Optimisation steps: 10,000
- Effective token exposure: 24 × 3 × 512 × 10,000 = 368,640,000 tokens
- Corpus target: 12,000 rehabilitation-domain PMC documents
- Checkpoint interval: 1,000 steps
- Validation interval: 250 steps
- Final evaluation: held-out loss/perplexity plus rehabilitation prompts

The corpus source is `common-pile/pubmed_filtered`, whose dataset card describes it as a filtered and deduplicated PubMed Central corpus with per-document licence metadata. RehabLLM adds a conservative CC0/CC BY/CC BY-SA check and rehabilitation-domain selection.

## Hugging Face Jobs launch

A Hugging Face account with positive compute credit can launch:

```bash
hf jobs run \
  --name rehabllm-17m-v03 \
  --flavor l4x1 \
  --timeout 12h \
  --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -- bash -lc 'apt-get update -qq && apt-get install -y -qq git && \
    git clone https://github.com/ZenKOH/RehabLLM.git && \
    cd RehabLLM && \
    pip install -e ".[cloud]" && \
    bash scripts/run_hf_gpu_training.sh'
```

The launcher refuses to proceed without CUDA. At completion it creates a **private** Hugging Face model repository under the authenticated user's namespace and uploads the final checkpoint, run manifest, tokenizer, evaluation output and corpus statistics.

## Why L4 first

The 17M model is too small to justify an A100 as the default economic choice. L4 has 24 GB VRAM and is sufficient for this architecture while preserving a realistic GPU training workflow. Once the 17M run establishes loss curves, throughput and data quality, the next scale experiment should move to a 50–100M parameter configuration and reconsider A100/L40S hardware.

## Cost control

Hugging Face Jobs bills while a job is Starting or Running. The launch command caps the run at 12 hours. Actual cost depends on runtime and the current hardware price.

## Outputs

The GPU job uploads only model/training artefacts, not the raw/full-text corpus:

- `final.pt`
- `run_manifest.json`
- SentencePiece `.model` and `.vocab`
- `l4_v0.3.json`
- `gpu_corpus_stats.json`
- model/training config
- model card

The output repo is private by default.
