#!/usr/bin/env bash
set -euo pipefail

TARGET_DOCS="${TARGET_DOCS:-12000}"
OUT_DIR="${OUT_DIR:-checkpoints/l4-v0.3}"

echo "=== RehabLLM v0.3 substantial GPU run ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required for this training run.")
print("gpu", torch.cuda.get_device_name(0))
PY

python scripts/build_hf_rehab_corpus.py \
  --target-docs "${TARGET_DOCS}" \
  --out-dir data/curated

python scripts/train_tokenizer.py \
  --input data/curated/train.txt \
  --prefix data/processed/rehab_sp \
  --vocab-size 8000

python scripts/prepare_data.py \
  --train-text data/curated/train.txt \
  --val-text data/curated/val.txt \
  --test-text data/curated/test.txt \
  --tokenizer data/processed/rehab_sp.model

python scripts/train_model.py \
  --config configs/gpu_l4.yaml \
  --train data/processed/train.bin \
  --val data/processed/val.bin \
  --out "${OUT_DIR}"

python scripts/evaluate_model.py \
  --checkpoint "${OUT_DIR}/final.pt" \
  --tokenizer data/processed/rehab_sp.model \
  --test data/processed/test.bin \
  --out eval/l4_v0.3.json \
  --batch-size 16 \
  --eval-batches 50

python scripts/upload_training_artifacts.py \
  --checkpoint-dir "${OUT_DIR}" \
  --tokenizer-prefix data/processed/rehab_sp \
  --eval eval/l4_v0.3.json \
  --stats data/curated/gpu_corpus_stats.json

echo "=== RehabLLM v0.3 job complete ==="
