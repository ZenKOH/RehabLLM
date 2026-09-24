# Free GPU notebook: Kaggle + Google Colab

`notebooks/RehabLLM_Train_Free_GPU.ipynb` is the single supported notebook for free-GPU training.

## What it does

The notebook detects whether it is running in Google Colab or Kaggle, verifies that CUDA is available, installs RehabLLM, prepares or restores the rehabilitation corpus and tokenizer, trains the 17.4M-parameter model, resumes from the latest checkpoint when present, evaluates the final checkpoint, and packages the artefacts.

## Google Colab

1. Open the notebook from GitHub.
2. Select a GPU runtime.
3. Run the notebook from top to bottom.
4. When prompted, mount Google Drive.

By default, persistent files are stored under `MyDrive/RehabLLM-FreeGPU`. Model checkpoints therefore survive a Colab runtime restart.

## Kaggle

1. Import the notebook from GitHub or upload it to Kaggle.
2. Enable a GPU accelerator.
3. Run all cells.

Kaggle uses `/kaggle/working/RehabLLM-FreeGPU` for outputs. Kaggle working files persist when you save a notebook version. To resume in a new notebook session, restore the previous output directory (or attach it as an input and copy it back) before the training cell.

## Default experiment

- model: 17,437,440 parameters
- context length: 512
- micro-batch: 16
- gradient accumulation: 4
- optimiser steps: 10,000
- effective token exposure: 327,680,000 tokens
- checkpoint interval: 500 steps
- corpus target: 12,000 rehabilitation-domain documents

The configuration lives in `configs/free_gpu.yaml`.

## Resume behaviour

Training supports:

```bash
python scripts/train_model.py ... --resume latest
```

If `step_*.pt` files are present in the output directory, the trainer restores the newest checkpoint's model weights, optimiser state and AMP scaler state, then continues from the saved step. If no checkpoint exists, `--resume latest` starts normally from step 0.

## Outputs

The notebook produces:

- final model checkpoint (`final.pt`)
- intermediate `step_*.pt` checkpoints
- `run_manifest.json`
- SentencePiece tokenizer
- held-out evaluation JSON
- corpus statistics
- a ZIP archive containing the key artefacts

The model remains a research model and is not clinically validated.
