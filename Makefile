.PHONY: install test fetch curate tokenizer prepare train-tiny evaluate v04-autopsy v04-robotics v04-build v04-instructions v04-readiness

install:
	pip install -e ".[dev]"

test:
	pytest -q

fetch:
	python scripts/fetch_pmc.py

curate:
	python scripts/curate_corpus.py

tokenizer:
	python scripts/train_tokenizer.py --input data/curated/train.txt

prepare:
	python scripts/prepare_data.py

train-tiny:
	python scripts/train_model.py --config configs/tiny.yaml

evaluate:
	python scripts/evaluate_model.py --checkpoint checkpoints/small/final.pt --tokenizer data/processed/rehab_sp.model

v04-autopsy:
	python scripts/autopsy_corpus.py --input data/curated/articles.jsonl

v04-robotics:
	python scripts/fetch_pmc.py --profile robotics --out data/raw/pmc_robotics_v04.jsonl

v04-build:
	python scripts/build_v04_corpus.py --input data/curated/articles.jsonl data/raw/pmc_robotics_v04.jsonl --out-dir data/v04

v04-instructions:
	python scripts/prepare_instructions.py --input data/instruction_seed.jsonl --out-dir data/v04/instructions

v04-readiness:
	python scripts/assess_v04_readiness.py --stats data/v04/v04_corpus_stats.json
