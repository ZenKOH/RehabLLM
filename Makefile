.PHONY: install test fetch curate tokenizer prepare train-tiny evaluate

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
