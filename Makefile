.PHONY: install test fetch tokenizer prepare train-tiny

install:
	pip install -e ".[dev]"

test:
	pytest -q

fetch:
	python scripts/fetch_pmc.py

tokenizer:
	python scripts/train_tokenizer.py

prepare:
	python scripts/prepare_data.py

train-tiny:
	python scripts/train_model.py --config configs/tiny.yaml
