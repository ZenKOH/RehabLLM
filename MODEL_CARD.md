# Model Card: RehabLLM v0.2

## Model summary

RehabLLM v0.2 is a small decoder-only causal language model intended for education and research into domain-specific LLM construction for rehabilitation and rehabilitation robotics.

The default `small` configuration uses 8 Transformer blocks, 6 attention heads, 384-dimensional embeddings, an 8,000-token vocabulary and a 512-token context window. No pretrained weights are committed to this repository.

## Intended uses

- understanding causal Transformer internals;
- small-scale language-modelling experiments;
- rehabilitation-domain corpus experiments;
- tokenizer/data-mixture studies;
- curation/deduplication research;
- later research into instruction tuning and retrieval augmentation.

## Out-of-scope uses

- clinical diagnosis or treatment;
- patient-specific recommendations;
- autonomous device selection/control;
- safety-critical decisions;
- substitution for licensed rehabilitation professionals;
- deployment as a medical device without appropriate validation/regulatory work.

## Training-data pipeline

v0.2 includes tooling to discover PMC Open Access material using official NCBI/PMC interfaces, verify article-level rights using PMC OAI-PMH `dc:rights`, retrieve full text through BioC, apply quality filters, remove exact and near duplicates, and create document-level train/validation/test splits.

The default rights allow-list is CC0, CC BY and CC BY-SA. Users remain responsible for confirming source licences and downstream obligations. Code licensing does not relicense training data.

## Known limitations

At 10–20M parameters, a from-scratch model will be dramatically less capable than contemporary billion-parameter models. Expected limitations include weak reasoning, hallucination, outdated knowledge, memorisation risk, poor instruction following before post-training, and unreliable medical factuality.

Quality and near-duplicate filtering are heuristic. The v0.2 MinHash/LSH implementation is designed for inspectability and moderate corpus sizes, not billion-document throughput.

## Evaluation

Evaluation covers held-out loss/perplexity plus domain, uncertainty and safety prompts. Generated benchmark responses are stored for human review. Clinical validation has not been performed.

## Safety

Do not use this model to make health decisions. Any future patient-facing application would require authoritative retrieval, provenance/citations, clinical evaluation, safety controls, privacy review and applicable regulatory assessment.
