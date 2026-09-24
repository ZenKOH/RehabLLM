# Model Card: RehabMiniLLM v0.1

## Model summary

RehabMiniLLM v0.1 is a small decoder-only causal language model intended for education and research into domain-specific LLM construction for rehabilitation and rehabilitation robotics.

The default `small` configuration uses 8 Transformer blocks, 6 attention heads, 384-dimensional embeddings, an 8,000-token vocabulary and a 512-token context window.

## Intended uses

- understanding causal Transformer internals;
- small-scale language-modelling experiments;
- rehabilitation-domain corpus experiments;
- tokenizer/data-mixture studies;
- later research into instruction tuning and retrieval augmentation.

## Out-of-scope uses

- clinical diagnosis or treatment;
- patient-specific recommendations;
- autonomous device selection/control;
- safety-critical decisions;
- substitution for licensed rehabilitation professionals;
- deployment as a medical device without appropriate validation/regulatory work.

## Training data

No weights are shipped in v0.1. The repository includes tooling to create a corpus from permissively licensed PMC Open Access material using CC0, CC BY and CC BY-SA search filters and official NCBI/PMC APIs.

Users are responsible for confirming source licences and downstream obligations. Code licensing does not relicense training data.

## Known limitations

At 10–20M parameters, a from-scratch model will be dramatically less capable than contemporary billion-parameter models. Expected limitations include weak reasoning, hallucination, outdated knowledge, memorisation risk, poor instruction following before post-training, and unreliable medical factuality.

## Evaluation

Initial evaluation covers validation loss/perplexity plus domain, uncertainty and safety prompts. Clinical validation has not been performed.

## Safety

Do not use this model to make health decisions. Any future patient-facing application would require authoritative retrieval, provenance/citations, clinical evaluation, safety controls, privacy review and applicable regulatory assessment.
