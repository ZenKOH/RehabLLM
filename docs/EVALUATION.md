# Evaluation plan

Perplexity is necessary but insufficient. The model should be evaluated across five layers.

## 1. Language-model metrics

- training loss
- held-out validation loss
- perplexity
- overfitting gap

## 2. Domain completion

Use held-out rehabilitation text to test terminology and concept completion. Splits should be by source/article, not random token windows from the same article.

## 3. Concept probes

Probe whether the model distinguishes:

- impairment vs activity vs participation;
- exoskeleton vs end-effector architectures;
- assistance vs resistance vs assessment modes;
- upper- vs lower-limb rehabilitation;
- FES vs mechanical robotic assistance;
- clinical outcome vs engineering metric;
- efficacy vs implementation/adoption.

## 4. Evidence calibration

Prompts should contain topics where evidence is mixed. A good answer should state scope and uncertainty rather than produce a universal conclusion. `data/eval_prompts.jsonl` includes initial cases.

## 5. Medical-safety behaviour

Before any patient-facing experiment, evaluate requests that ask for personalised device selection, exercise dose, stimulation settings or prognosis. A research model should not be treated as safe simply because it produces fluent cautious language.

## Benchmark versioning

Store each benchmark item with:

- stable ID
- domain
- prompt
- expected behaviour
- source/guideline date where applicable
- scoring rubric

As the project advances, use blinded human review by rehabilitation clinicians/engineers alongside automated checks.

## Reproducible checkpoint comparison (v0.4.1)

Use `--prompt-format instruction` for instruction-tuned checkpoints. For a paired behavioural experiment, use that same wrapper for the baseline. `continuation` remains the default for reproducing earlier raw-completion experiments. Do not compare reports with different wrappers as if only the weights changed.

When a benchmark item supplies context, instruction mode uses the SFT context section; continuation mode prefixes that context to the raw prompt. Expected-behaviour annotations are never included in model input.

```bash
python scripts/evaluate_model.py \
  --checkpoint checkpoints/v04_sft/final.pt \
  --tokenizer data/processed/rehab_sp.model \
  --test data/processed/test.bin \
  --prompt-format instruction --seed 42 --out eval/candidate.json
```

Run the baseline with identical arguments except checkpoint and output, then:

```bash
python scripts/compare_evaluations.py \
  --baseline eval/baseline.json --candidate eval/candidate.json \
  --out eval/comparison.json
```

Schema-v2 reports record the input hashes, decoding settings, completion-only outputs and EOS/context/budget stopping reasons. Comparison rejects mismatched tokenizer/test/prompt hashes or evaluation settings. Review incomplete generations as such. Sampled test loss uses a fixed local RNG without disturbing training; reproducibility still depends on hardware, library versions and numerical execution. It is a sampled language-model loss, not a full test-set sweep or clinical score.

The benchmark now has 14 prompts, including missing sources, conflicting synthetic findings, assisted performance versus retention, utilisation versus benefit, source-text instruction injection and individual stimulation requests. Synthetic contexts are labelled. They are behavioural probes, not a representative clinical benchmark, and their expected behaviours require human assessment.

SFT validation weights loss by unmasked response-token count. It measures a different objective from held-out full-text perplexity; do not compare the two as interchangeable scores. The paired report is pending human review until assessed; it does not automatically declare the candidate better.
