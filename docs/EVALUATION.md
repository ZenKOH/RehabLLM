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
