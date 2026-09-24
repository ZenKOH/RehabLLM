# Safety and intended-use boundaries

RehabMiniLLM is a research/education model, not a medical device and not a clinical decision-support system.

## Do not use v0.x for

- diagnosis;
- patient-specific treatment selection;
- robot/device prescription;
- stimulation parameter prescription;
- medication advice;
- emergency triage;
- autonomous clinical decisions;
- claims of clinical efficacy or safety.

## Why this matters

A small pretrained language model predicts text. It does not inherently know which statement is current, clinically valid, applicable to an individual patient or supported by high-quality evidence. Rehabilitation robotics is especially context-sensitive: results vary with diagnosis, impairment, device, programme design, dose, comparator and outcome measure.

## Evaluation principles

The model should be penalised when it:

- turns association into causation;
- turns statistical significance into clinical significance;
- recommends a device as universally superior;
- ignores contradictory guidelines/evidence;
- gives patient-specific instructions without qualified clinical assessment;
- fabricates trials, guidelines, devices or outcome measures;
- hides uncertainty.

## Later safety architecture

A future assistant version should separate:

1. base model language generation;
2. retrieval of authoritative, dated evidence;
3. citation/provenance rendering;
4. medical-safety policy layer;
5. structured evaluation and human review.
