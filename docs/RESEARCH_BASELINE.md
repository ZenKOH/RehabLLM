# Research baseline — September 2026

This note records the design implications of the initial literature/web review. It is not a systematic review.

## Rehabilitation scope

WHO defines rehabilitation around optimising functioning and reducing disability in interaction with a person's health condition and environment. WHO estimates that about 2.4 billion people live with a condition that may benefit from rehabilitation, and its Package of Interventions spans multiple disease areas rather than treating rehabilitation as a robotics-only discipline.

**Design implication:** the model corpus must include functioning, activity, participation, therapy, assistive products, service delivery and outcomes — not only device engineering.

## Rehabilitation robotics evidence is heterogeneous

Recent systematic reviews of stroke rehabilitation robotics report benefits in some outcomes, but effect sizes, persistence of benefit and results by device/programme vary. A 2025 upper-limb review reported a small positive effect on upper-limb capacity in dose-matched comparisons that was not maintained at follow-up. Other reviews report different subgroup effects.

NICE's current adult stroke-rehabilitation recommendations say not to offer robot-assisted arm training as part of an upper-limb rehabilitation programme.

**Design implication:** evaluation must reward evidence calibration and recognition of guideline conflicts. The model should not learn the simplistic statement “robots improve rehabilitation”.

## SCI exoskeleton evidence is promising but not settled

Recent reviews report improvements in some walking/motor outcomes after exoskeleton training in spinal cord injury, while heterogeneity and study limitations remain.

**Design implication:** train/evaluate language that separates potential benefit, measured outcome, population, comparator and evidence quality.

## Data rights are a first-class ML problem

PMC states that not every article in PMC is available for text mining/reuse. Its Open Access Subset has article-specific licences and automated retrieval must use approved services. Current PMC search filters expose CC0, CC BY, CC BY-SA and other licence classes.

**Design implication:** the corpus builder starts from rights metadata and provenance, not from indiscriminate scraping.

## Technical choices

SentencePiece supports BPE and unigram tokenisation and documents an 8,000-token vocabulary default. PyTorch's current mixed-precision guidance uses `torch.autocast` with `torch.amp.GradScaler`.

**Design implication:** v0.1 uses an 8k SentencePiece vocabulary and AMP-capable training, while keeping attention code explicit for educational transparency.

## Sources

- WHO Rehabilitation: https://www.who.int/news-room/fact-sheets/detail/rehabilitation
- WHO Package of Interventions: https://www.who.int/teams/noncommunicable-diseases/sensory-functions-disability-and-rehabilitation/rehabilitation/service-delivery/package-of-interventions-for-rehabilitation
- NICE Stroke Rehabilitation: https://www.nice.org.uk/guidance/ng236/chapter/Recommendations
- PMC OA Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- PMC licence filters: https://pmc.ncbi.nlm.nih.gov/about/userguide/
- PMC BioC API: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
- SentencePiece: https://github.com/google/sentencepiece
