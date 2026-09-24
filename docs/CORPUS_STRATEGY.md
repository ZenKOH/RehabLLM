# Corpus strategy

## Objective

The aim is not to maximise the number of rehabilitation papers. It is to create a corpus that gives a small model enough general scientific language to learn coherent generation while disproportionately representing rehabilitation, rehabilitation robotics and adjacent neurotechnology.

## Default source: PMC Open Access Subset

PMC explicitly distinguishes between material that is free to access and material that is licensed for reuse. Automated retrieval must use approved services such as PMC Cloud, OAI-PMH, E-Utilities or BioC. RehabMiniLLM uses E-Utilities for discovery and BioC for retrieval.

The default allow-list is deliberately narrower than PMC's commercial-use grouping:

| Licence | Default | Rationale |
|---|---:|---|
| CC0 | Yes | Public-domain dedication |
| CC BY | Yes | Permits reuse with attribution |
| CC BY-SA | Yes | Permits reuse; share-alike obligations need tracking |
| CC BY-ND | No | Conservative exclusion because model training/transformation may raise derivative-work questions |
| CC BY-NC | No | Excluded to avoid locking the baseline to non-commercial use |
| CC BY-NC-SA | No | Same non-commercial concern plus share-alike |
| CC BY-NC-ND | No | Most restrictive CC combination |
| Custom/unknown | No | Requires manual rights review |

## Topic coverage

The query plan covers:

- general rehabilitation and neurorehabilitation
- upper-limb and hand robotics
- gait and lower-limb robotics
- spinal cord injury/exoskeletons
- stroke rehabilitation
- FES and neuromodulation-adjacent rehabilitation
- BCI/neurotechnology
- prosthetics, orthotics and assistive technology
- therapy professions and rehabilitation outcomes

## Mixture recommendation

For a useful 10–20M-parameter experiment, target at least tens of millions of tokens. A domain-only corpus can be too narrow, causing memorisation and weak general language. A practical later mix is approximately:

- 40% broad rehabilitation/clinical science
- 30% rehabilitation robotics + neurotechnology
- 20% broader permissively licensed biomedical/scientific text
- 10% methods, implementation, measurement and health-systems material

Those percentages are starting hypotheses, not fixed truths. Measure domain validation loss, general-language degradation and memorisation.

## Cleaning

Before serious training, add:

1. duplicate and near-duplicate detection;
2. reference/bibliography removal;
3. boilerplate removal;
4. retraction/expression-of-concern exclusion;
5. language detection;
6. minimum quality/length thresholds;
7. source-level train/validation/test splits to reduce leakage;
8. PHI/PII review for any non-public dataset.

## WHO and guideline content

WHO rehabilitation materials are highly valuable for evaluation and RAG, but some WHO publications use CC BY-NC-SA licences. They are therefore **not** automatically mixed into the default corpus. NICE recommendations are used as evaluation/reference material rather than assumed training text.

## Important source references

- PMC Open Access Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- PMC User Guide/licence filters: https://pmc.ncbi.nlm.nih.gov/about/userguide/
- PMC BioC API: https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/
- NCBI E-Utilities: https://www.ncbi.nlm.nih.gov/books/NBK25499/
- PMC developer rules: https://pmc.ncbi.nlm.nih.gov/tools/developers/
