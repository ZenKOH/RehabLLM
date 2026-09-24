# RehabLLM instruction-data contract

Each JSONL row must contain:

- `instruction`: required user task or question
- `response`: required desired answer
- `context`: optional source material or evidence
- `id`: optional stable identifier

The v0.4 SFT format is:

```text
### Instruction:
...

### Context:
...  # omitted when empty

### Response:
...
```

Prompt and context tokens are masked from SFT loss. Only response tokens train the answer behaviour.

## Review standard

Instruction examples should be:

- factually supportable;
- explicit about uncertainty where evidence is mixed;
- free of invented references;
- non-prescriptive for patient-specific clinical decisions;
- free of unsafe device-control or stimulation-parameter instructions;
- written as direct answers rather than journal-paper continuations.

The bundled seed set is a smoke test, not a complete instruction corpus.
