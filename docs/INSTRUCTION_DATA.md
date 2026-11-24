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

## v0.4.1 checks

The complete prompt, context, response and EOS must fit in `context_length + 1` tokens for next-token training. Oversized examples fail CPU preflight with their ID; responses are never silently truncated. Run `scripts/audit_instructions.py` with the original tokenizer before allocating a GPU.

Preparation preserves any supplied provenance fields and keeps alternate answers to the same normalised instruction/context in one split. The trainer also rejects overlapping prompt groups in manually prepared splits. This does not detect paraphrases or shared-source leakage; curate those groups before splitting.
