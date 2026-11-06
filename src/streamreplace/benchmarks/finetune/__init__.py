"""Benchmarks of the fine-tuning experiment (generalization/finetune; docs/generalization/finetune.md): each has an
official training split (prepare_train -> prepared/train.jsonl) besides its test split.

- lingoqa     LingoQA driving video QA, scored by the official Lingo-Judge (judges/lingo.py)
- eyepcr      EyePCR ophthalmic-surgery multiple-choice questions, matched to the option texts (no official scorer)
- openuav_qa  OpenUAV-QA drone-trajectory multiple choice with the official extractor (reported, not in the primary
              endpoint)

Fine-tuning reads three optional module constants: GROUP (the metadata key whose value clusters questions: the
paired statistics resample it and the training pool drops every sample sharing a value with the test split),
STRATUM_KEYS (the strata of the <= 2K test subset and of the bootstrap) and INTERLEAVE / CHECK_NEAR_DUPLICATES (how
the training order is built)."""
