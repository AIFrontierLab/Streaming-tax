"""Generalization experiments of the paper, beyond the P/C/S evaluation:

- robustness/: prompt robustness. The question stem (or the answer instruction) is perturbed with typos,
  paraphrases or distractor sentences and the offline pair members are re-evaluated on the perturbed prompts.
- finetune/: LoRA fine-tuning. Every pair member is fine-tuned on the same nested training subsets of a new
  domain (LingoQA driving, EyePCR ophthalmic surgery; OpenUAV-QA optional) and the parent-child gap is compared before
  and after.

The core packages never import this package; it builds on the public interfaces (benchmarks, adapters with the
optional Trainable capability, the run and scoring pipeline)."""
