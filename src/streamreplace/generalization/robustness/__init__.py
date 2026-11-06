"""Prompt robustness: do streaming children lose more accuracy than their offline parents when the wording of a
question (or of the answer instruction) changes? (Table 2's prompt-robustness columns, Supp. Tables eval_robustness
and eval_instruction; docs/generalization/robustness.md.)

The question stem of each item is perturbed with typos, a paraphrase or an appended distractor sentence (PromptRobust,
Zhu et al. 2023), the offline parent and child are run on the perturbed prompts, and the accuracy drops are compared:
dRob = Drop_parent - Drop_child, + = the child loses less. Every condition is an ordinary outputs folder holding a
subset.json and a frozen prompt_overrides.jsonl of exact [old, new] edits, which the runner applies through its
generic prompt-override hook (data/prompts.py), so the runner knows nothing about perturbations.

Modules:
  perturb      pure text functions: typos, the StressTest sentence, irrelevant sentences, paraphrase rule checks
  make         the specification, analysis units, item selection and the condition folders' subset.json
  edits        the frozen edits: generation, prompt_overrides.jsonl, checks, audit pages
  paraphrase   back-translated paraphrases from any OpenAI-compatible endpoint (not exactly regenerable)
  instruction  instruction-level edits and their analysis
  report       unit outcomes, Drop, PDR, dRob with the paper's statistics, the report files
Command line: `streamreplace robustness subset|paraphrase|edits|check|audit|report` (cli/cmd_robustness.py)."""
