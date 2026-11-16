"""Statistics, reports and paper tables computed from scored records (numpy + PyYAML + stdlib only).

Pipeline (docs/statistics.md):
  records.py   counted records of one job (generation rows + score rows), correctness and flags
  arms.py      overlays on jobs (generation-cap reruns, plugin refusals, item restrictions) and the P/C/S arms
  specs.py     the analysis profile (configs/analysis.yaml): metrics, units, Holm families, margins
  stats.py     the multinomial cluster bootstrap behind every P/C/S contrast; McNemar, sign-flip, Holm, TOST, NI
  stats_index.py  the integer-index bootstrap used by the robustness, fine-tuning and generation-cap reports
  vmdt.py      the VMDT official-composite bootstrap
  contrasts.py all C-P / S-C / S-P contrasts with families, margins and verdicts
  metrics.py   per-job metrics and counts
  flags.py     cell qualifications (forced, invalid, non-answer, chance, artifact)
  pipeline.py  the steps behind `streamreplace report` (metrics.json, RESULTS.md), `stats` (contrasts.json, GAPS.md)
               and `tables` (the paper's tables; check_tables compares them with paper/expected/tables.json)
  render/      RESULTS.md, GAPS.md, Table 2, the MCV table, gap-figure data, absolute-score tables
The generation-cap reruns themselves (selection, rerun jobs, sensitivity report) are run/rerun_capped.py."""
