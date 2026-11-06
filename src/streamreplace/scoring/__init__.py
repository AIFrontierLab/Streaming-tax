"""Scoring of generation rows: score.py writes `<job>.scores.jsonl` (rule layer, then cached judge answers);
outcome.py holds the counting rules every metric uses (final correctness, errors, silence)."""
from .outcome import answered, counted, final, outcome, safe_on_error, silent
from .score import CacheMiss, JobFile, Report, find_jobs, rule_score, score

__all__ = ["answered", "counted", "final", "outcome", "safe_on_error", "silent", "CacheMiss", "JobFile", "Report",
           "find_jobs", "rule_score", "score"]
