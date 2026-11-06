"""Core contracts with no third-party dependencies: conditions and arms (protocol), samples, requests,
generations and output rows (schema), run identity and meta.json (identity), and output-file primitives (io).
Every model environment imports these, so they must stay importable on Python 3.10-3.12 with the standard library
only."""
from .protocol import (FIXED_REFUSAL, GAPS, PLUGIN_CONTRASTS, VARIANTS, Arm, Condition, Mode, Role, condition_arm,
                       conditions_for)
from .schema import (FINISH, GEN_KEYS, KINDS, SCORE_KEYS, SCORE_ROW_KEYS, Generation, Request, Sample, SampleView,
                     generation_row, score_row, seed_of, utc_now)

__all__ = ["FIXED_REFUSAL", "GAPS", "PLUGIN_CONTRASTS", "VARIANTS", "Arm", "Condition", "Mode", "Role",
           "condition_arm", "conditions_for", "FINISH", "GEN_KEYS", "KINDS", "SCORE_KEYS", "SCORE_ROW_KEYS",
           "Generation", "Request", "Sample", "SampleView", "generation_row", "score_row", "seed_of", "utc_now"]
