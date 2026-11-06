"""Shared benchmark code (ported unchanged from the paper's runs):

- prompts: the generic answer instructions (MCQ, YESNO, NUMERIC, TEXT) and the default prompt();
- parsing: clean(), final_answer(), parse_choice/parse_yesno/parse_number, equivalent(), default parse()/correct();
- metrics: pct(), by(), by_length(), group_all() and the default metrics();
- smoke: the deterministic whole-group order used for limited runs;
- io: preparation helpers and loaders for the benchmarks' official (fetched upstream) code.

Benchmark modules use them as `common.<name>`, e.g. `common.pct(...)`, `common.smoke(samples, group="video")`.
Note that the package attributes `common.metrics` and `common.smoke` are the default FUNCTIONS, not the submodules:
import submodule members with `from streamreplace.benchmarks.common.metrics import ...`."""
from .io import (extract_tar, extract_zip, letters, read_parquet, save_image, upstream_assignment, upstream_constant,
                 upstream_functions, upstream_module, upstream_path)
from .metrics import by, by_length, group_all, metrics, pct
from .parsing import (clean, correct, equivalent, final_answer, normalize, numeric_sequence, parse, parse_choice,
                      parse_number, parse_yesno, scalar)
from .prompts import MCQ, NUMERIC, TEXT, YESNO, options_block, prompt
from .smoke import smoke

__all__ = ["extract_tar", "extract_zip", "letters", "read_parquet", "save_image", "upstream_assignment",
           "upstream_constant", "upstream_functions", "upstream_module", "upstream_path", "by", "by_length",
           "group_all", "metrics", "pct", "clean", "correct", "equivalent", "final_answer", "normalize",
           "numeric_sequence", "parse", "parse_choice", "parse_number", "parse_yesno", "scalar", "MCQ", "NUMERIC",
           "TEXT", "YESNO", "options_block", "prompt", "smoke"]
