"""Calibrating the plugin for a new streaming child (docs/plugin.md, "Calibrating a new child").

Stages (`streamreplace plugin calibrate CHILD --parent PARENT --stage ...`):
  directions  offline, child and parent: answer-position states of 600 AdvBench / Alpaca texts and 144 Video-MME
              videos (inputs.py); refusal direction and its layer by fold-1 AUC; parent-restoration vector, its layer
              and strength by KL(parent || steered child) (directions.py)        -> <work>/<child>/directions.npz
  freeze      rhat at r_layer and alpha * p at p_layer, with sha256 (freeze.py) -> plugin_artefacts/<child>/
  record      native stream with Recall + Respond, gate in record mode, on the 301 threshold items -> tau_closed
  harmless    the S_fix arm (gate in shadow) on the 100 harmless fit requests (harmless.py)          -> tau_free
The shipped artefacts of the paper's six children are frozen: a rebuild is not bit-identical (GPU kernels, the
float16 state cache), so paper numbers always use plugin_artefacts/ as shipped."""
