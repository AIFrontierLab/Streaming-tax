"""Generation runs for the conditions P, C and S (and the plugin arms): planning, dispatch and the model worker.

- plan.py      jobs = pairs x conditions x benchmarks (+ plugin variants, + models from extra config files), their
               resolved settings and run identities
- dispatch.py  resume checks (identity, prompt drift, --fresh archive), GPU slot pool, one worker process per
               (arm, shard), logs and status
- worker.py    runs inside the model's own environment: loads the model once, appends one row per sample
- envs.py      interpreter and environment variables of a worker process

Typical use (what `streamreplace run` does):
    from streamreplace.config import load_config
    from streamreplace.run import dispatch, plan
    jobs = plan.plan(load_config(), pairs="p2", protocols="P,C,S", benchmarks="paper-core")
    status = dispatch.run(jobs, gpus=[0, 0, 1, 1])

This package's __init__ imports nothing, so `python -m streamreplace.run.worker` stays light."""
