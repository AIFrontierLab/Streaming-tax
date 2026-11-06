"""Image understanding benchmarks, evaluated for the parent (P) and the child (C) in the same offline pipeline (image
benchmarks have no streaming condition): DynaMath, MathVista, MMBench, MME, MMMU, MMStar, ScienceQA, and VisuLogic
(supplement). One module per benchmark; use them through `streamreplace.benchmarks.registry.get(name)`, which fills
in the shared defaults of benchmarks/common for the hooks a module does not define. docs/benchmarks/<name>.md
describes each benchmark's data, prompt, parsing, metric and deviations from the official protocol."""
