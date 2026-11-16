"""streamreplace: can streaming video-language models replace the offline models they were initialised from?

Evaluates each streaming child against its offline parent under three conditions: P (parent, offline pipeline),
C (child, same offline pipeline) and S (child, native streaming interface, question at the end of the video), with
the streaming tax S - P = (C - P) + (S - C); plus the diagnostic experiments, the training-free plugin Stream++ and
the generalization experiments. Start with `streamreplace --help` or `python -m streamreplace --help`."""

__version__ = "0.1.0"
