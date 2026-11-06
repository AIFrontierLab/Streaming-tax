"""Model adapters: the interfaces (base.py), the name -> class registry (registry.py), and one module per adapter.
Adapter modules import torch and the upstream code lazily, inside the model's own Python environment; nothing in
this package's __init__ imports them."""
