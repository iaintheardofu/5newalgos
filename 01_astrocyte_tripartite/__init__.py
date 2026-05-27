"""
algorithms/01_astrocyte_tripartite
===================================

Astrocyte-Modulated Tripartite-Synapse Network (NALSM)
-------------------------------------------------------

Faithful implementation of:
  - Kozachkov, Kastanenka & Krotov (PNAS 2023):
      "Building transformers from neurons and astrocytes"
  - Ivanov & Michmizos (NeurIPS 2021):
      "Increasing Liquid State Machine Performance with Edge-of-Chaos Dynamics"

Quick start
-----------
>>> from algorithms.01_astrocyte_tripartite import TripartiteNetwork, NetworkConfig
>>> from algorithms.01_astrocyte_tripartite import make_synthetic_mnist
>>>
>>> cfg = NetworkConfig(n_input=784, n_hidden=500, n_output=10)
>>> net = TripartiteNetwork(cfg)
>>> X_tr, y_tr, X_te, y_te = make_synthetic_mnist(2000, 400)
>>> reps = net.train_unsupervised(X_tr)
>>> net.readout.fit(reps, y_tr)
>>> print(net.score(X_te, y_te))

Module layout
-------------
astrocyte_network.py   Core implementation:
                       - LIFConfig, AstrocyteConfig, STDPConfig, NetworkConfig
                       - LIFLayer, AstrocyteLayer, TripartiteSynapseMatrix
                       - LinearReadout, TripartiteNetwork
                       - make_synthetic_mnist()

visualize.py           Six publication-quality figures saved as PNG:
                       1. astrocyte_ca_dynamics.png
                       2. spike_raster.png
                       3. attention_heatmap.png
                       4. weight_evolution.png
                       5. network_architecture.png
                       6. energy_comparison.png

benchmark.py           Full benchmark suite:
                       1. MNIST accuracy
                       2. Parameter count vs. Transformer / MLP
                       3. Energy estimation (pJ model)
                       4. Memory footprint
                       5. Training throughput
                       6. Ablation study (Ca2+ gating effect)

Run benchmarks
--------------
    python benchmark.py           # full suite
    python benchmark.py --fast    # reduced sizes for quick testing

Generate figures
----------------
    python visualize.py
"""

from .astrocyte_network import (
    AstrocyteConfig,
    AstrocyteLayer,
    LIFConfig,
    LIFLayer,
    LinearReadout,
    NetworkConfig,
    STDPConfig,
    TripartiteNetwork,
    TripartiteSynapseMatrix,
    make_synthetic_mnist,
)

__all__ = [
    # Configuration dataclasses
    "LIFConfig",
    "AstrocyteConfig",
    "STDPConfig",
    "NetworkConfig",
    # Core layers
    "LIFLayer",
    "AstrocyteLayer",
    "TripartiteSynapseMatrix",
    "LinearReadout",
    # Top-level network
    "TripartiteNetwork",
    # Utility
    "make_synthetic_mnist",
]

__version__ = "1.0.0"
__authors__ = [
    "Implementation: AI Cowboys Workforce (Sonnet 4.6)",
    "Paper: Kozachkov, Kastanenka & Krotov (PNAS 2023)",
    "Paper: Ivanov & Michmizos (NeurIPS 2021)",
]
