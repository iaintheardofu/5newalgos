"""
algorithms.03_myelin_delay
===========================

Oligodendrocyte / Myelin-Plastic Polychronous Spiking Neural Network.

Implements three canonical contributions in pure numpy:

  * DCLS delays          — Hammouamri, Khalfaoui-Hassani & Masquelier
                           (ICLR 2024, arXiv 2306.17670)
  * OMP myelination      — Talidou et al. (eLife 2023, 12:e76893)
  * Polychronous groups  — Izhikevich (Neural Computation 2006, 18(2))

Public exports
--------------
Core model
    LIFNeuron               Leaky-integrate-and-fire neuron population
    LIFParams               LIF biophysical parameters
    DCLSDelay               Differentiable delay layer (Gaussian relaxation)
    OligodendrocyteMod      Homeostatic myelination rule (OMP)
    STDPRule                STDP with delay-modulated eligibility traces
    STDPParams              STDP hyper-parameters
    PolychronousSNN         Full network (LIF + DCLS + OMP + STDP)
    NetworkConfig           Network construction parameters

Group detection
    PolychronousGroup       Dataclass for a detected polychronous group
    detect_polychronous_groups  Offline PG finder (Izhikevich 2006 §3)

Utilities
    generate_temporal_patterns   Synthetic spatiotemporal spike patterns
    ReadoutLayer                 Linear readout for classification

Visualisations (requires matplotlib)
    visualize.generate_all       Generate all seven PNG plots

Benchmarks
    temporal_benchmark.benchmark_shd
    temporal_benchmark.benchmark_pattern_discrimination
    temporal_benchmark.benchmark_spike_efficiency
    temporal_benchmark.benchmark_pg_scaling
    temporal_benchmark.run_all_benchmarks

Quick start
-----------
>>> from algorithms.myelin_delay import PolychronousSNN, NetworkConfig
>>> net = PolychronousSNN(NetworkConfig(n_excit=200, T_sim=5000))
>>> results = net.run()
>>> print(len(results["polychronous_groups"]), "polychronous groups detected")
"""

from polychronous_snn import (
    # Neurons
    LIFNeuron,
    LIFParams,
    # Delay layer
    DCLSDelay,
    # Myelination
    OligodendrocyteMod,
    # STDP
    STDPRule,
    STDPParams,
    # Full network
    PolychronousSNN,
    NetworkConfig,
    # Group detection
    PolychronousGroup,
    detect_polychronous_groups,
    # Utilities
    generate_temporal_patterns,
    ReadoutLayer,
)

__all__ = [
    # Neurons
    "LIFNeuron",
    "LIFParams",
    # Delay layer
    "DCLSDelay",
    # Myelination
    "OligodendrocyteMod",
    # STDP
    "STDPRule",
    "STDPParams",
    # Full network
    "PolychronousSNN",
    "NetworkConfig",
    # Group detection
    "PolychronousGroup",
    "detect_polychronous_groups",
    # Utilities
    "generate_temporal_patterns",
    "ReadoutLayer",
]

__version__ = "1.0.0"
__authors__ = [
    "Implementation based on Hammouamri et al. (ICLR 2024)",
    "Talidou et al. (eLife 2023)",
    "Izhikevich (Neural Computation 2006)",
]
