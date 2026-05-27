"""
Three-Factor Neuromodulated Plasticity
=======================================

Implements the Fremaux-Gerstner (2016) three-factor learning rule combined
with Tadros et al. (2022) sleep-replay consolidation and Sandia (2017)
neurogenesis-as-regularization for lifelong learning without a rehearsal buffer.

Public API
----------
Core components:

    EligibilityTraceBuffer
        Synapse-local STDP-like eligibility trace matrix.
        e(t+1) = (1 - 1/tau_e)*e(t) + x_pre*x_post
        dw = eta * e * M(t)

    NeuromodulatorSignals
        Container for DA / ACh / NE broadcast signals.
        M(t) = (DA - 1) * ACh * NE

    ThreeFactorLayer
        Fully-connected hidden layer with eligibility traces and
        the three-factor update rule baked in.

    OutputLayer
        Linear softmax output layer trained by cross-entropy gradient.

    SleepReplayConsolidator
        Offline Hebbian consolidation (Tadros et al. 2022):
        dw = alpha * (H(h) outer x_noise - lambda * W)

    NeurogenesisRegularizer
        Structured unit re-initialization (Sandia 2017):
        Replace bottom 5% of hidden units by contribution score.

    ThreeFactorNetwork
        Full combined system: hidden (three-factor) + output (SGD),
        with sleep-replay and neurogenesis integrated.

    RewardPredictionError
        TD(0) estimator for generating principled dopamine (DA) signals.

    SyntheticTaskGenerator
        Generates N synthetic classification tasks with configurable overlap.

Benchmark:

    run_forgetting_benchmark(...)
        Compare three-factor+sleep+neuro vs naive SGD on sequential tasks.
        Returns BWT, FWT, and per-task accuracy matrices.

Demo:

    lifelong_learning_demo.LifelongLearningSession
        Full 12-task sequential training session with BWT/FWT reporting.

Visualizations:

    visualize.main()
        Generate all 8 PNG figures.

References
----------
Fremaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of
  three-factor learning rules. Frontiers in Neural Circuits, 9, 85.

Tadros, T. et al. (2022). Sleep-replay consolidation prevents catastrophic
  forgetting in neural networks. Nature Communications, 13, 7842.

Sandia National Laboratories (2017). Neurogenesis Deep Learning.
  arXiv:1710.06759 [cs.NE].
"""

from three_factor_system import (
    EligibilityTraceBuffer,
    NeuromodulatorSignals,
    ThreeFactorLayer,
    OutputLayer,
    SleepReplayConsolidator,
    NeurogenesisRegularizer,
    ThreeFactorNetwork,
    RewardPredictionError,
    SyntheticTaskGenerator,
    run_forgetting_benchmark,
)

__all__ = [
    # Core building blocks
    "EligibilityTraceBuffer",
    "NeuromodulatorSignals",
    "ThreeFactorLayer",
    "OutputLayer",
    # Subsystems
    "SleepReplayConsolidator",
    "NeurogenesisRegularizer",
    # Full network
    "ThreeFactorNetwork",
    # Utilities
    "RewardPredictionError",
    "SyntheticTaskGenerator",
    # Benchmarks
    "run_forgetting_benchmark",
]

__version__ = "1.0.0"
__authors__ = [
    "Fremaux & Gerstner (2016) — three-factor rule",
    "Tadros et al. (2022) — sleep-replay consolidation",
    "Sandia NL (2017) — neurogenesis-as-regularization",
]
