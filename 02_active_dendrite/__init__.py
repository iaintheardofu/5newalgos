"""
Active-Dendrite NMDA Sub-unit Networks
=======================================
Package implementing multi-compartment neurons with active dendritic branches
for continual learning without catastrophic forgetting.

References
----------
- Poirazi, P. & Mel, B. W. (2003). Impact of Active Dendrites and Structural
  Plasticity on the Memory Capacity of Neural Tissue.
  Neuron, 29(3), 779-796.

- Iyer, R., Bhatt, U., Ahmad, S., & Hawkins, J. (2022). The Role of Dendritic
  Computation in Sparse Distributed Representations for Continual Learning.
  Frontiers in Computational Neuroscience, 16.

- Cichon, J. & Gan, W.-B. (2015). Branch-specific dendritic Ca2+ spikes cause
  persistent synaptic plasticity. Nature, 520(7546), 180-185.

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural
  networks. PNAS, 114(13), 3521-3526.

- Losonczy, A. & Magee, J. C. (2006). Integrative properties of radial oblique
  dendrites in hippocampal CA1 pyramidal neurons. Neuron, 50(2), 291-307.

Public API
----------
Core network::

    from algorithms.02_active_dendrite import (
        ActiveDendriteNetwork,
        ActiveDendriteNeuron,
        DendriticBranch,
        NetworkConfig,
        BranchConfig,
        NeuronConfig,
    )

NMDA nonlinearities::

    from algorithms.02_active_dendrite import (
        sigmoid_plateau,
        hard_threshold_plateau,
        relu_plateau,
    )

Baselines::

    from algorithms.02_active_dendrite import (
        BaselineMLP,
        EWCBaseline,
    )

Plasticity::

    from algorithms.02_active_dendrite import STDPTrace

Quick start::

    cfg = NetworkConfig(n_neurons=32, n_input=128, n_output=5)
    net = ActiveDendriteNetwork(cfg)

    # Register a task — generates a sparse context vector (SDR)
    net.register_task(task_id=0)

    # Forward pass with task context
    import numpy as np
    x = np.random.randn(128)
    logits = net.forward(x, task_id=0)   # shape: (5,)

    # Train
    net.train_task(
        task_id=0,
        x_train=x_train,
        y_train=y_train,
        n_epochs=10,
    )

    # Consolidate (EWC) before next task
    net.consolidate_after_task(0, x_train, y_train)
"""

from .dendrite_network import (
    # Configuration
    BranchConfig,
    NeuronConfig,
    NetworkConfig,
    # NMDA nonlinearities
    sigmoid_plateau,
    hard_threshold_plateau,
    relu_plateau,
    apply_nonlinearity,
    # Plasticity
    STDPTrace,
    # Core components
    DendriticBranch,
    ActiveDendriteNeuron,
    ActiveDendriteNetwork,
    # Baselines
    BaselineMLP,
    EWCBaseline,
)

__all__ = [
    # Configuration
    "BranchConfig",
    "NeuronConfig",
    "NetworkConfig",
    # NMDA nonlinearities
    "sigmoid_plateau",
    "hard_threshold_plateau",
    "relu_plateau",
    "apply_nonlinearity",
    # Plasticity
    "STDPTrace",
    # Core components
    "DendriticBranch",
    "ActiveDendriteNeuron",
    "ActiveDendriteNetwork",
    # Baselines
    "BaselineMLP",
    "EWCBaseline",
]

__version__ = "1.0.0"
__authors__ = ["Workforce v16"]
__references__ = [
    "Poirazi & Mel (2003) Neuron 29(3):779-796",
    "Iyer et al. (2022) Front. Comput. Neurosci. 16",
    "Cichon & Gan (2015) Nature 520:180-185",
    "Kirkpatrick et al. (2017) PNAS 114(13):3521-3526",
]
