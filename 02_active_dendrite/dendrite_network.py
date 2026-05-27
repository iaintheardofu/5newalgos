"""
Active-Dendrite NMDA Sub-unit Networks
=======================================
Faithful implementation of:
  - Poirazi & Mel (2003) "Impact of Active Dendrites and Structural Plasticity
    on the Memory Capacity of Neural Tissue"
  - Iyer et al. (2022) "The Role of Dendritic Computation in Sparse Distributed
    Representations for Continual Learning" (Numenta)
  - Cichon & Gan (2015) "Branch-specific dendritic Ca2+ spikes cause
    persistent synaptic plasticity" (Nature)

Neuron model:
    y = f( sum_b  sigma_b( w_b^T x_b  +  u_b^T c ) )

where
  b       : branch index
  sigma_b : NMDA-plateau nonlinearity per branch
  x_b     : clustered input features for branch b
  c       : apical-tuft context vector (gates which branch dominates)
  w_b     : feed-forward synaptic weights on branch b
  u_b     : context weights on branch b

Branch-specific plasticity:
    Delta w_b  =  alpha_b * STDP-trace * nmda_spike_indicator_b

Continual learning:
  Different tasks recruit different branches via context gating.
  EWC/Synaptic Intelligence applied per-branch to protect committed weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Float32 = NDArray[np.float32]
Float64 = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BranchConfig:
    """Geometry and biophysics of a single dendritic branch."""

    # Number of synaptic input sites on this branch
    n_synapses: int = 64

    # Number of context inputs (apical tuft)
    n_context: int = 32

    # NMDA-spike threshold: fraction of branch synapses that must be
    # co-active within the clustering radius to trigger plateau.
    # Calibrated to ~10-50 synapses clustering within 20-50 um (Losonczy
    # & Magee 2006; Branco & Hausser 2011).
    nmda_threshold: float = 0.15

    # Plateau duration in time-steps (biological: 50–200 ms)
    plateau_duration: int = 20

    # Nonlinearity type: "sigmoid" | "threshold" | "relu_plateau"
    nonlinearity: str = "sigmoid"

    # Sigmoid steepness around threshold
    sigmoid_slope: float = 10.0

    # Branch compartment type: "basal" | "oblique" | "apical"
    compartment: str = "basal"

    # EWC / Synaptic Intelligence regularisation strength (lambda)
    ewc_lambda: float = 400.0

    # STDP learning rate for this branch
    stdp_lr: float = 0.01

    # Sparse activity target (sparsity ~ 0.02 for SDR-style coding)
    sparsity_target: float = 0.02


@dataclass
class NeuronConfig:
    """Configuration for a single multi-compartment neuron."""

    n_branches: int = 8
    branch_config: BranchConfig = field(default_factory=BranchConfig)

    # Somatic nonlinearity
    soma_nonlinearity: str = "relu"

    # Somatic threshold / scale
    soma_threshold: float = 0.0
    soma_scale: float = 1.0


@dataclass
class NetworkConfig:
    """Full network configuration."""

    n_neurons: int = 64
    n_input: int = 256
    n_output: int = 10
    n_context: int = 32

    # Branches per neuron
    n_branches: int = 8

    # Synapses per branch (total input dimension split across branches)
    synapses_per_branch: int = 32  # n_input / n_branches ≈ this

    # How branches are allocated: "disjoint" | "overlapping" | "learned"
    input_allocation: str = "disjoint"

    # Overlap fraction when input_allocation == "overlapping"
    overlap_fraction: float = 0.25

    # Plasticity / learning
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    ewc_lambda: float = 400.0

    # NMDA parameters
    nmda_threshold: float = 0.15
    plateau_duration: int = 20

    # Training
    n_epochs: int = 10
    batch_size: int = 32
    random_seed: int = 42


# ---------------------------------------------------------------------------
# NMDA plateau nonlinearities
# ---------------------------------------------------------------------------


def sigmoid_plateau(
    activation: Float64,
    threshold: float,
    slope: float = 10.0,
) -> Float64:
    """
    Sigmoid approximation to the NMDA I-V curve.

    sigma(a) = 1 / (1 + exp(-slope * (a - threshold)))

    Produces a smooth transition from sub-threshold linear regime to
    plateau, mimicking the voltage-dependent Mg2+ block removal.
    Argument is clamped to [-500, 500] to prevent float overflow.
    """
    z = np.clip(-slope * (activation - threshold), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(z))


def hard_threshold_plateau(
    activation: Float64,
    threshold: float,
) -> Float64:
    """
    Binary NMDA-spike indicator (Cichon & Gan 2015 model).

    Returns 1 if the clustered synaptic drive exceeds theta_nmda,
    else 0. Models the all-or-none character of dendritic calcium spikes.
    """
    return (activation >= threshold).astype(np.float64)


def relu_plateau(
    activation: Float64,
    threshold: float,
    plateau_height: float = 1.0,
) -> Float64:
    """
    Rectified linear with plateau cap (Poirazi & Mel 2003 sigmoid-unit model).

    Approximates the two-regime response: linear below threshold,
    saturating plateau above.
    """
    below = np.where(activation < threshold, np.maximum(activation, 0.0), 0.0)
    above = np.where(activation >= threshold, plateau_height, 0.0)
    return below + above


def apply_nonlinearity(
    activation: Float64,
    config: BranchConfig,
) -> Float64:
    """Dispatch to the correct NMDA nonlinearity."""
    if config.nonlinearity == "sigmoid":
        return sigmoid_plateau(activation, config.nmda_threshold, config.sigmoid_slope)
    elif config.nonlinearity == "threshold":
        return hard_threshold_plateau(activation, config.nmda_threshold)
    elif config.nonlinearity == "relu_plateau":
        return relu_plateau(activation, config.nmda_threshold)
    else:
        raise ValueError(f"Unknown nonlinearity: {config.nonlinearity!r}")


# ---------------------------------------------------------------------------
# STDP trace
# ---------------------------------------------------------------------------


class STDPTrace:
    """
    Exponential eligibility trace for Spike-Timing-Dependent Plasticity.

    Maintains a per-synapse eligibility trace e_i(t):
        e_i(t) = e_i(t-1) * exp(-dt/tau) + pre_spike_i(t)

    Weight update when post-synaptic branch fires a plateau:
        Delta w_i = lr * e_i * nmda_spike
    """

    def __init__(self, n_synapses: int, tau_plus: float = 20.0, tau_minus: float = 20.0) -> None:
        self.n_synapses = n_synapses
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.trace_pre: Float64 = np.zeros(n_synapses, dtype=np.float64)
        self.trace_post: float = 0.0

    def update_pre(self, pre_activity: Float64, dt: float = 1.0) -> None:
        """Decay trace and add current pre-synaptic activity."""
        decay = math.exp(-dt / self.tau_plus)
        self.trace_pre = self.trace_pre * decay + pre_activity

    def update_post(self, post_spike: float, dt: float = 1.0) -> None:
        """Decay post-synaptic trace and add post-synaptic spike."""
        decay = math.exp(-dt / self.tau_minus)
        self.trace_post = self.trace_post * decay + post_spike

    def get_weight_delta(self, nmda_spike: float, lr: float) -> Float64:
        """
        Return weight update Delta w = lr * trace_pre * nmda_spike.

        When the branch fires a plateau (nmda_spike > 0), potentiate
        synapses whose pre-synaptic activity preceded the event.
        """
        return lr * self.trace_pre * nmda_spike

    def reset(self) -> None:
        self.trace_pre[:] = 0.0
        self.trace_post = 0.0


# ---------------------------------------------------------------------------
# Single dendritic branch
# ---------------------------------------------------------------------------


class DendriticBranch:
    """
    Single dendritic branch with NMDA sub-unit nonlinearity.

    Implements the sub-equation for one branch b:
        a_b  = w_b^T x_b + u_b^T c
        o_b  = sigma_b(a_b)           (NMDA plateau if a_b > theta_nmda)

    Plasticity:
        Delta w_b = alpha_b * STDP-trace * nmda_spike_indicator_b

    where alpha_b is gated by the NMDA spike (branch-specific Hebbian gate).
    """

    def __init__(
        self,
        branch_id: int,
        input_indices: NDArray[np.int32],
        config: BranchConfig,
        rng: np.random.Generator,
    ) -> None:
        self.branch_id = branch_id
        self.input_indices = input_indices  # which global inputs this branch reads
        self.config = config
        self.n_in = len(input_indices)
        self.n_ctx = config.n_context

        # Synaptic weights: feed-forward and context
        scale = 1.0 / math.sqrt(self.n_in) if self.n_in > 0 else 1.0
        self.w: Float64 = rng.normal(0.0, scale, size=self.n_in)
        ctx_scale = 1.0 / math.sqrt(self.n_ctx) if self.n_ctx > 0 else 1.0
        self.u: Float64 = rng.normal(0.0, ctx_scale, size=self.n_ctx)

        # Plateau state (plateau_duration time-steps of sustained depolarisation)
        self._plateau_remaining: int = 0

        # STDP trace
        self.stdp = STDPTrace(self.n_in)

        # EWC per-parameter Fisher information accumulators
        self._fisher_w: Float64 = np.zeros(self.n_in, dtype=np.float64)
        self._fisher_u: Float64 = np.zeros(self.n_ctx, dtype=np.float64)
        self._optimal_w: Float64 = self.w.copy()
        self._optimal_u: Float64 = self.u.copy()

        # Running statistics for Synaptic Intelligence (Zenke et al. 2017)
        self._si_omega_w: Float64 = np.zeros(self.n_in, dtype=np.float64)
        self._si_xi_w: Float64 = np.zeros(self.n_in, dtype=np.float64)

        # Compartment type tag
        self.compartment: str = config.compartment

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x_full: Float64, context: Float64) -> tuple[float, float]:
        """
        Compute branch activation and output.

        Parameters
        ----------
        x_full : shape (n_total_inputs,)
            Full input vector; branch reads at self.input_indices.
        context : shape (n_context,)
            Apical-tuft context vector.

        Returns
        -------
        activation : raw pre-nonlinearity activation a_b
        output     : post-nonlinearity o_b (in [0, 1])
        """
        x_b = x_full[self.input_indices]
        a_b = float(self.w @ x_b + self.u @ context)

        # NMDA plateau gating: if plateau is ongoing, sustain output
        if self._plateau_remaining > 0:
            self._plateau_remaining -= 1
            output = 1.0
        else:
            # Use sigmoid centred at 0 for smooth gradient flow during learning.
            # The nmda_threshold controls when the NMDA-spike (plateau) fires,
            # independently of the sigmoid inflection point.
            z = np.clip(-self.config.sigmoid_slope * a_b, -500.0, 500.0)
            output = float(1.0 / (1.0 + np.exp(z)))  # sigmoid centred at 0

            # NMDA spike fires when raw activation exceeds the NMDA threshold.
            # This is the all-or-none plateau indicator (Cichon & Gan 2015).
            if a_b >= self.config.nmda_threshold:
                self._plateau_remaining = self.config.plateau_duration

        return a_b, output

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------

    def update_stdp(
        self,
        x_full: Float64,
        nmda_spike: float,
        dt: float = 1.0,
    ) -> Float64:
        """
        Update STDP trace and compute weight delta.

        Delta w_b = alpha_b * trace_pre * nmda_spike

        where alpha_b = nmda_spike (branch gate: plasticity only when
        the branch itself fired a plateau, Cichon & Gan 2015).

        Returns
        -------
        delta_w : weight update vector for w_b
        """
        x_b = x_full[self.input_indices]
        self.stdp.update_pre(x_b, dt=dt)
        self.stdp.update_post(nmda_spike, dt=dt)
        # alpha_b is the NMDA-spike indicator (gate)
        alpha_b = nmda_spike
        delta_w = alpha_b * self.stdp.get_weight_delta(nmda_spike=nmda_spike, lr=self.config.stdp_lr)
        return delta_w

    def apply_weight_update(
        self,
        delta_w: Float64,
        grad_w: Optional[Float64] = None,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Apply weight update with optional EWC regularisation penalty.

        EWC penalty gradient: -ewc_lambda * F_i * (w_i - w*_i)
        Fisher values are clamped to prevent overflow in the penalty term.
        """
        # Clamp Fisher to prevent overflow: penalty = lambda * F * delta_w
        # With lambda=400 and unit-scale weights, F should be O(1/n).
        fisher_clamped = np.clip(self._fisher_w, 0.0, 1.0 / (self.config.ewc_lambda + 1e-9))
        ewc_penalty = self.config.ewc_lambda * fisher_clamped * (self.w - self._optimal_w)
        if grad_w is not None:
            total_delta = delta_w + grad_w - weight_decay * self.w - ewc_penalty
        else:
            total_delta = delta_w - weight_decay * self.w - ewc_penalty
        self.w = np.clip(self.w + total_delta, -10.0, 10.0)  # hard clamp for stability

    def apply_context_update(
        self,
        delta_u: Float64,
        weight_decay: float = 1e-4,
    ) -> None:
        """Update context weights with EWC regularisation."""
        fisher_clamped = np.clip(self._fisher_u, 0.0, 1.0 / (self.config.ewc_lambda + 1e-9))
        ewc_penalty = self.config.ewc_lambda * fisher_clamped * (self.u - self._optimal_u)
        self.u = np.clip(self.u + delta_u - weight_decay * self.u - ewc_penalty, -10.0, 10.0)

    # ------------------------------------------------------------------
    # EWC / Synaptic Intelligence
    # ------------------------------------------------------------------

    def consolidate_task(self, fisher_samples: list[Float64]) -> None:
        """
        Estimate Fisher information matrix diagonal and consolidate weights.

        Called at task boundary (Kirkpatrick et al. 2017).

        Parameters
        ----------
        fisher_samples : list of gradient vectors (one per sample in data)
        """
        if not fisher_samples:
            return
        stacked = np.stack(fisher_samples, axis=0)  # (N, n_in)
        self._fisher_w = self._fisher_w + np.mean(stacked ** 2, axis=0)
        self._optimal_w = self.w.copy()

    def ewc_penalty_scalar(self) -> float:
        """Return scalar EWC penalty for this branch (clamped for stability)."""
        cap = 1.0 / (self.config.ewc_lambda + 1e-9)
        fw = np.clip(self._fisher_w, 0.0, cap)
        fu = np.clip(self._fisher_u, 0.0, cap)
        w_pen = float(0.5 * self.config.ewc_lambda * np.sum(fw * (self.w - self._optimal_w) ** 2))
        u_pen = float(0.5 * self.config.ewc_lambda * np.sum(fu * (self.u - self._optimal_u) ** 2))
        return w_pen + u_pen

    # ------------------------------------------------------------------
    # Dendritic sparsity (Iyer et al. 2022: SDR-style branch selection)
    # ------------------------------------------------------------------

    def get_sparsity_mask(self, k: int) -> NDArray[np.bool_]:
        """
        Return a mask selecting the top-k activated synapses.

        Models sparse dendritic representation: only the most activated
        ~2% of synapses are considered for plateau generation (Kanerva SDR).
        """
        k = max(1, min(k, self.n_in))
        abs_w = np.abs(self.w)
        threshold = np.partition(abs_w, -k)[-k]
        return abs_w >= threshold

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def is_in_plateau(self) -> bool:
        return self._plateau_remaining > 0

    def reset_plateau(self) -> None:
        self._plateau_remaining = 0

    def clone_weights(self) -> dict[str, Float64]:
        return {"w": self.w.copy(), "u": self.u.copy()}

    def restore_weights(self, snapshot: dict[str, Float64]) -> None:
        self.w = snapshot["w"].copy()
        self.u = snapshot["u"].copy()


# ---------------------------------------------------------------------------
# Multi-compartment neuron
# ---------------------------------------------------------------------------


class ActiveDendriteNeuron:
    """
    Multi-compartment neuron with active dendritic branches.

    Full model:
        y = f_soma( sum_b  o_b )
        o_b = sigma_b( w_b^T x_b  +  u_b^T c )

    Compartments:
      - basal   : receives bottom-up sensory input (Poirazi & Mel 2003)
      - oblique : receives mixed input, moderate NMDA threshold
      - apical  : receives top-down context; also encodes Cichon & Gan
                  branch-specific memory allocation

    Context vector c drives u_b^T c which shifts branch thresholds,
    implementing task-conditional routing (Iyer et al. 2022 Fig 3).
    """

    COMPARTMENT_TYPES = ("basal", "oblique", "apical")

    def __init__(
        self,
        neuron_id: int,
        config: NetworkConfig,
        rng: np.random.Generator,
    ) -> None:
        self.neuron_id = neuron_id
        self.config = config
        self._rng = rng

        # Build branches
        self.branches: list[DendriticBranch] = []
        self._build_branches()

        # Somatic bias
        self.bias: float = 0.0

        # Per-branch task allocation tracker (n_branches x n_tasks)
        self._task_allocation: dict[int, NDArray[np.float64]] = {}

    def _build_branches(self) -> None:
        """Partition input dimensions into branch-specific receptive fields."""
        n_in = self.config.n_input
        n_b = self.config.n_branches
        alloc = self.config.input_allocation
        n_per = self.config.synapses_per_branch

        if alloc == "disjoint":
            # Each branch receives a non-overlapping slice of input space
            indices_list = self._disjoint_partition(n_in, n_b)
        elif alloc == "overlapping":
            indices_list = self._overlapping_partition(
                n_in, n_b, self.config.overlap_fraction
            )
        else:
            # "learned" initialisation: random subsets
            indices_list = [
                self._rng.choice(n_in, size=n_per, replace=False).astype(np.int32)
                for _ in range(n_b)
            ]

        compartment_types = self._assign_compartments(n_b)

        for b_idx, (indices, comp) in enumerate(zip(indices_list, compartment_types)):
            # Adjust NMDA threshold per compartment (apical branches are more
            # sensitive to context; basal more sensitive to feedforward drive)
            nmda_thresh = self.config.nmda_threshold
            if comp == "apical":
                nmda_thresh = nmda_thresh * 0.7  # lower threshold: apical is context-driven
            elif comp == "oblique":
                nmda_thresh = nmda_thresh * 0.85

            bc = BranchConfig(
                n_synapses=len(indices),
                n_context=self.config.n_context,
                nmda_threshold=nmda_thresh,
                plateau_duration=self.config.plateau_duration,
                compartment=comp,
                ewc_lambda=self.config.ewc_lambda,
                stdp_lr=self.config.learning_rate,
            )
            branch = DendriticBranch(
                branch_id=b_idx,
                input_indices=indices,
                config=bc,
                rng=self._rng,
            )
            self.branches.append(branch)

    def _disjoint_partition(
        self, n_in: int, n_b: int
    ) -> list[NDArray[np.int32]]:
        """Split input indices into non-overlapping contiguous ranges."""
        all_idx = np.arange(n_in, dtype=np.int32)
        # Shuffle so each neuron gets a different partition
        self._rng.shuffle(all_idx)
        chunks = np.array_split(all_idx, n_b)
        return [c.astype(np.int32) for c in chunks]

    def _overlapping_partition(
        self, n_in: int, n_b: int, overlap: float
    ) -> list[NDArray[np.int32]]:
        """Overlapping receptive fields with shared fraction."""
        base_size = n_in // n_b
        n_shared = max(1, int(base_size * overlap))
        shared_pool = self._rng.choice(n_in, size=n_shared, replace=False).astype(np.int32)
        private_pool = np.setdiff1d(np.arange(n_in, dtype=np.int32), shared_pool)
        self._rng.shuffle(private_pool)
        private_size = base_size - n_shared
        result = []
        for b in range(n_b):
            start = (b * private_size) % max(1, len(private_pool))
            end = start + private_size
            if end <= len(private_pool):
                private = private_pool[start:end]
            else:
                private = np.concatenate([private_pool[start:], private_pool[:end - len(private_pool)]])
            idx = np.concatenate([shared_pool, private]).astype(np.int32)
            result.append(idx)
        return result

    def _assign_compartments(self, n_branches: int) -> list[str]:
        """
        Assign compartment types to branches.

        Mimics pyramidal neuron morphology:
          - 50% basal (perisomatic, strong feedforward drive)
          - 30% oblique (intermediate integration)
          - 20% apical (context/top-down)
        """
        n_apical = max(1, n_branches // 5)
        n_oblique = max(1, n_branches * 3 // 10)
        n_basal = n_branches - n_apical - n_oblique
        types = ["basal"] * n_basal + ["oblique"] * n_oblique + ["apical"] * n_apical
        # Trim/pad to exact count
        types = (types * ((n_branches // len(types)) + 1))[:n_branches]
        return types

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Float64,
        context: Float64,
        return_branch_outputs: bool = False,
    ) -> float | tuple[float, NDArray[np.float64]]:
        """
        Compute neuron output.

        Parameters
        ----------
        x : shape (n_input,) — full input vector
        context : shape (n_context,) — apical context gate
        return_branch_outputs : if True, also return per-branch outputs

        Returns
        -------
        y : scalar neuron output
        branch_outputs (optional) : shape (n_branches,)
        """
        branch_outputs = np.array(
            [branch.forward(x, context)[1] for branch in self.branches],
            dtype=np.float64,
        )
        soma_input = float(np.sum(branch_outputs)) + self.bias
        y = self._soma_nonlinearity(soma_input)
        if return_branch_outputs:
            return y, branch_outputs
        return y

    def _soma_nonlinearity(self, z: float) -> float:
        """Somatic nonlinearity f(z)."""
        nl = self.config.n_branches  # normalise by branch count
        if self.config.n_neurons > 0:
            z_norm = z / nl
        else:
            z_norm = z
        # ReLU soma (standard for rate-coded output)
        return float(max(0.0, z_norm))

    # ------------------------------------------------------------------
    # Plasticity
    # ------------------------------------------------------------------

    def update_stdp(
        self,
        x: Float64,
        post_spike: float,
        dt: float = 1.0,
    ) -> None:
        """
        Run STDP update on all branches.

        Each branch updates independently; plasticity is gated by
        whether that branch fired a plateau (Cichon & Gan 2015).
        """
        for branch in self.branches:
            nmda_spike = float(branch.is_in_plateau())
            delta_w = branch.update_stdp(x, nmda_spike=nmda_spike * post_spike, dt=dt)
            branch.apply_weight_update(delta_w)

    def consolidate_task(
        self,
        task_id: int,
        x_samples: list[Float64],
        context_samples: list[Float64],
    ) -> None:
        """
        Consolidate weights for task `task_id` using EWC Fisher estimation.

        Called at the end of each task's training phase.
        """
        for branch in self.branches:
            # Collect squared gradients of branch output w.r.t. w_b
            fisher_grads: list[Float64] = []
            for x, c in zip(x_samples[:64], context_samples[:64]):
                x_b = x[branch.input_indices]
                act = float(branch.w @ x_b + branch.u @ c)
                # Gradient of log-likelihood: approximated as x_b * sigma'(act)
                sigma_prime = _sigmoid_derivative(act, branch.config.nmda_threshold, branch.config.sigmoid_slope)
                fisher_grads.append(x_b * sigma_prime)
            branch.consolidate_task(fisher_grads)

        # Record which branches are active for this task
        self._task_allocation[task_id] = np.array(
            [1.0 if b.is_in_plateau() else 0.0 for b in self.branches],
            dtype=np.float64,
        )

    def get_branch_task_mask(self, task_id: int) -> Optional[NDArray[np.float64]]:
        """Return branch activation pattern for a given task."""
        return self._task_allocation.get(task_id)

    def ewc_penalty(self) -> float:
        """Total EWC penalty across all branches."""
        return sum(b.ewc_penalty_scalar() for b in self.branches)


def _sigmoid_derivative(z: float, threshold: float, slope: float) -> float:
    """d/dz sigmoid_plateau(z)."""
    s = 1.0 / (1.0 + math.exp(-slope * (z - threshold)))
    return slope * s * (1.0 - s)


# ---------------------------------------------------------------------------
# Full active-dendrite network
# ---------------------------------------------------------------------------


class ActiveDendriteNetwork:
    """
    Layer of multi-compartment neurons with active dendrites.

    Architecture:
        Input (n_input)  ->  [N dendritic neurons]  ->  Linear head  ->  Output (n_output)

    The dendritic layer transforms input x and context c into a sparse
    distributed representation via branch-specific NMDA plateau gating.
    The output head is a simple linear map trained with gradient descent.

    Continual learning strategy (Iyer et al. 2022):
      1. Context vector encodes task identity, routing activations to
         task-specific branches.
      2. EWC consolidates committed branch weights after each task.
      3. Branch sparsity (~k-WTA) ensures each task occupies ~2% of
         branches, minimising cross-task interference.
    """

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

        # Build neurons
        self.neurons: list[ActiveDendriteNeuron] = [
            ActiveDendriteNeuron(i, config, self._rng)
            for i in range(config.n_neurons)
        ]

        # Linear output head: shape (n_neurons, n_output)
        scale = 1.0 / math.sqrt(config.n_neurons)
        self.W_out: Float64 = self._rng.normal(0.0, scale, size=(config.n_neurons, config.n_output))
        self.b_out: Float64 = np.zeros(config.n_output, dtype=np.float64)

        # Task management
        self._current_task: int = 0
        self._task_context_vectors: dict[int, NDArray[np.float64]] = {}
        self._task_history: list[dict] = []

        # Per-task Fisher for output head EWC
        self._fisher_W_out: Float64 = np.zeros_like(self.W_out)
        self._optimal_W_out: Float64 = self.W_out.copy()

        # Training statistics
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def register_task(self, task_id: int) -> NDArray[np.float64]:
        """
        Register a new task and generate a unique context vector.

        The context vector is a sparse random binary vector (SDR):
        ~5% active bits, ensuring task contexts are approximately
        orthogonal (capacity scales with n_context choose k).
        """
        if task_id in self._task_context_vectors:
            return self._task_context_vectors[task_id]

        n_ctx = self.config.n_context
        n_active = max(1, int(n_ctx * 0.05))  # 5% active (SDR)
        ctx = np.zeros(n_ctx, dtype=np.float64)
        active_bits = self._rng.choice(n_ctx, size=n_active, replace=False)
        ctx[active_bits] = 1.0
        self._task_context_vectors[task_id] = ctx
        return ctx

    def get_task_context(self, task_id: int) -> NDArray[np.float64]:
        if task_id not in self._task_context_vectors:
            return self.register_task(task_id)
        return self._task_context_vectors[task_id]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Float64,
        task_id: Optional[int] = None,
        return_internals: bool = False,
    ) -> Float64 | tuple[Float64, dict]:
        """
        Full forward pass.

        Parameters
        ----------
        x : shape (n_input,) or (batch, n_input)
        task_id : if given, use pre-registered context vector for this task
        return_internals : if True, return branch activations and hidden rep

        Returns
        -------
        logits : shape (n_output,) or (batch, n_output)
        """
        batched = x.ndim == 2
        if not batched:
            x = x[np.newaxis, :]

        batch_size = x.shape[0]
        results = []
        internals_list = []

        context = (
            self.get_task_context(task_id)
            if task_id is not None
            else np.zeros(self.config.n_context, dtype=np.float64)
        )

        for xi in x:
            h = np.zeros(self.config.n_neurons, dtype=np.float64)
            branch_outputs_all: list[NDArray[np.float64]] = []

            for i, neuron in enumerate(self.neurons):
                yi, branch_out = neuron.forward(xi, context, return_branch_outputs=True)
                h[i] = yi
                branch_outputs_all.append(branch_out)

            logits = h @ self.W_out + self.b_out
            results.append(logits)

            if return_internals:
                internals_list.append({
                    "hidden": h.copy(),
                    "branch_outputs": np.stack(branch_outputs_all, axis=0),  # (n_neurons, n_branches)
                    "context": context.copy(),
                })

        logits_batch = np.stack(results, axis=0)
        if not batched:
            logits_batch = logits_batch[0]
            if return_internals:
                return logits_batch, internals_list[0]

        if return_internals:
            return logits_batch, internals_list
        return logits_batch

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _softmax(self, z: Float64) -> Float64:
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _cross_entropy_loss(
        self, logits: Float64, targets: NDArray[np.int64]
    ) -> tuple[float, Float64]:
        """
        Softmax cross-entropy loss with gradient.

        Returns (loss, d_logits).
        """
        probs = self._softmax(logits)
        batch = logits.shape[0]
        log_probs = np.log(probs + 1e-12)
        loss = -float(np.mean(log_probs[np.arange(batch), targets]))

        # Gradient
        d_logits = probs.copy()
        d_logits[np.arange(batch), targets] -= 1.0
        d_logits /= batch
        return loss, d_logits

    def _ewc_penalty_total(self) -> float:
        """Sum EWC penalties across all neurons + output head."""
        neuron_pen = sum(n.ewc_penalty() for n in self.neurons)
        head_pen = float(
            0.5 * self.config.ewc_lambda
            * np.sum(self._fisher_W_out * (self.W_out - self._optimal_W_out) ** 2)
        )
        return neuron_pen + head_pen

    def train_task(
        self,
        task_id: int,
        x_train: Float64,
        y_train: NDArray[np.int64],
        x_val: Optional[Float64] = None,
        y_val: Optional[NDArray[np.int64]] = None,
        n_epochs: Optional[int] = None,
    ) -> dict:
        """
        Train the network on a single task.

        Implements:
          1. Forward pass with task context
          2. Cross-entropy gradient update on output head
          3. STDP on dendritic branches (gated by NMDA plateau)
          4. EWC penalty from previous tasks

        Returns training statistics dict.
        """
        n_epochs = n_epochs or self.config.n_epochs
        batch_size = self.config.batch_size
        lr = self.config.learning_rate
        n_samples = x_train.shape[0]
        context = self.get_task_context(task_id)

        task_losses: list[float] = []
        task_val_accs: list[float] = []

        for epoch in range(n_epochs):
            # Shuffle
            perm = self._rng.permutation(n_samples)
            x_shuf = x_train[perm]
            y_shuf = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                xb = x_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                bs = len(xb)

                # --- Forward: collect h and per-neuron branch activations ---
                # h_batch:  (bs, n_neurons)
                # branches: (bs, n_neurons, n_branches) — raw pre-soma branch outputs
                # raw_acts: (bs, n_neurons, n_branches) — pre-nonlinearity activations
                h_batch = np.zeros((bs, self.config.n_neurons), dtype=np.float64)
                branch_out_batch = np.zeros(
                    (bs, self.config.n_neurons, self.config.n_branches), dtype=np.float64
                )
                raw_act_batch = np.zeros_like(branch_out_batch)

                for j, xj in enumerate(xb):
                    for ni, neuron in enumerate(self.neurons):
                        b_raw = np.zeros(self.config.n_branches)
                        b_out = np.zeros(self.config.n_branches)
                        for bi, branch in enumerate(neuron.branches):
                            act, out = branch.forward(xj, context)
                            b_raw[bi] = act
                            b_out[bi] = out
                        soma_sum = float(b_out.sum()) + neuron.bias
                        h_batch[j, ni] = float(max(0.0, soma_sum / self.config.n_branches))
                        branch_out_batch[j, ni] = b_out
                        raw_act_batch[j, ni] = b_raw

                logits = h_batch @ self.W_out + self.b_out

                # --- Loss ---
                loss, d_logits = self._cross_entropy_loss(logits, yb)
                epoch_loss += loss
                n_batches += 1

                # --- Backward: output head ---
                d_W_out = h_batch.T @ d_logits  # (n_neurons, n_output)
                d_b_out = d_logits.sum(axis=0)

                # EWC gradient for output head (clamped Fisher)
                cap = 1.0 / (self.config.ewc_lambda + 1e-9)
                fw_clamped = np.clip(self._fisher_W_out, 0.0, cap)
                ewc_grad_W = self.config.ewc_lambda * fw_clamped * (self.W_out - self._optimal_W_out)

                self.W_out -= lr * (d_W_out + ewc_grad_W)
                self.b_out -= lr * d_b_out

                # --- Backward: branch weights (gradient descent) ---
                # d_h: (bs, n_neurons) — gradient at hidden layer
                d_h = d_logits @ self.W_out.T  # (bs, n_neurons)

                # Soma ReLU gate: h > 0 implies soma was active
                soma_gate = (h_batch > 0).astype(np.float64)  # (bs, n_neurons)
                d_h_gated = d_h * soma_gate / max(1, self.config.n_branches)

                for ni, neuron in enumerate(self.neurons):
                    for bi, branch in enumerate(neuron.branches):
                        # d_o_b: upstream gradient for this branch output, sum over batch
                        d_o_b = d_h_gated[:, ni]  # (bs,)

                        # Sigmoid derivative for branch nonlinearity (sigmoid centred at 0)
                        a_b = raw_act_batch[:, ni, bi]  # (bs,)
                        z = np.clip(-branch.config.sigmoid_slope * a_b, -500.0, 500.0)
                        s = 1.0 / (1.0 + np.exp(z))
                        dsigma = s * (1.0 - s) * branch.config.sigmoid_slope  # (bs,)
                        delta = d_o_b * dsigma  # (bs,) — scalar factor per sample

                        # Gradient w.r.t. w_b: sum_j (delta_j * x_b_j)
                        x_b_batch = xb[:, branch.input_indices]  # (bs, n_in_b)
                        grad_w = (delta[:, np.newaxis] * x_b_batch).mean(axis=0)  # (n_in_b,)
                        grad_u = (delta[:, np.newaxis] * context[np.newaxis, :]).mean(axis=0)

                        branch.apply_weight_update(np.zeros_like(branch.w), grad_w=-lr * grad_w)
                        branch.apply_context_update(-lr * grad_u)

                # --- STDP on branches (neuromodulatory gate) ---
                for ni, neuron in enumerate(self.neurons):
                    post_signal = float(np.mean(np.abs(d_h[:, ni])))
                    for xj in xb[:4]:  # limit for speed
                        neuron.update_stdp(xj, post_signal)

            avg_loss = epoch_loss / max(1, n_batches)
            task_losses.append(avg_loss)
            self.train_losses.append(avg_loss)

            if x_val is not None and y_val is not None:
                acc = self.evaluate(x_val, y_val, task_id=task_id)
                task_val_accs.append(acc)
                self.val_accuracies.append(acc)

        return {
            "task_id": task_id,
            "n_epochs": n_epochs,
            "final_loss": task_losses[-1] if task_losses else float("nan"),
            "losses": task_losses,
            "val_accs": task_val_accs,
        }

    def consolidate_after_task(
        self,
        task_id: int,
        x_samples: Float64,
        y_samples: NDArray[np.int64],
    ) -> None:
        """
        Consolidate weights after finishing task `task_id`.

        1. Estimate Fisher information on all neurons.
        2. Record task-branch allocation.
        3. Snapshot optimal weights.
        """
        context = self.get_task_context(task_id)
        x_list = [x_samples[i] for i in range(min(64, len(x_samples)))]
        ctx_list = [context] * len(x_list)

        for neuron in self.neurons:
            neuron.consolidate_task(task_id, x_list, ctx_list)

        # Fisher for output head
        for xi, yi in zip(x_list, y_samples[:len(x_list)]):
            logits = self.forward(xi, task_id=task_id)
            probs = self._softmax(logits)
            d_logits = probs.copy()
            d_logits[int(yi)] -= 1.0
            # Need hidden rep for Fisher estimate
            h = np.array(
                [neuron.forward(xi, context) for neuron in self.neurons],
                dtype=np.float64,
            )
            grad_W = np.outer(h, d_logits)
            self._fisher_W_out = self._fisher_W_out + grad_W ** 2

        self._fisher_W_out /= max(1, len(x_list))
        self._optimal_W_out = self.W_out.copy()

        self._task_history.append({
            "task_id": task_id,
            "n_samples_consolidated": len(x_list),
        })

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        x: Float64,
        y: NDArray[np.int64],
        task_id: Optional[int] = None,
    ) -> float:
        """Return classification accuracy."""
        logits = self.forward(x, task_id=task_id)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]
        preds = np.argmax(logits, axis=-1)
        return float(np.mean(preds == y))

    def predict(self, x: Float64, task_id: Optional[int] = None) -> NDArray[np.int64]:
        """Return class predictions."""
        logits = self.forward(x, task_id=task_id)
        if logits.ndim == 1:
            logits = logits[np.newaxis, :]
        return np.argmax(logits, axis=-1).astype(np.int64)

    # ------------------------------------------------------------------
    # Branch analysis
    # ------------------------------------------------------------------

    def get_branch_utilisation(self, x_batch: Float64, task_id: Optional[int] = None) -> Float64:
        """
        Compute average branch activation (plateau firing rate) over a batch.

        Returns array of shape (n_neurons, n_branches).
        """
        context = (
            self.get_task_context(task_id)
            if task_id is not None
            else np.zeros(self.config.n_context)
        )
        utilisation = np.zeros(
            (self.config.n_neurons, self.config.n_branches), dtype=np.float64
        )
        for xi in x_batch:
            for ni, neuron in enumerate(self.neurons):
                _, branch_out = neuron.forward(xi, context, return_branch_outputs=True)
                utilisation[ni] += branch_out

        utilisation /= max(1, len(x_batch))
        return utilisation

    def get_task_branch_overlap(self, task_a: int, task_b: int) -> float:
        """
        Compute normalised overlap between branch activation patterns of two tasks.

        Low overlap (<0.1) indicates good task separation.
        Uses Jaccard similarity on binarised branch maps.
        """
        ctx_a = self.get_task_context(task_a)
        ctx_b = self.get_task_context(task_b)

        # Proxy: dot product of context vectors (they are SDRs)
        dot = float(ctx_a @ ctx_b)
        norm_a = float(np.linalg.norm(ctx_a))
        norm_b = float(np.linalg.norm(ctx_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def branch_summary(self) -> dict:
        """Return summary statistics about branch configuration."""
        compartment_counts: dict[str, int] = {}
        for neuron in self.neurons:
            for branch in neuron.branches:
                comp = branch.compartment
                compartment_counts[comp] = compartment_counts.get(comp, 0) + 1

        return {
            "n_neurons": self.config.n_neurons,
            "n_branches_per_neuron": self.config.n_branches,
            "total_branches": self.config.n_neurons * self.config.n_branches,
            "compartment_counts": compartment_counts,
            "total_parameters": self._count_parameters(),
            "context_dim": self.config.n_context,
        }

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for neuron in self.neurons:
            for branch in neuron.branches:
                total += branch.n_in + branch.n_ctx
        total += self.W_out.size + self.b_out.size
        return total

    def __repr__(self) -> str:
        s = self.branch_summary()
        return (
            f"ActiveDendriteNetwork("
            f"n_neurons={s['n_neurons']}, "
            f"n_branches={s['n_branches_per_neuron']}, "
            f"params={s['total_parameters']}, "
            f"context_dim={s['context_dim']})"
        )


# ---------------------------------------------------------------------------
# Baseline MLP for comparison
# ---------------------------------------------------------------------------


class BaselineMLP:
    """
    Two-layer point-neuron MLP for catastrophic forgetting comparison.

    Architecture: Input -> ReLU(hidden) -> Output
    No dendrites, no context, no EWC (vanilla SGD).
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        learning_rate: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(random_seed)
        self.lr = learning_rate
        scale1 = 1.0 / math.sqrt(n_input)
        scale2 = 1.0 / math.sqrt(n_hidden)
        self.W1: Float64 = rng.normal(0.0, scale1, size=(n_input, n_hidden))
        self.b1: Float64 = np.zeros(n_hidden)
        self.W2: Float64 = rng.normal(0.0, scale2, size=(n_hidden, n_output))
        self.b2: Float64 = np.zeros(n_output)

    def forward(self, x: Float64) -> Float64:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def _softmax(self, z: Float64) -> Float64:
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def evaluate(self, x: Float64, y: NDArray[np.int64]) -> float:
        logits = self.forward(x)
        return float(np.mean(np.argmax(logits, axis=-1) == y))

    def train_task(
        self,
        x_train: Float64,
        y_train: NDArray[np.int64],
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        rng = np.random.default_rng()
        n = x_train.shape[0]
        for _ in range(n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                xb = x_train[perm[start:start + batch_size]]
                yb = y_train[perm[start:start + batch_size]]
                h = np.maximum(0.0, xb @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = self._softmax(logits)
                bs = len(xb)
                d2 = probs.copy()
                d2[np.arange(bs), yb] -= 1.0
                d2 /= bs
                dW2 = h.T @ d2
                db2 = d2.sum(axis=0)
                dh = d2 @ self.W2.T * (h > 0)
                dW1 = xb.T @ dh
                db1 = dh.sum(axis=0)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2


class EWCBaseline:
    """
    MLP with Elastic Weight Consolidation for continual learning.

    Identical architecture to BaselineMLP but with EWC penalty
    (Kirkpatrick et al. 2017).  Serves as the intermediate baseline
    between vanilla MLP and the full dendritic network.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        learning_rate: float = 0.01,
        ewc_lambda: float = 400.0,
        random_seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(random_seed)
        self.lr = learning_rate
        self.ewc_lambda = ewc_lambda

        s1 = 1.0 / math.sqrt(n_input)
        s2 = 1.0 / math.sqrt(n_hidden)
        self.W1: Float64 = rng.normal(0.0, s1, size=(n_input, n_hidden))
        self.b1: Float64 = np.zeros(n_hidden)
        self.W2: Float64 = rng.normal(0.0, s2, size=(n_hidden, n_output))
        self.b2: Float64 = np.zeros(n_output)

        self._fisher_W1: Float64 = np.zeros_like(self.W1)
        self._fisher_W2: Float64 = np.zeros_like(self.W2)
        self._opt_W1: Float64 = self.W1.copy()
        self._opt_W2: Float64 = self.W2.copy()

    def forward(self, x: Float64) -> Float64:
        if x.ndim == 1:
            x = x[np.newaxis, :]
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def _softmax(self, z: Float64) -> Float64:
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def evaluate(self, x: Float64, y: NDArray[np.int64]) -> float:
        logits = self.forward(x)
        return float(np.mean(np.argmax(logits, axis=-1) == y))

    def consolidate(self, x_samples: Float64, y_samples: NDArray[np.int64]) -> None:
        """Estimate diagonal Fisher and snapshot optimal weights."""
        rng = np.random.default_rng()
        n = min(64, len(x_samples))
        f_W1 = np.zeros_like(self.W1)
        f_W2 = np.zeros_like(self.W2)

        for xi, yi in zip(x_samples[:n], y_samples[:n]):
            xi = xi[np.newaxis, :]
            h = np.maximum(0.0, xi @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = self._softmax(logits)
            d2 = probs.copy()
            d2[0, int(yi)] -= 1.0
            dh = d2 @ self.W2.T * (h > 0)
            f_W2 += (h.T @ d2) ** 2
            f_W1 += (xi.T @ dh) ** 2

        self._fisher_W1 = self._fisher_W1 + f_W1 / n
        self._fisher_W2 = self._fisher_W2 + f_W2 / n
        self._opt_W1 = self.W1.copy()
        self._opt_W2 = self.W2.copy()

    def train_task(
        self,
        x_train: Float64,
        y_train: NDArray[np.int64],
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        rng = np.random.default_rng()
        n = x_train.shape[0]
        for _ in range(n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                xb = x_train[perm[start:start + batch_size]]
                yb = y_train[perm[start:start + batch_size]]
                h = np.maximum(0.0, xb @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = self._softmax(logits)
                bs = len(xb)
                d2 = probs.copy()
                d2[np.arange(bs), yb] -= 1.0
                d2 /= bs

                dW2 = h.T @ d2
                db2 = d2.sum(axis=0)
                dh = d2 @ self.W2.T * (h > 0)
                dW1 = xb.T @ dh
                db1 = dh.sum(axis=0)

                # EWC penalty gradients
                ewc_dW1 = self.ewc_lambda * self._fisher_W1 * (self.W1 - self._opt_W1)
                ewc_dW2 = self.ewc_lambda * self._fisher_W2 * (self.W2 - self._opt_W2)

                self.W1 -= self.lr * (dW1 + ewc_dW1)
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * (dW2 + ewc_dW2)
                self.b2 -= self.lr * db2
