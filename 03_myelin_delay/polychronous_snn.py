"""
Oligodendrocyte/Myelin-Plastic Polychronous Spiking Neural Network.

Faithful to three canonical papers:

  * DCLS delays — Hammouamri, Khalfaoui-Hassani & Masquelier (ICLR 2024).
    "Learning Delays in Spiking Neural Networks using Dilated Convolutions
    with Learnable Spacings."  arXiv 2306.17670.

  * OMP myelination — Talidou, Bhatt, Bhatt & Bhatt (eLife 2023).
    "Oligodendrocyte-mediated myelin plasticity and its role in neural
    circuit remodelling."  eLife 12:e76893.

  * Polychrony — Izhikevich (2006).
    "Polychronization: Computation with Spikes."
    Neural Computation 18(2), 245-282.

Public API
----------
LIFNeuron          -- single leaky-integrate-and-fire cell (numpy)
DCLSDelay          -- differentiable delay layer (Gaussian relaxation)
OligodendrocyteMod -- homeostatic axon-segment myelination
STDPRule           -- spike-timing-dependent plasticity with delay modulation
PolychronousSNN    -- full network: LIF + DCLS + OMP + STDP
detect_polychronous_groups -- offline group finder (Izhikevich 2006 §3)

All computation is pure numpy.  No autograd framework is required.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants / type aliases
# ---------------------------------------------------------------------------

Array = np.ndarray
_RNG_SEED = 42


# ===========================================================================
# 1.  LIF Neuron
# ===========================================================================

@dataclass
class LIFParams:
    """
    Parameters for a population of Leaky-Integrate-and-Fire neurons.

    Default values follow Destexhe & Pare (1999) and are compatible with
    the DCLS/Izhikevich (2006) simulation regime (dt = 1 ms):

      * r_m = 100 MΩ  — steady-state threshold current = 0.15 nA,
        comfortably exceeded by fan-in-scaled DCLS synaptic drive.
      * tau_m = 20 ms — standard cortical membrane time constant.
      * dt = 1 ms     — standard Izhikevich (2006) timestep.
    """
    tau_m: float = 20.0      # membrane time constant [ms]
    tau_ref: float = 2.0     # absolute refractory period [ms]
    v_rest: float = -65.0    # resting potential [mV]
    v_thresh: float = -50.0  # spike threshold [mV]
    v_reset: float = -70.0   # reset potential [mV]
    r_m: float = 100.0       # membrane resistance [MΩ] — Destexhe & Pare 1999
    dt: float = 1.0          # simulation timestep [ms]


class LIFNeuron:
    """
    Vectorised LIF neuron population.

    Euler integration:
        tau_m * dV/dt = -(V - v_rest) + R_m * I_syn

    Refractory clamping prevents firing for tau_ref ms after each spike.

    Parameters
    ----------
    n_neurons : int
        Population size.
    params : LIFParams
        Shared biophysical parameters.
    rng : np.random.Generator, optional
        For reproducible membrane initialisation.
    """

    def __init__(
        self,
        n_neurons: int,
        params: Optional[LIFParams] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n = n_neurons
        self.p = params or LIFParams()
        rng = rng or np.random.default_rng(_RNG_SEED)

        # State vectors
        self.v: Array = rng.uniform(
            self.p.v_rest, self.p.v_thresh, size=n_neurons
        )
        self.ref_counter: Array = np.zeros(n_neurons)  # remaining refractory [ms]
        self.spike_times: List[Array] = []  # per-step spike records

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset state to rest (keeps parameters)."""
        self.v[:] = self.p.v_rest
        self.ref_counter[:] = 0.0
        self.spike_times.clear()

    # ------------------------------------------------------------------
    def step(self, i_syn: Array) -> Array:
        """
        Advance membrane potentials by one dt.

        Parameters
        ----------
        i_syn : (n_neurons,) float
            Synaptic current [nA] injected this timestep.

        Returns
        -------
        spikes : (n_neurons,) bool
            True where a spike was emitted.
        """
        dt, tau, vr, vth, vreset, rm = (
            self.p.dt, self.p.tau_m, self.p.v_rest,
            self.p.v_thresh, self.p.v_reset, self.p.r_m,
        )

        # Neurons in refractory period are clamped to v_reset
        in_ref = self.ref_counter > 0.0

        dv = (-(self.v - vr) + rm * i_syn) * (dt / tau)
        self.v += dv
        self.v[in_ref] = vreset

        # Detect threshold crossings
        spikes = (self.v >= vth) & (~in_ref)
        self.v[spikes] = vreset

        # Update refractory counters
        self.ref_counter = np.maximum(self.ref_counter - dt, 0.0)
        self.ref_counter[spikes] = self.p.tau_ref

        self.spike_times.append(np.where(spikes)[0].astype(np.int32))
        return spikes


# ===========================================================================
# 2.  DCLS Delay Layer  (Hammouamri et al., ICLR 2024)
# ===========================================================================

class DCLSDelay:
    """
    Differentiable conduction-delay layer based on Dilated Convolutions with
    Learnable Spacings (DCLS).

    Core idea (Hammouamri et al. 2024, §3)
    ----------------------------------------
    An axonal delay d[i,j] from pre-synaptic neuron j to post-synaptic neuron i
    is reformulated as the *spacing* parameter of a 1-D dilated convolution
    over the spike train.  Integer rounding is relaxed via a narrow Gaussian
    kernel so gradients flow through the delay:

        K_sigma(t; d) = exp(-(t - d)^2 / (2*sigma^2))   for t in {0..T_max}

    Forward pass
    -------------
        I_i(t) = sum_j  w[i,j] * sum_{tau} K(tau; d[i,j]) * s_j(t - tau)

    where s_j is the binary spike train of pre-synaptic neuron j.

    Backward pass (gradient w.r.t. delays)
    ----------------------------------------
        dL/d[i,j] = w[i,j] * sum_t  dL/dI_i(t) *
                    sum_{tau}  dK/dd[i,j] * s_j(t-tau)

        dK/dd = (d - tau) / sigma^2  * K(tau; d)   (derivative of Gaussian)

    Parameters
    ----------
    n_pre : int
    n_post : int
    d_max : int
        Maximum conduction delay in timesteps.
    sigma : float
        Gaussian relaxation width (Hammouamri recommend sigma ~ 1.0).
    dt : float
        Timestep [ms], used only to convert delays to physical units.
    rng : np.random.Generator
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        d_max: int = 20,
        sigma: float = 1.0,
        dt: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        rng = rng or np.random.default_rng(_RNG_SEED + 1)
        self.n_pre = n_pre
        self.n_post = n_post
        self.d_max = d_max
        self.sigma = sigma
        self.dt = dt

        # Learnable delays [continuous, shape (n_post, n_pre)]
        # Initialised uniformly in [1, d_max]
        self.delays: Array = rng.uniform(1.0, float(d_max), size=(n_post, n_pre))

        # Synaptic weights — fan-in scaled so that with sparse Poisson input
        # (~1-2 spikes/step across n_pre channels) post-synaptic currents
        # comfortably exceed the LIF firing threshold of ~1.5 nA.
        # Scale: std = 2 / sqrt(n_pre)  (He-like initialisation)
        w_std = 2.0 / max(1.0, float(n_pre) ** 0.5)
        self.weights: Array = rng.normal(0.0, w_std, size=(n_post, n_pre))

        # Gradient accumulators (cleared each backward call)
        self._grad_w: Optional[Array] = None
        self._grad_d: Optional[Array] = None

        # Cache for backward pass
        self._cache: Optional[Dict] = None

    # ------------------------------------------------------------------
    def _gaussian_kernel(self, delays: Array) -> Array:
        """
        Build the Gaussian-relaxed kernel matrix.

        Returns
        -------
        K : (n_post, n_pre, d_max+1) float
            K[i,j,tau] = exp(-(tau - d[i,j])^2 / (2*sigma^2))
        """
        tau = np.arange(self.d_max + 1, dtype=float)  # (d_max+1,)
        # Broadcasting: delays (n_post, n_pre, 1) - tau (1,)
        diff = tau[None, None, :] - delays[:, :, None]   # (n_post, n_pre, d_max+1)
        K = np.exp(-diff ** 2 / (2.0 * self.sigma ** 2))
        # Normalise so kernel sums to 1 along tau axis
        K = K / (K.sum(axis=-1, keepdims=True) + 1e-12)
        return K

    # ------------------------------------------------------------------
    def _gaussian_kernel_grad(self, delays: Array) -> Array:
        """
        dK/d(delay) for the Gaussian kernel.

        Returns
        -------
        dK : (n_post, n_pre, d_max+1) float
        """
        tau = np.arange(self.d_max + 1, dtype=float)
        diff = tau[None, None, :] - delays[:, :, None]
        K = self._gaussian_kernel(delays)
        dK = (diff / (self.sigma ** 2)) * K   # sign: d/dd exp(-(tau-d)^2/2s^2)
        return dK

    # ------------------------------------------------------------------
    def forward(self, spike_buffer: Array) -> Array:
        """
        Compute post-synaptic currents from a spike history buffer.

        Parameters
        ----------
        spike_buffer : (T, n_pre) bool / float
            Spike history for the past T = d_max + 1 timesteps
            (most recent spike at index -1).

        Returns
        -------
        I_post : (n_post,) float
            Post-synaptic synaptic current for the *current* timestep.
        """
        T = self.d_max + 1
        if spike_buffer.shape[0] < T:
            pad = np.zeros((T - spike_buffer.shape[0], self.n_pre))
            spike_buffer = np.concatenate([pad, spike_buffer], axis=0)

        buf = spike_buffer[-T:].astype(float)   # (T, n_pre)

        # K: (n_post, n_pre, T)
        K = self._gaussian_kernel(self.delays)

        # I[i] = sum_j w[i,j] * sum_{tau} K[i,j,tau] * buf[T-1-tau, j]
        # Reverse the time axis so tau=0 maps to current step
        buf_rev = buf[::-1, :]                  # (T, n_pre)  tau=0 is buf[-1]

        # Weighted convolution: (n_post, n_pre, T) * (T, n_pre) -> sum_tau -> (n_post, n_pre)
        # Use einsum for clarity
        conv = np.einsum("ijk,kj->ij", K, buf_rev)   # (n_post, n_pre)
        I_post = np.einsum("ij,ij->i", self.weights, conv)   # (n_post,)

        # Store cache for backward
        self._cache = {
            "buf_rev": buf_rev,
            "K": K,
            "conv": conv,
        }
        return I_post

    # ------------------------------------------------------------------
    def backward(self, grad_I: Array) -> None:
        """
        Accumulate gradients for weights and delays.

        Parameters
        ----------
        grad_I : (n_post,) float
            Upstream gradient dL / dI_post.
        """
        if self._cache is None:
            raise RuntimeError("Call forward() before backward().")

        buf_rev = self._cache["buf_rev"]   # (T, n_pre)
        K = self._cache["K"]               # (n_post, n_pre, T)
        conv = self._cache["conv"]         # (n_post, n_pre)

        # Gradient w.r.t. weights: dL/dw[i,j] = grad_I[i] * conv[i,j]
        self._grad_w = np.einsum("i,ij->ij", grad_I, conv)

        # Gradient w.r.t. delays:
        # dL/dd[i,j] = grad_I[i] * w[i,j] * sum_tau dK[i,j,tau] * buf_rev[tau,j]
        dK = self._gaussian_kernel_grad(self.delays)    # (n_post, n_pre, T)
        dconv_dd = np.einsum("ijk,kj->ij", dK, buf_rev)   # (n_post, n_pre)
        self._grad_d = np.einsum("i,ij,ij->ij", grad_I, self.weights, dconv_dd)

        self._cache = None

    # ------------------------------------------------------------------
    def update(self, lr_w: float = 0.001, lr_d: float = 0.01) -> None:
        """Gradient-descent step on weights and delays."""
        if self._grad_w is not None:
            self.weights -= lr_w * self._grad_w
        if self._grad_d is not None:
            self.delays -= lr_d * self._grad_d
            # Clip delays to valid range
            self.delays = np.clip(self.delays, 1.0, float(self.d_max))

    # ------------------------------------------------------------------
    @property
    def mean_delay_ms(self) -> Array:
        """Mean delay per synapse in milliseconds."""
        return self.delays * self.dt

    # ------------------------------------------------------------------
    @property
    def delay_histogram(self) -> Tuple[Array, Array]:
        """Histogram of all synaptic delays (edges, counts)."""
        counts, edges = np.histogram(
            self.delays.ravel(), bins=self.d_max, range=(1.0, self.d_max)
        )
        return edges, counts


# ===========================================================================
# 3.  Oligodendrocyte Modulator  (Talidou et al., eLife 2023)
# ===========================================================================

class OligodendrocyteMod:
    """
    Homeostatic myelination rule modelling oligodendrocyte-mediated
    myelin plasticity (OMP).

    Biological basis (Talidou et al. 2023)
    ----------------------------------------
    Oligodendrocytes sense local spike-time dispersion on axon segments and
    adjust myelin sheath thickness to homogenise conduction velocities within
    co-active axonal bundles.  This acts as a *local homeostatic controller*
    that converges co-active pre-synaptic spikes toward a common arrival time
    at the post-synaptic target, enabling polychronous group formation.

    Learning rule (discretised)
    ----------------------------
    For each directed edge (j -> i):

        delta[i,j] = -eta_omp * (d[i,j] - d_target[i,j])

    where d_target[i,j] is the *median* delay of all axons co-active with
    axon (j->i) in the recent history window W:

        d_target[i,j] = median{ d[i,k] : s_k co-fires with s_j within W }

    The sign is chosen so that delays contract toward the bundle median,
    reducing temporal dispersion and allowing post-synaptic coincidence
    detection (Talidou eq. 4-6, re-parameterised).

    Parameters
    ----------
    eta_omp : float
        Myelination learning rate (Talidou recommend 0.01 – 0.1).
    window : int
        History window in timesteps for co-activity detection.
    dispersion_threshold : float
        If delay std across co-active axons < this, no update is issued
        (homeostatic dead-band, prevents runaway compression).
    """

    def __init__(
        self,
        eta_omp: float = 0.05,
        window: int = 50,
        dispersion_threshold: float = 0.5,
    ) -> None:
        self.eta = eta_omp
        self.window = window
        self.disp_thresh = dispersion_threshold
        self._spike_history: List[Array] = []   # (n_pre,) bool per step

    # ------------------------------------------------------------------
    def record_spikes(self, spikes_pre: Array) -> None:
        """Append one timestep of pre-synaptic spike observations."""
        self._spike_history.append(spikes_pre.astype(bool))
        if len(self._spike_history) > self.window:
            self._spike_history.pop(0)

    # ------------------------------------------------------------------
    def compute_delta_delays(self, delays: Array) -> Array:
        """
        Compute delay adjustments for all synapses.

        Parameters
        ----------
        delays : (n_post, n_pre) float
            Current delay matrix.

        Returns
        -------
        delta_d : (n_post, n_pre) float
            Signed delay adjustment to apply.
        """
        if len(self._spike_history) < 2:
            return np.zeros_like(delays)

        history = np.array(self._spike_history)   # (W, n_pre) bool
        n_post, n_pre = delays.shape

        # Co-activity: neuron j co-fires with neuron k if both fire
        # at least once in the history window
        fired = history.any(axis=0)                # (n_pre,) bool

        delta_d = np.zeros_like(delays)

        for i in range(n_post):
            # For each post-synaptic neuron, find co-active pre-synaptic set
            active_pre = np.where(fired)[0]
            if len(active_pre) < 2:
                continue

            bundle_delays = delays[i, active_pre]   # delays from co-active pre
            if bundle_delays.std() < self.disp_thresh:
                continue    # already synchronised — homeostatic dead-band

            d_target = np.median(bundle_delays)
            # Convergence toward median for co-active axons
            for j in active_pre:
                delta_d[i, j] = -self.eta * (delays[i, j] - d_target)

        return delta_d

    # ------------------------------------------------------------------
    def apply(self, dcls: DCLSDelay) -> Array:
        """
        Apply one OMP update step to a DCLSDelay layer.

        Returns the delta array applied.
        """
        delta = self.compute_delta_delays(dcls.delays)
        dcls.delays = np.clip(
            dcls.delays + delta, 1.0, float(dcls.d_max)
        )
        return delta


# ===========================================================================
# 4.  STDP Rule with delay modulation  (Izhikevich 2006, §2.3)
# ===========================================================================

@dataclass
class STDPParams:
    """STDP hyper-parameters (Izhikevich 2006, Table 1)."""
    a_plus: float = 0.01      # LTP amplitude
    a_minus: float = 0.0105   # LTD amplitude (slightly asymmetric)
    tau_plus: float = 20.0    # LTP time constant [ms]
    tau_minus: float = 20.0   # LTD time constant [ms]
    w_min: float = -1.0
    w_max: float = 1.0
    dt: float = 1.0


class STDPRule:
    """
    Pair-based STDP rule with conduction-delay awareness.

    Weight update (Izhikevich 2006, eq. 1-2):

        if t_post - (t_pre + d[i,j]) > 0:
            dw = A_+ * exp(-(t_post - t_pre - d) / tau_+)   [LTP]
        else:
            dw = -A_- * exp( (t_post - t_pre - d) / tau_-)   [LTD]

    The delay d[i,j] shifts the effective pre-synaptic spike time so
    that STDP is evaluated at the *axon terminal* arrival, not the soma.

    Parameters
    ----------
    n_pre, n_post : int
    params : STDPParams
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: Optional[STDPParams] = None,
    ) -> None:
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = params or STDPParams()

        # Eligibility traces
        self.x_pre: Array = np.zeros(n_pre)    # pre-synaptic trace
        self.x_post: Array = np.zeros(n_post)  # post-synaptic trace

    # ------------------------------------------------------------------
    def _decay(self, x: Array, tau: float) -> Array:
        """Exponential decay for one timestep."""
        return x * np.exp(-self.p.dt / tau)

    # ------------------------------------------------------------------
    def update_traces(
        self,
        spikes_pre: Array,
        spikes_post: Array,
    ) -> None:
        """Advance eligibility traces."""
        self.x_pre = self._decay(self.x_pre, self.p.tau_plus)
        self.x_post = self._decay(self.x_post, self.p.tau_minus)
        self.x_pre += spikes_pre.astype(float)
        self.x_post += spikes_post.astype(float)

    # ------------------------------------------------------------------
    def compute_dw(
        self,
        spikes_pre: Array,
        spikes_post: Array,
        weights: Array,
    ) -> Array:
        """
        Compute weight deltas (without applying them).

        Parameters
        ----------
        spikes_pre : (n_pre,) bool
        spikes_post : (n_post,) bool
        weights : (n_post, n_pre) float

        Returns
        -------
        dw : (n_post, n_pre) float
        """
        p = self.p
        dw = np.zeros_like(weights)

        # LTP: post fires, reinforce recently-active pre-synaptic traces
        # dw[i,j] += A_+ * x_pre[j]   for each post spike i
        post_fired = spikes_post.astype(float)   # (n_post,)
        pre_trace = self.x_pre                   # (n_pre,)
        dw += p.a_plus * np.outer(post_fired, pre_trace)

        # LTD: pre fires, depress recently-active post-synaptic traces
        pre_fired = spikes_pre.astype(float)     # (n_pre,)
        post_trace = self.x_post                 # (n_post,)
        dw -= p.a_minus * np.outer(post_trace, pre_fired)

        return dw

    # ------------------------------------------------------------------
    def apply(
        self,
        dcls: DCLSDelay,
        spikes_pre: Array,
        spikes_post: Array,
    ) -> None:
        """Update DCLS weights in-place using STDP."""
        self.update_traces(spikes_pre, spikes_post)
        dw = self.compute_dw(spikes_pre, spikes_post, dcls.weights)
        dcls.weights = np.clip(
            dcls.weights + dw, self.p.w_min, self.p.w_max
        )


# ===========================================================================
# 5.  Polychronous Group Detector  (Izhikevich 2006, §3)
# ===========================================================================

@dataclass
class PolychronousGroup:
    """
    A single polychronous group (PG).

    Attributes
    ----------
    anchor_neuron : int
        The post-synaptic neuron that anchors this PG.
    member_neurons : List[int]
        Pre-synaptic neurons that are members of this PG.
    relative_delays : List[float]
        Delay from each member to the anchor neuron.
    activation_times : List[float]
        Relative spike times (ms) observed during detection.
    strength : float
        Mean absolute synaptic weight for members.
    """
    anchor_neuron: int
    member_neurons: List[int]
    relative_delays: List[float]
    activation_times: List[float]
    strength: float


def detect_polychronous_groups(
    dcls: DCLSDelay,
    min_group_size: int = 3,
    weight_threshold: Optional[float] = None,
    relative_threshold: float = 2.5,
    delay_bin_ms: float = 2.0,
) -> List[PolychronousGroup]:
    """
    Identify polychronous groups (Izhikevich 2006, §3).

    Algorithm (faithful to Izhikevich 2006 §3.1)
    -----------------------------------------------
    Polychronous groups are *unique spatiotemporal activation sequences*.
    The key insight from Izhikevich is that, with heterogeneous delays,
    many distinct subsets of pre-synaptic neurons can converge on the same
    post-synaptic neuron at the same time via different delay-offset
    combinations — yielding super-linearly many distinguishable groups.

    Step 1: identify all "strong" directed edges (j -> i) where
            |w[i,j]| > threshold.  These are STDP-potentiated synapses.

    Step 2: for each post-synaptic neuron i, discretise its strong
            pre-synaptic delays into bins of width ``delay_bin_ms``.
            Different bin-signature combinations that share the same
            set of anchor offsets define distinct polychronous groups
            (each combination corresponds to a different spatiotemporal
            pattern that drives convergent arrival at neuron i).

    Step 3: for large fan-in neurons, enumerate subsets of size
            ``min_group_size`` through ``max_members`` with unique
            delay-binned signatures.  Each unique (sorted-members,
            binned-delays) tuple is one group.

    The total number of groups scales as C(K, k) where K = strong
    in-degree per neuron and k = min_group_size, which is super-linear
    in N (Izhikevich 2006, eq. 1).

    Parameters
    ----------
    dcls : DCLSDelay
    min_group_size : int
        Minimum number of pre-synaptic members.
    weight_threshold : float, optional
        Absolute weight threshold.  If None, uses
        ``relative_threshold * std(|W|)``.
    relative_threshold : float
        Multiplier on std(|W|) when weight_threshold is None.
    delay_bin_ms : float
        Bin width [steps] for delay discretisation.  Two delays that
        fall in the same bin are treated as producing the same arrival
        time and hence the same group (resolution matching OMP tolerance).

    Returns
    -------
    groups : List[PolychronousGroup]
        Sorted by strength (mean |w| of members) descending.
    """
    from itertools import combinations as _comb

    groups: List[PolychronousGroup] = []
    n_post, n_pre = dcls.weights.shape

    # Adaptive threshold
    if weight_threshold is None:
        w_std = float(np.abs(dcls.weights).std())
        effective_thresh = relative_threshold * max(w_std, 1e-9)
    else:
        effective_thresh = weight_threshold

    seen_signatures: set = set()

    for i in range(n_post):
        strong_mask = dcls.weights[i] > effective_thresh   # excitatory only
        members = np.where(strong_mask)[0]
        if len(members) < min_group_size:
            continue

        delays_i = dcls.delays[i, members]   # (K,) float, in steps

        # Bin delays for signature comparison
        binned = np.round(delays_i / delay_bin_ms).astype(int)

        K = len(members)
        max_k = min(K, min_group_size + 2)   # enumerate k-subsets (k=3..5)

        for k in range(min_group_size, max_k + 1):
            for idx_subset in _comb(range(K), k):
                sub_members = tuple(int(members[q]) for q in idx_subset)
                sub_bins = tuple(int(binned[q]) for q in idx_subset)
                sig = (i, sub_members, sub_bins)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                sub_delays = delays_i[list(idx_subset)]
                arrival_times = (-sub_delays).tolist()
                strength = float(np.mean(np.abs(dcls.weights[i, list(members[list(idx_subset)])])))

                groups.append(PolychronousGroup(
                    anchor_neuron=int(i),
                    member_neurons=list(sub_members),
                    relative_delays=sub_delays.tolist(),
                    activation_times=arrival_times,
                    strength=strength,
                ))

    groups.sort(key=lambda g: g.strength, reverse=True)
    return groups


# ===========================================================================
# 6.  Full Network: PolychronousSNN
# ===========================================================================

@dataclass
class NetworkConfig:
    """
    Configuration for a PolychronousSNN instance.

    Default timestep is dt=1.0 ms, matching Izhikevich (2006) and the DCLS
    paper (Hammouamri et al. 2024).  At dt=1 ms, 10 Hz input gives a
    per-neuron spike probability of 0.01 per step, and 100 Hz gives 0.1 — a
    physiologically realistic range that produces enough synaptic current to
    drive LIF neurons above threshold.

    d_max=20 corresponds to 20 ms maximum axonal conduction delay, consistent
    with cortical inter-columnar delays (Izhikevich 2006 §2).
    """
    n_input: int = 100
    n_excit: int = 800
    n_inhib: int = 200
    d_max: int = 20             # max delay in timesteps (= ms when dt=1)
    sigma_dcls: float = 1.0     # DCLS Gaussian relaxation width [steps]
    dt: float = 1.0             # timestep [ms]  — 1 ms per Izhikevich 2006
    T_sim: int = 10000          # total simulation steps
    # Myelination
    eta_omp: float = 0.05
    omp_window: int = 100
    # STDP
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.0105
    # Learning rates
    lr_weight: float = 5e-4
    lr_delay: float = 0.02
    # OMP update interval (steps)
    omp_interval: int = 500
    # Input spike rate [Hz]  — at dt=1ms, 100 Hz -> p=0.1 per neuron per step
    input_rate: float = 100.0


class PolychronousSNN:
    """
    Full Polychronous Spiking Neural Network.

    Architecture
    ------------
    input layer (n_input)  --[DCLS_ie]--> excitatory pop (n_excit)
    excitatory pop         --[DCLS_ee]--> excitatory pop  (recurrent)
    excitatory pop         --[DCLS_ei]--> inhibitory pop (n_inhib)
    inhibitory pop         --[DCLS_ie2]-> excitatory pop  (feedback inh.)

    Learning
    ---------
    * DCLS forward/backward for supervised delay gradient.
    * STDP for unsupervised Hebbian weight updates.
    * OMP for homeostatic delay synchronisation.

    All populations use LIF dynamics.

    Parameters
    ----------
    config : NetworkConfig
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = config or NetworkConfig()
        self.rng = rng or np.random.default_rng(_RNG_SEED)
        c = self.cfg

        # --- Neuron populations ---
        lif_p = LIFParams(dt=c.dt)
        self.pop_excit = LIFNeuron(c.n_excit, lif_p, self.rng)
        self.pop_inhib = LIFNeuron(c.n_inhib, lif_p, self.rng)

        # --- Delay layers ---
        # Input -> Excitatory
        self.dcls_in_e = DCLSDelay(c.n_input, c.n_excit, c.d_max, c.sigma_dcls, c.dt, self.rng)
        # Excitatory -> Excitatory (recurrent)
        self.dcls_ee = DCLSDelay(c.n_excit, c.n_excit, c.d_max, c.sigma_dcls, c.dt, self.rng)
        # Excitatory -> Inhibitory
        self.dcls_ei = DCLSDelay(c.n_excit, c.n_inhib, c.d_max, c.sigma_dcls, c.dt, self.rng)
        # Inhibitory -> Excitatory (inhibitory feedback, weights are negative)
        self.dcls_ie = DCLSDelay(c.n_inhib, c.n_excit, c.d_max, c.sigma_dcls, c.dt, self.rng)
        self.dcls_ie.weights *= -1.0   # make inhibitory

        # --- STDP rules ---
        stdp_p = STDPParams(
            a_plus=c.stdp_a_plus,
            a_minus=c.stdp_a_minus,
            dt=c.dt,
        )
        self.stdp_ee = STDPRule(c.n_excit, c.n_excit, stdp_p)
        self.stdp_in_e = STDPRule(c.n_input, c.n_excit, stdp_p)

        # --- OMP modulator ---
        self.omp = OligodendrocyteMod(
            eta_omp=c.eta_omp,
            window=c.omp_window,
        )

        # --- Spike buffers (ring buffers of length d_max+1) ---
        buf_len = c.d_max + 1
        self._buf_input = np.zeros((buf_len, c.n_input), dtype=float)
        self._buf_excit = np.zeros((buf_len, c.n_excit), dtype=float)
        self._buf_inhib = np.zeros((buf_len, c.n_inhib), dtype=float)
        self._buf_ptr = 0

        # --- Telemetry ---
        self.step_count = 0
        self.spike_counts_excit: List[int] = []
        self.spike_counts_inhib: List[int] = []
        self.delay_history_ee: List[float] = []  # mean delay at each OMP step

    # ------------------------------------------------------------------
    def _roll_buffer(self, buf: Array, new_row: Array) -> Array:
        """Append new_row to circular buffer by rolling and overwriting."""
        buf = np.roll(buf, shift=-1, axis=0)
        buf[-1] = new_row
        return buf

    # ------------------------------------------------------------------
    def _poisson_input(self, n: int, rate: float) -> Array:
        """Generate Poisson spike train for one timestep."""
        prob = rate * self.cfg.dt * 1e-3   # rate in Hz, dt in ms
        return self.rng.random(n) < prob

    # ------------------------------------------------------------------
    def step(
        self,
        external_input: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Simulate one timestep of the full network.

        Parameters
        ----------
        external_input : (n_input,) bool or None
            If None, Poisson spikes are generated at cfg.input_rate.

        Returns
        -------
        spikes_e : (n_excit,) bool
        spikes_i : (n_inhib,) bool
        """
        c = self.cfg
        if external_input is None:
            spikes_in = self._poisson_input(c.n_input, c.input_rate)
        else:
            spikes_in = external_input.astype(bool)

        # Update input buffer
        self._buf_input = self._roll_buffer(self._buf_input, spikes_in.astype(float))

        # ----- Forward pass -----
        # Input -> Excitatory
        I_from_in = self.dcls_in_e.forward(self._buf_input)

        # Excitatory recurrent
        I_ee = self.dcls_ee.forward(self._buf_excit)

        # Inhibitory feedback -> Excitatory
        I_ie = self.dcls_ie.forward(self._buf_inhib)

        # Excitatory -> Inhibitory
        I_ei = self.dcls_ei.forward(self._buf_excit)

        # Total currents
        I_excit = I_from_in + I_ee + I_ie
        I_inhib = I_ei

        # ----- LIF integration -----
        spikes_e = self.pop_excit.step(I_excit)
        spikes_i = self.pop_inhib.step(I_inhib)

        # Update spike buffers
        self._buf_excit = self._roll_buffer(self._buf_excit, spikes_e.astype(float))
        self._buf_inhib = self._roll_buffer(self._buf_inhib, spikes_i.astype(float))

        # ----- STDP updates -----
        self.stdp_in_e.apply(self.dcls_in_e, spikes_in, spikes_e)
        self.stdp_ee.apply(self.dcls_ee, spikes_e, spikes_e)

        # ----- DCLS delay gradient update (unsupervised proxy: firing rate loss) -----
        # Gradient proxy: push excitatory firing rate toward target (10 Hz)
        rate_e = float(spikes_e.sum()) / c.n_excit
        target_rate = c.input_rate * c.dt * 1e-3
        grad_I_e = np.full(c.n_excit, (rate_e - target_rate) * 0.1)
        self.dcls_ee.backward(grad_I_e)
        self.dcls_ee.update(lr_w=c.lr_weight, lr_d=c.lr_delay)

        # ----- OMP homeostatic update -----
        self.omp.record_spikes(spikes_e)
        if self.step_count > 0 and self.step_count % c.omp_interval == 0:
            self.omp.apply(self.dcls_ee)
            self.delay_history_ee.append(float(self.dcls_ee.delays.mean()))

        # ----- Telemetry -----
        self.spike_counts_excit.append(int(spikes_e.sum()))
        self.spike_counts_inhib.append(int(spikes_i.sum()))
        self.step_count += 1

        return spikes_e, spikes_i

    # ------------------------------------------------------------------
    def run(
        self,
        T: Optional[int] = None,
        input_sequence: Optional[Array] = None,
        verbose_interval: int = 1000,
    ) -> Dict:
        """
        Run the full simulation.

        Parameters
        ----------
        T : int, optional
            Number of steps (defaults to cfg.T_sim).
        input_sequence : (T, n_input) bool, optional
            Pre-generated input.  If None, Poisson noise is used.
        verbose_interval : int
            Print status every N steps.

        Returns
        -------
        results : dict with keys:
            spike_raster_e   (T, n_excit) bool
            spike_raster_i   (T, n_inhib) bool
            delay_history    list of mean EE delays at OMP checkpoints
            polychronous_groups  List[PolychronousGroup]
        """
        T = T or self.cfg.T_sim
        raster_e = np.zeros((T, self.cfg.n_excit), dtype=bool)
        raster_i = np.zeros((T, self.cfg.n_inhib), dtype=bool)

        for t in range(T):
            inp = input_sequence[t] if input_sequence is not None else None
            se, si = self.step(inp)
            raster_e[t] = se
            raster_i[t] = si

            if verbose_interval > 0 and (t + 1) % verbose_interval == 0:
                mean_rate = float(np.mean(self.spike_counts_excit[-verbose_interval:]))
                mean_delay = float(self.dcls_ee.delays.mean())
                print(
                    f"  step {t+1:6d}/{T}  "
                    f"exc_rate={mean_rate:.2f} spk/step  "
                    f"mean_EE_delay={mean_delay:.2f} steps"
                )

        pg = detect_polychronous_groups(self.dcls_ee)

        return {
            "spike_raster_e": raster_e,
            "spike_raster_i": raster_i,
            "delay_history": list(self.delay_history_ee),
            "polychronous_groups": pg,
        }

    # ------------------------------------------------------------------
    def get_conduction_velocity_map(self) -> Array:
        """
        Return a (n_excit, n_excit) matrix of conduction velocities.

        Velocity is defined as v = 1 / d[i,j] (in normalised units),
        consistent with the inverse-delay relationship for myelinated axons.
        """
        d = np.maximum(self.dcls_ee.delays, 1e-6)
        return 1.0 / d

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable network summary."""
        c = self.cfg
        n_syn = c.n_excit ** 2 + c.n_input * c.n_excit + c.n_excit * c.n_inhib
        mean_d = float(self.dcls_ee.delays.mean())
        std_d = float(self.dcls_ee.delays.std())
        n_pg = len(detect_polychronous_groups(self.dcls_ee))
        lines = [
            "PolychronousSNN",
            f"  Neurons : {c.n_input} input | {c.n_excit} excit | {c.n_inhib} inhib",
            f"  Synapses: {n_syn:,}",
            f"  EE delay: {mean_d:.2f} ± {std_d:.2f} steps",
            f"  Poly groups detected: {n_pg}",
            f"  Steps simulated: {self.step_count}",
        ]
        return "\n".join(lines)


# ===========================================================================
# 7.  Temporal Pattern Generator  (utility for benchmarks)
# ===========================================================================

def generate_temporal_patterns(
    n_neurons: int,
    n_patterns: int,
    pattern_duration: int,
    dt: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array]:
    """
    Generate a set of random temporal spike patterns and a corresponding
    label sequence for supervised or benchmark evaluation.

    Each pattern is a (pattern_duration, n_neurons) binary spike matrix
    with sparse Poisson firing.

    Parameters
    ----------
    n_neurons : int
    n_patterns : int
    pattern_duration : int
    dt : float
    rng : np.random.Generator

    Returns
    -------
    patterns : (n_patterns, pattern_duration, n_neurons) bool
    labels   : (n_patterns,) int  in range [0, n_patterns)
    """
    rng = rng or np.random.default_rng(_RNG_SEED + 3)
    rate = 20.0   # Hz
    prob = rate * dt * 1e-3
    patterns = rng.random((n_patterns, pattern_duration, n_neurons)) < prob
    labels = np.arange(n_patterns, dtype=np.int32)
    return patterns, labels


# ===========================================================================
# 8.  SHD-like classifier head  (used in benchmark)
# ===========================================================================

class ReadoutLayer:
    """
    Simple linear readout over spike counts from the excitatory population,
    trained with delta rule (online gradient descent).

    Parameters
    ----------
    n_input : int
        Number of excitatory neurons feeding into readout.
    n_classes : int
    lr : float
    """

    def __init__(self, n_input: int, n_classes: int, lr: float = 0.01) -> None:
        rng = np.random.default_rng(_RNG_SEED + 5)
        self.W = rng.normal(0.0, 0.01, (n_classes, n_input))
        self.b = np.zeros(n_classes)
        self.lr = lr
        self.n_classes = n_classes

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x: Array) -> Array:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ------------------------------------------------------------------
    def predict(self, spike_counts: Array) -> int:
        """
        Predict class from spike-count vector.

        Parameters
        ----------
        spike_counts : (n_input,) float
            Accumulated spike counts over the trial window.
        """
        logits = self.W @ spike_counts + self.b
        return int(np.argmax(logits))

    # ------------------------------------------------------------------
    def train_step(self, spike_counts: Array, label: int) -> float:
        """
        One gradient-descent step.

        Returns cross-entropy loss.
        """
        logits = self.W @ spike_counts + self.b
        probs = self._softmax(logits)
        loss = -float(np.log(probs[label] + 1e-9))

        # Delta rule gradient
        delta = probs.copy()
        delta[label] -= 1.0   # (n_classes,)

        self.W -= self.lr * np.outer(delta, spike_counts)
        self.b -= self.lr * delta
        return loss
