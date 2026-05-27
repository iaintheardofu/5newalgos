"""
Astrocyte-Modulated Tripartite-Synapse Network (NALSM)
=======================================================

Faithful implementation of:
  - Kozachkov, Kastanenka & Krotov (PNAS 2023):
      "Building transformers from neurons and astrocytes"
  - Ivanov & Michmizos (NeurIPS 2021):
      "Increasing Liquid State Machine Performance with Edge-of-Chaos Dynamics
       Organized by a Chaotic Reservoir Network"

Architecture
------------
  Neurons  : Leaky Integrate-and-Fire (LIF) with adaptive threshold
  Astrocytes: Continuous-time process units, one per K synapses
              tau_a * dc_i/dt = -c_i + sum_j(w_ij * r_j(t))
  Gating   : W_ij_eff(t) = W_ij * g(c_{a(i,j)}(t))
             g() is a sigmoid that recovers QKV-attention semantics
  Plasticity: Tripartite STDP where astrocyte Ca2+ gates weight updates

All computation uses only NumPy — no autograd / deep learning frameworks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LIFConfig:
    """Configuration for the Leaky Integrate-and-Fire neuron layer."""

    n_neurons: int = 256
    """Number of neurons in the population."""

    tau_m: float = 20.0
    """Membrane time constant in ms."""

    v_rest: float = -65.0
    """Resting membrane potential in mV."""

    v_reset: float = -70.0
    """Post-spike reset potential in mV."""

    v_thresh: float = -50.0
    """Spike threshold in mV."""

    dt: float = 1.0
    """Simulation time step in ms."""

    refrac_period: int = 5
    """Absolute refractory period in time steps."""

    adaptive_thresh: bool = True
    """Enable spike-frequency adaptation (SFA)."""

    tau_adapt: float = 100.0
    """Adaptation time constant in ms."""

    adapt_increment: float = 0.5
    """Threshold increase per spike (mV)."""

    sparsity_target: float = 0.05
    """Target fraction of neurons active per timestep (homeostatic)."""


@dataclass
class AstrocyteConfig:
    """Configuration for the astrocyte layer."""

    coverage_k: int = 8
    """Number of synapses monitored per astrocyte (coverage ratio)."""

    tau_ca: float = 200.0
    """Calcium dynamics time constant in ms (slow: 100 ms – 10 s)."""

    tau_glutamate: float = 30.0
    """Glutamate (presynaptic EMA) time constant in ms."""

    ca_threshold: float = 0.005
    """Calcium level threshold for gliotransmitter release.

    Calibrated to the operating range produced by normalised W_astro inputs:
    steady-state Ca2+ is typically 0.001–0.02 for MNIST-rate inputs.
    """

    g_max: float = 2.0
    """Maximum gating multiplier."""

    g_min: float = 0.3
    """Minimum gating multiplier (prevents silent synapses)."""

    dt: float = 1.0
    """Simulation time step in ms (must match LIFConfig.dt)."""


@dataclass
class STDPConfig:
    """Configuration for tripartite STDP learning rule."""

    tau_plus: float = 20.0
    """Pre-before-post (LTP) trace time constant in ms."""

    tau_minus: float = 20.0
    """Post-before-pre (LTD) trace time constant in ms."""

    a_plus: float = 0.01
    """LTP amplitude."""

    a_minus: float = 0.012
    """LTD amplitude (slightly asymmetric for BCM-like stability)."""

    w_max: float = 1.0
    """Maximum synaptic weight."""

    w_min: float = 0.0
    """Minimum synaptic weight."""

    ca_ltp_gate: float = 0.001
    """Astrocyte Ca2+ threshold for enabling LTP.

    Calibrated to steady-state Ca2+ operating range (~0.001–0.02).
    """

    ca_ltd_gate: float = 0.0005
    """Astrocyte Ca2+ threshold for enabling LTD."""

    learning_rate: float = 0.005
    """Global learning rate scalar."""

    dt: float = 1.0
    """Simulation time step in ms."""


@dataclass
class NetworkConfig:
    """Top-level network configuration."""

    n_input: int = 784
    """Input dimension (e.g., MNIST pixels)."""

    n_hidden: int = 1000
    """Hidden LIF neuron count."""

    n_output: int = 10
    """Readout layer size (linear classifier)."""

    lif: LIFConfig = field(default_factory=LIFConfig)
    astrocyte: AstrocyteConfig = field(default_factory=AstrocyteConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)

    presentation_time: int = 50
    """Number of timesteps per input sample."""

    dt: float = 1.0
    """Global timestep in ms."""

    seed: int = 42
    """Random seed for reproducibility."""


# ---------------------------------------------------------------------------
# LIF Neuron Layer
# ---------------------------------------------------------------------------

class LIFLayer:
    """
    Vectorised Leaky Integrate-and-Fire neuron population.

    State variables (all shape [n_neurons]):
        v          : membrane potential (mV)
        threshold  : adaptive firing threshold (mV)
        refrac     : remaining refractory steps
        spike      : binary spike indicator at current step
        trace_pre  : pre-synaptic STDP eligibility trace
        trace_post : post-synaptic STDP eligibility trace

    Update rule (Euler, dt):
        dv/dt = (v_rest - v) / tau_m + I_syn / C_m
    """

    def __init__(self, config: LIFConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        n = config.n_neurons

        # State
        self.v: np.ndarray = np.full(n, config.v_rest, dtype=np.float64)
        self.threshold: np.ndarray = np.full(n, config.v_thresh, dtype=np.float64)
        self.refrac: np.ndarray = np.zeros(n, dtype=np.int32)
        self.spike: np.ndarray = np.zeros(n, dtype=bool)

        # STDP traces
        self.trace_pre: np.ndarray = np.zeros(n, dtype=np.float64)
        self.trace_post: np.ndarray = np.zeros(n, dtype=np.float64)

        # Homeostatic activity history
        self._activity_ema: np.ndarray = np.zeros(n, dtype=np.float64)
        self._homeo_lr: float = 1e-4

        # Decay factors (precomputed)
        self._decay_m: float = np.exp(-config.dt / config.tau_m)
        self._decay_adapt: float = np.exp(-config.dt / config.tau_adapt)

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reset membrane potentials to resting state."""
        n = self.cfg.n_neurons
        self.v[:] = self.cfg.v_rest
        self.threshold[:] = self.cfg.v_thresh
        self.refrac[:] = 0
        self.spike[:] = False
        self.trace_pre[:] = 0.0
        self.trace_post[:] = 0.0
        self._activity_ema[:] = 0.0

    # ------------------------------------------------------------------
    def step(self, I_syn: np.ndarray) -> np.ndarray:
        """
        Advance the layer by one timestep.

        Parameters
        ----------
        I_syn : ndarray, shape [n_neurons]
            Total synaptic current arriving at each neuron (nA).

        Returns
        -------
        spike : ndarray of bool, shape [n_neurons]
        """
        cfg = self.cfg
        dt = cfg.dt

        # --- Refractory gate ---
        active = self.refrac <= 0

        # --- Euler membrane update (only for non-refractory neurons) ---
        dv = (cfg.v_rest - self.v) / cfg.tau_m + I_syn
        self.v += dt * dv * active

        # --- Detect spikes ---
        self.spike = active & (self.v >= self.threshold)

        # --- Reset & refractory ---
        self.v[self.spike] = cfg.v_reset
        self.refrac[self.spike] = cfg.refrac_period
        self.refrac = np.maximum(self.refrac - 1, 0)

        # --- Adaptive threshold ---
        if cfg.adaptive_thresh:
            self.threshold += (
                self._decay_adapt - 1.0
            ) * (self.threshold - cfg.v_thresh)  # decay toward base
            self.threshold[self.spike] += cfg.adapt_increment  # spike increment

        # --- STDP traces ---
        tau_p_cfg = 20.0  # default; caller overrides via STDPEngine
        tau_m_cfg = 20.0
        self.trace_pre *= np.exp(-dt / tau_p_cfg)
        self.trace_post *= np.exp(-dt / tau_m_cfg)
        self.trace_pre[self.spike] += 1.0
        self.trace_post[self.spike] += 1.0

        # --- Homeostatic activity tracking ---
        self._activity_ema += 0.001 * (self.spike.astype(float) - self._activity_ema)

        return self.spike.copy()


# ---------------------------------------------------------------------------
# Astrocyte Layer
# ---------------------------------------------------------------------------

class AstrocyteLayer:
    """
    Continuous-time astrocyte process units.

    Each astrocyte monitors K pre-synaptic neurons and integrates their
    glutamate proxy (spike-rate EMA) into a calcium-like internal state:

        tau_a * dc_i/dt = -c_i + sum_j( w_ij * r_j(t) )

    The gating function:
        g(c) = g_min + (g_max - g_min) * sigmoid(scale * (c - ca_threshold))

    recovers query-key-value attention semantics when multiple astrocytes
    compete over the same post-synaptic neuron (softmax normalisation).

    Parameters
    ----------
    n_pre : int
        Number of pre-synaptic neurons being monitored.
    n_astrocytes : int
        Number of astrocyte process units.
    config : AstrocyteConfig
    rng : np.random.Generator
    """

    def __init__(
        self,
        n_pre: int,
        n_astrocytes: int,
        config: AstrocyteConfig,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = config
        self.n_pre = n_pre
        self.n_astro = n_astrocytes
        dt = config.dt

        # Calcium state and glutamate proxy
        self.ca: np.ndarray = np.zeros(n_astrocytes, dtype=np.float64)
        self.glutamate: np.ndarray = np.zeros(n_pre, dtype=np.float64)

        # Astrocyte-to-pre-synapse coupling weights (sparse: each astrocyte
        # monitors exactly coverage_k inputs chosen uniformly)
        self.W_astro: np.ndarray = np.zeros(
            (n_astrocytes, n_pre), dtype=np.float64
        )
        k = config.coverage_k
        for i in range(n_astrocytes):
            idx = rng.choice(n_pre, size=min(k, n_pre), replace=False)
            self.W_astro[i, idx] = rng.uniform(0.5, 1.5, size=idx.size)

        # Normalise each astrocyte's input weights
        row_sums = self.W_astro.sum(axis=1, keepdims=True) + 1e-8
        self.W_astro /= row_sums

        # Precomputed decay factors
        self._decay_ca: float = np.exp(-dt / config.tau_ca)
        self._decay_glu: float = np.exp(-dt / config.tau_glutamate)

        # History buffers for visualisation
        self._ca_history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reset all astrocyte states to zero."""
        self.ca[:] = 0.0
        self.glutamate[:] = 0.0
        self._ca_history.clear()

    # ------------------------------------------------------------------
    def step(self, pre_spikes: np.ndarray, record: bool = False) -> np.ndarray:
        """
        Advance astrocyte dynamics by one timestep.

        Parameters
        ----------
        pre_spikes : ndarray of bool, shape [n_pre]
        record : bool
            If True, append current Ca2+ state to history buffer.

        Returns
        -------
        ca : ndarray, shape [n_astrocytes]
            Updated calcium levels.
        """
        cfg = self.cfg

        # Update glutamate proxy (EMA of presynaptic spike rate)
        self.glutamate = self._decay_glu * self.glutamate + (
            1.0 - self._decay_glu
        ) * pre_spikes.astype(np.float64)

        # Astrocyte calcium ODE (implicit Euler approximation)
        # tau_a * dc/dt = -c + W_astro @ r
        drive = self.W_astro @ self.glutamate  # shape [n_astro]
        self.ca = self._decay_ca * self.ca + (1.0 - self._decay_ca) * drive

        if record:
            self._ca_history.append(self.ca.copy())

        return self.ca.copy()

    # ------------------------------------------------------------------
    def gating_values(self, ca: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the gating multiplier g(c) for each astrocyte.

        g(c) = g_min + (g_max - g_min) * sigmoid(20 * (c - threshold))

        The scale=20 makes the sigmoid sharp, approaching a step function
        as in the biological Ca2+ threshold for gliotransmitter release.

        Parameters
        ----------
        ca : optional ndarray, shape [n_astrocytes]
            Use provided values; default uses self.ca.

        Returns
        -------
        g : ndarray, shape [n_astrocytes]
        """
        cfg = self.cfg
        c = self.ca if ca is None else ca
        scale = 20.0
        sig = 1.0 / (1.0 + np.exp(-scale * (c - cfg.ca_threshold)))
        return cfg.g_min + (cfg.g_max - cfg.g_min) * sig

    # ------------------------------------------------------------------
    def attention_weights(
        self, synapse_to_astrocyte: np.ndarray
    ) -> np.ndarray:
        """
        Compute softmax-normalised attention weights over post-synaptic targets.

        This recovers the QKV-attention interpretation from Kozachkov et al.
        (PNAS 2023): when astrocytes compete over the same post-synaptic
        neuron, their gating values, after softmax, implement attention.

        Parameters
        ----------
        synapse_to_astrocyte : ndarray of int, shape [n_synapses]
            Maps each synapse index to its supervising astrocyte index.

        Returns
        -------
        attn : ndarray, shape [n_astrocytes]
            Softmax-normalised gating values.
        """
        g = self.gating_values()
        # Numerically stable softmax
        g_shifted = g - g.max()
        exp_g = np.exp(g_shifted)
        return exp_g / (exp_g.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Synaptic Weight Matrix with Astrocyte Gating
# ---------------------------------------------------------------------------

class TripartiteSynapseMatrix:
    """
    Synaptic weight matrix with real-time astrocyte gating.

    Effective weight:
        W_eff[i, j] = W[i, j] * g( c_{a(i,j)} )

    where a(i,j) is the astrocyte supervising synapse (i,j).

    Parameters
    ----------
    n_pre : int
    n_post : int
    astrocyte_layer : AstrocyteLayer
    config : STDPConfig
    rng : np.random.Generator
    connection_prob : float
        Sparse connectivity probability.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        astrocyte_layer: AstrocyteLayer,
        config: STDPConfig,
        rng: np.random.Generator,
        connection_prob: float = 0.1,
    ) -> None:
        self.n_pre = n_pre
        self.n_post = n_post
        self.astro = astrocyte_layer
        self.cfg = config
        self.rng = rng

        # Raw weights — sparse initialisation
        mask = rng.random((n_post, n_pre)) < connection_prob
        self.W: np.ndarray = np.where(
            mask,
            rng.uniform(0.0, 0.3, (n_post, n_pre)),
            0.0,
        ).astype(np.float64)
        self.mask: np.ndarray = mask  # fixed topology

        # Assign each synapse (post_i, pre_j) to one astrocyte
        # Strategy: tile astrocytes uniformly across post-synaptic neurons
        n_astro = astrocyte_layer.n_astro
        # synapse_astrocyte[i, j] = astrocyte index responsible for W[i,j]
        self.synapse_astrocyte: np.ndarray = (
            np.arange(n_post * n_pre).reshape(n_post, n_pre) % n_astro
        ).astype(np.int32)

        # Weight history for visualisation
        self._w_norm_history: list[float] = []

    # ------------------------------------------------------------------
    def effective_weights(self) -> np.ndarray:
        """
        Return W_eff = W * g(c_{astrocyte}).

        The gating value for each synapse is looked up from its supervising
        astrocyte.  Shape: [n_post, n_pre].
        """
        g = self.astro.gating_values()  # [n_astro]
        gate_matrix = g[self.synapse_astrocyte]  # [n_post, n_pre]
        return self.W * gate_matrix

    # ------------------------------------------------------------------
    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Compute post-synaptic currents given pre-synaptic spikes.

        Parameters
        ----------
        pre_spikes : ndarray of bool or float, shape [n_pre]

        Returns
        -------
        I_post : ndarray, shape [n_post]
        """
        W_eff = self.effective_weights()
        return W_eff @ pre_spikes.astype(np.float64)

    # ------------------------------------------------------------------
    def stdp_update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        pre_traces: np.ndarray,
        post_traces: np.ndarray,
        ca: np.ndarray,
    ) -> None:
        """
        Tripartite STDP update gated by astrocyte Ca2+ level.

        Classical STDP weight update:
            ΔW_ltp[i,j] = A+ * pre_trace[j]   (when post_i fires)
            ΔW_ltd[i,j] = -A- * post_trace[i]  (when pre_j fires)

        Astrocyte gate:
            LTP enabled when Ca2+_{a(i,j)} > ca_ltp_gate
            LTD enabled when Ca2+_{a(i,j)} > ca_ltd_gate

        Parameters
        ----------
        pre_spikes  : bool array [n_pre]
        post_spikes : bool array [n_post]
        pre_traces  : float array [n_pre]   — pre-synaptic STDP trace
        post_traces : float array [n_post]  — post-synaptic STDP trace
        ca          : float array [n_astro] — astrocyte calcium levels
        """
        cfg = self.cfg
        lr = cfg.learning_rate

        # Ca2+ gate matrices [n_post, n_pre]
        ca_syn = ca[self.synapse_astrocyte]  # [n_post, n_pre]
        ltp_gate = (ca_syn >= cfg.ca_ltp_gate).astype(np.float64)
        ltd_gate = (ca_syn >= cfg.ca_ltd_gate).astype(np.float64)

        # LTP: post fires, pre trace available
        post_fire = post_spikes.astype(np.float64)[:, np.newaxis]  # [n_post, 1]
        pre_tr = pre_traces[np.newaxis, :]  # [1, n_pre]
        dW_ltp = lr * cfg.a_plus * post_fire * pre_tr * ltp_gate

        # LTD: pre fires, post trace available
        pre_fire = pre_spikes.astype(np.float64)[np.newaxis, :]  # [1, n_pre]
        post_tr = post_traces[:, np.newaxis]  # [n_post, 1]
        dW_ltd = -lr * cfg.a_minus * pre_fire * post_tr * ltd_gate

        # Apply update only to existing connections
        delta = (dW_ltp + dW_ltd) * self.mask
        self.W = np.clip(self.W + delta, cfg.w_min, cfg.w_max)

        # Track weight norm for convergence monitoring
        self._w_norm_history.append(float(np.linalg.norm(self.W)))


# ---------------------------------------------------------------------------
# Readout Layer (supervised, trained offline after unsupervised pre-training)
# ---------------------------------------------------------------------------

class LinearReadout:
    """
    Simple linear readout trained with least-squares (offline).

    Used for classification after unsupervised representation learning
    in the tripartite SNN.  Follows Liquid State Machine convention.

    Parameters
    ----------
    n_in : int
        Dimensionality of the SNN population activity vector.
    n_out : int
        Number of output classes.
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearReadout":
        """
        Fit readout weights via closed-form least squares (ridge regression).

        Parameters
        ----------
        X : ndarray, shape [n_samples, n_in]
            Spike-rate activity vectors from SNN hidden layer.
        y : ndarray of int, shape [n_samples]
            Class labels.

        Returns
        -------
        self
        """
        n_classes = self.n_out
        n_samples = X.shape[0]

        # One-hot encode labels
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y[np.arange(n_samples), y] = 1.0

        # Ridge regression: (X^T X + λI)^{-1} X^T Y
        lam = 1e-4
        A = X.T @ X + lam * np.eye(self.n_in)
        b_vec = X.T @ Y
        W_combined = np.linalg.solve(A, b_vec)  # [n_in, n_classes]

        self.W = W_combined
        self.b = np.zeros(n_classes, dtype=np.float64)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : ndarray, shape [n_samples, n_in]

        Returns
        -------
        labels : ndarray of int, shape [n_samples]
        """
        if self.W is None:
            raise RuntimeError("LinearReadout has not been fitted.")
        logits = X @ self.W + self.b  # [n_samples, n_classes]
        return np.argmax(logits, axis=1)

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy in [0, 1]."""
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Full Tripartite SNN Network
# ---------------------------------------------------------------------------

class TripartiteNetwork:
    """
    Full Astrocyte-Modulated Tripartite-Synapse Network (NALSM).

    Combines:
        - Input encoding (rate or temporal)
        - LIF hidden neuron population
        - Astrocyte layer monitoring input synapses
        - Tripartite synaptic weight matrix (input → hidden)
        - Recurrent synapses within hidden layer
        - Tripartite STDP unsupervised learning
        - Linear readout (offline supervised)

    Parameters
    ----------
    config : NetworkConfig
    """

    def __init__(self, config: NetworkConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        n_in = config.n_input
        n_hid = config.n_hidden
        n_out = config.n_output
        lif_cfg = config.lif
        lif_cfg.n_neurons = n_hid
        astro_cfg = config.astrocyte
        stdp_cfg = config.stdp

        # Number of astrocytes = ceil(n_input / coverage_k)
        n_astro_fwd = max(1, n_in // astro_cfg.coverage_k)
        n_astro_rec = max(1, n_hid // astro_cfg.coverage_k)

        # Layers
        self.hidden = LIFLayer(lif_cfg, self.rng)

        self.astro_fwd = AstrocyteLayer(n_in, n_astro_fwd, astro_cfg, self.rng)
        self.astro_rec = AstrocyteLayer(n_hid, n_astro_rec, astro_cfg, self.rng)

        self.W_fwd = TripartiteSynapseMatrix(
            n_in, n_hid, self.astro_fwd, stdp_cfg, self.rng,
            connection_prob=0.15,
        )
        self.W_rec = TripartiteSynapseMatrix(
            n_hid, n_hid, self.astro_rec, stdp_cfg, self.rng,
            connection_prob=0.05,
        )

        self.readout = LinearReadout(n_hid, n_out)

        # Input-side STDP traces — tracks pre-synaptic spikes from the
        # *input* layer (shape [n_input]); distinct from hidden.trace_pre
        # which tracks hidden neuron spikes (shape [n_hidden]).
        self._input_trace: np.ndarray = np.zeros(n_in, dtype=np.float64)
        self._stdp_tau_pre: float = stdp_cfg.tau_plus
        self._stdp_tau_post: float = stdp_cfg.tau_minus
        self._input_trace_decay: float = np.exp(-config.dt / stdp_cfg.tau_plus)

        # Statistics
        self.total_steps: int = 0
        self.n_samples_processed: int = 0

    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reset all dynamic state (between samples)."""
        self.hidden.reset_state()
        self.astro_fwd.reset_state()
        self.astro_rec.reset_state()
        self._input_trace[:] = 0.0

    # ------------------------------------------------------------------
    def encode_input(self, x: np.ndarray, t: int) -> np.ndarray:
        """
        Rate-code encoding: convert pixel intensities to Poisson spike trains.

        Parameters
        ----------
        x : ndarray, shape [n_input], values in [0, 1]
        t : int
            Current timestep within the presentation window (unused for rate code).

        Returns
        -------
        spikes : bool ndarray, shape [n_input]
        """
        return self.rng.random(self.cfg.n_input) < x

    # ------------------------------------------------------------------
    def step(
        self,
        input_spikes: np.ndarray,
        learn: bool = True,
        record: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single timestep simulation.

        Parameters
        ----------
        input_spikes : bool ndarray, shape [n_input]
        learn : bool
            Apply tripartite STDP if True.
        record : bool
            Record Ca2+ history.

        Returns
        -------
        hidden_spikes : bool ndarray, shape [n_hidden]
        ca_fwd : float ndarray, shape [n_astro_fwd]
        """
        # 1. Advance astrocyte layers
        ca_fwd = self.astro_fwd.step(input_spikes, record=record)
        ca_rec = self.astro_rec.step(self.hidden.spike, record=record)

        # 2. Compute synaptic currents (astrocyte-gated)
        I_fwd = self.W_fwd.forward(input_spikes)
        I_rec = self.W_rec.forward(self.hidden.spike)
        I_total = I_fwd + 0.3 * I_rec  # recurrent gain

        # 3. Advance neuron layer
        hidden_spikes = self.hidden.step(I_total)

        # 4. Update input-side STDP trace (decays + increments on input spikes)
        self._input_trace *= self._input_trace_decay
        self._input_trace += input_spikes.astype(np.float64)

        # 5. Tripartite STDP
        if learn:
            # Forward synapse: pre=input neurons, post=hidden neurons
            self.W_fwd.stdp_update(
                input_spikes,
                hidden_spikes,
                self._input_trace,       # pre-trace has shape [n_input]
                self.hidden.trace_post,  # post-trace has shape [n_hidden]
                ca_fwd,
            )
            # Recurrent synapse: pre=hidden (prev step), post=hidden (now)
            self.W_rec.stdp_update(
                self.hidden.spike,       # previous hidden spike (before step)
                hidden_spikes,
                self.hidden.trace_pre,   # pre-trace shape [n_hidden]
                self.hidden.trace_post,  # post-trace shape [n_hidden]
                ca_rec,
            )

        self.total_steps += 1
        return hidden_spikes, ca_fwd

    # ------------------------------------------------------------------
    def run_sample(
        self,
        x: np.ndarray,
        learn: bool = True,
        record: bool = False,
    ) -> np.ndarray:
        """
        Present one input sample for `presentation_time` steps and collect
        the population spike-rate vector for readout.

        Parameters
        ----------
        x : ndarray, shape [n_input], values in [0, 1]
        learn : bool
        record : bool

        Returns
        -------
        rate_vec : ndarray, shape [n_hidden]
            Mean firing rate per neuron over the presentation window.
        """
        T = self.cfg.presentation_time
        spike_accum = np.zeros(self.cfg.n_hidden, dtype=np.float64)

        self.reset_state()
        for t in range(T):
            in_spikes = self.encode_input(x, t)
            hid_spikes, _ = self.step(in_spikes, learn=learn, record=record)
            spike_accum += hid_spikes.astype(np.float64)

        self.n_samples_processed += 1
        return spike_accum / T  # mean rate

    # ------------------------------------------------------------------
    def train_unsupervised(
        self,
        X_train: np.ndarray,
        verbose: bool = True,
        record_interval: int = 500,
    ) -> np.ndarray:
        """
        Unsupervised training phase: run all training samples through the
        network with STDP active.  Collect spike-rate representations.

        Parameters
        ----------
        X_train : ndarray, shape [n_samples, n_input], values in [0, 1]
        verbose : bool
        record_interval : int
            Log progress every this many samples.

        Returns
        -------
        representations : ndarray, shape [n_samples, n_hidden]
        """
        n = X_train.shape[0]
        representations = np.zeros((n, self.cfg.n_hidden), dtype=np.float64)

        t_start = time.perf_counter()
        for i in range(n):
            representations[i] = self.run_sample(X_train[i], learn=True)
            if verbose and (i + 1) % record_interval == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [unsupervised] {i+1}/{n} samples | "
                    f"elapsed {elapsed:.1f}s | "
                    f"W_fwd norm {self.W_fwd._w_norm_history[-1]:.4f}"
                )

        return representations

    # ------------------------------------------------------------------
    def extract_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Run samples through the network without learning to collect
        spike-rate representations.

        Parameters
        ----------
        X : ndarray, shape [n_samples, n_input]

        Returns
        -------
        representations : ndarray, shape [n_samples, n_hidden]
        """
        n = X.shape[0]
        reps = np.zeros((n, self.cfg.n_hidden), dtype=np.float64)
        for i in range(n):
            reps[i] = self.run_sample(X[i], learn=False)
        return reps

    # ------------------------------------------------------------------
    def fit_readout(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> "TripartiteNetwork":
        """
        Fit the linear readout after unsupervised feature learning.

        Parameters
        ----------
        X_train : ndarray, shape [n_samples, n_input]
        y_train : ndarray of int, shape [n_samples]

        Returns
        -------
        self
        """
        reps = self.extract_representations(X_train)
        self.readout.fit(reps, y_train)
        return self

    # ------------------------------------------------------------------
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate classification accuracy on test data.

        Parameters
        ----------
        X_test : ndarray, shape [n_samples, n_input]
        y_test : ndarray of int, shape [n_samples]

        Returns
        -------
        accuracy : float in [0, 1]
        """
        reps = self.extract_representations(X_test)
        return self.readout.score(reps, y_test)

    # ------------------------------------------------------------------
    def parameter_count(self) -> dict[str, int]:
        """Return a breakdown of learnable parameters."""
        return {
            "W_fwd": int(self.W_fwd.mask.sum()),
            "W_rec": int(self.W_rec.mask.sum()),
            "W_astro_fwd": int(self.astro_fwd.W_astro.size),
            "W_astro_rec": int(self.astro_rec.W_astro.size),
            "readout": int(self.readout.n_in * self.readout.n_out),
        }

    # ------------------------------------------------------------------
    def total_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(self.parameter_count().values())

    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """Save network weights to a compressed numpy archive."""
        np.savez_compressed(
            path,
            W_fwd=self.W_fwd.W,
            W_rec=self.W_rec.W,
            W_astro_fwd=self.astro_fwd.W_astro,
            W_astro_rec=self.astro_rec.W_astro,
            readout_W=self.readout.W if self.readout.W is not None else np.array([]),
            readout_b=self.readout.b if self.readout.b is not None else np.array([]),
        )

    # ------------------------------------------------------------------
    def load_weights(self, path: str) -> None:
        """Load network weights from a compressed numpy archive."""
        data = np.load(path)
        self.W_fwd.W = data["W_fwd"]
        self.W_rec.W = data["W_rec"]
        self.astro_fwd.W_astro = data["W_astro_fwd"]
        self.astro_rec.W_astro = data["W_astro_rec"]
        if data["readout_W"].size > 0:
            self.readout.W = data["readout_W"]
            self.readout.b = data["readout_b"]


# ---------------------------------------------------------------------------
# Utility: synthetic MNIST-like data generator
# ---------------------------------------------------------------------------

def make_synthetic_mnist(
    n_train: int = 5000,
    n_test: int = 1000,
    n_classes: int = 10,
    image_size: int = 784,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic MNIST-like data for testing without torchvision.

    Each class is defined by a random prototype image; samples are noisy
    versions of their prototype (realistic enough for benchmarking).

    Parameters
    ----------
    n_train, n_test, n_classes, image_size, seed

    Returns
    -------
    X_train, y_train, X_test, y_test
        X arrays have dtype float64 with values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Class prototypes — sparse (like digits)
    prototypes = np.zeros((n_classes, image_size), dtype=np.float64)
    for c in range(n_classes):
        # Activate ~15% of pixels with class-specific pattern
        active = rng.choice(image_size, size=image_size // 7, replace=False)
        prototypes[c, active] = rng.uniform(0.6, 1.0, size=active.size)

    def _make_split(n: int) -> Tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(0, n_classes, size=n)
        X = prototypes[labels] + rng.normal(0, 0.1, (n, image_size))
        X = np.clip(X, 0.0, 1.0)
        return X.astype(np.float64), labels.astype(np.int32)

    X_train, y_train = _make_split(n_train)
    X_test, y_test = _make_split(n_test)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Tripartite Network self-test (synthetic data, small scale)")

    cfg = NetworkConfig(
        n_input=784,
        n_hidden=200,
        n_output=10,
        lif=LIFConfig(n_neurons=200, adaptive_thresh=True),
        astrocyte=AstrocyteConfig(coverage_k=8),
        stdp=STDPConfig(learning_rate=0.005),
        presentation_time=20,
        seed=0,
    )

    net = TripartiteNetwork(cfg)
    print(f"Parameters: {net.parameter_count()}")
    print(f"Total: {net.total_parameters():,}")

    X_tr, y_tr, X_te, y_te = make_synthetic_mnist(500, 100, seed=0)
    print("Running unsupervised training on 500 samples...")
    reps = net.train_unsupervised(X_tr, verbose=True, record_interval=100)
    net.readout.fit(reps, y_tr)

    acc = net.score(X_te, y_te)
    print(f"Test accuracy: {acc:.3f}")
