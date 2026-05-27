#!/usr/bin/env python3
"""
Three-Factor Neuromodulated Plasticity System
=============================================

Full numpy implementation of the Fremaux-Gerstner three-factor learning rule
combined with Tadros et al. (2022) sleep-replay consolidation and Sandia (2017)
neurogenesis-as-regularization.

Mathematical foundations
------------------------
Three-factor rule (Fremaux & Gerstner, 2016):
    dw_ij(t) = eta * e_ij(t) * M(t)

    where:
      w_ij(t)  = synaptic weight from unit j to unit i
      eta      = learning rate
      e_ij(t)  = eligibility trace (synapse-local, STDP-like):
                   de/dt = -e/tau_e + x_pre(t) * x_post(t)
                 held at the synapse, bridges temporal credit gap
      M(t)     = globally broadcast neuromodulator:
                   Dopamine (DA)  = reward prediction error (RPE)
                   Acetylcholine (ACh) = attention / learning-rate gain
                   Norepinephrine (NE) = surprise / gain scaling

Sleep-replay consolidation (Tadros et al. Nature Comms 2022):
    - Freeze active-task weights
    - Binarize activations: x -> sign(x) (Heaviside approximation)
    - Apply unsupervised local Hebbian rule over noisy spontaneous inputs
    - dw_ij proportional to x_i * x_j - lambda * w_ij   (Oja-like decay)
    - Recovers forgotten tasks without rehearsal buffer

Neurogenesis-as-regularization (Sandia "Neurogenesis Deep Learning" 2017):
    - Every N update steps, identify the 5% of hidden units with lowest
      contribution to recent outputs (L1 norm of outgoing weights * activation)
    - Re-initialize those units: weights ~ N(0, sigma_init)
    - Acts as structured dropout: removes dead/saturated units
    - Prevents capacity saturation in continual learning

Combined system:
    - Forward pass with eligibility-trace accumulation
    - Neuromodulator broadcast gating weight updates
    - Periodic sleep phases for Hebbian consolidation
    - Periodic neurogenesis for capacity maintenance

References
----------
Fremaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent
  plasticity, and theory of three-factor learning rules.
  Frontiers in Neural Circuits, 9, 85.

Tadros, T. et al. (2022). Sleep-replay consolidation prevents catastrophic
  forgetting in neural networks. Nature Communications, 13, 7842.

Sandia National Laboratories (2017). Neurogenesis Deep Learning.
  arXiv:1710.06759.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Neuromodulator signals
# ---------------------------------------------------------------------------

@dataclass
class NeuromodulatorSignals:
    """
    Globally broadcast neuromodulatory signals at time t.

    Each modulator scales a different aspect of plasticity:
      DA  (dopamine)         -- reward prediction error; gates Hebbian updates
      ACh (acetylcholine)    -- attention / learning-rate gain multiplier
      NE  (norepinephrine)   -- surprise / gain scaling of activations

    All signals are non-negative reals.  Baseline = 1.0 (neutral).
    DA > 1.0 = unexpected reward; DA < 1.0 = unexpected punishment.
    ACh > 1.0 = high attention (faster learning).
    NE > 1.0 = high surprise / arousal (stronger gain).
    """
    DA: float = 1.0     # dopamine: reward prediction error
    ACh: float = 1.0    # acetylcholine: attention / learning gain
    NE: float = 1.0     # norepinephrine: surprise / gain

    def combined_modulation(self) -> float:
        """
        Scalar modulation factor M(t) used in the three-factor rule.

        M(t) = (DA - 1.0) * ACh * NE

        Interpretation:
          - DA - 1.0 converts RPE to a signed teaching signal
          - Multiplied by ACh (attention boost) and NE (surprise/gain)
          - Result can be negative (punishment), zero (neutral), or positive
        """
        return (self.DA - 1.0) * self.ACh * self.NE

    def effective_lr_gain(self) -> float:
        """ACh * NE gain factor applied to the learning rate."""
        return self.ACh * self.NE


# ---------------------------------------------------------------------------
# Eligibility traces
# ---------------------------------------------------------------------------

class EligibilityTraceBuffer:
    """
    Synapse-local eligibility trace matrix for a weight matrix W[n_out, n_in].

    The trace accumulates STDP-like pre-post correlations and decays
    exponentially.  It bridges the temporal gap between the synaptic
    activity (pre-post coincidence) and the delayed neuromodulatory signal.

    Update rule (continuous-time Euler discretization at dt=1 step):
        e(t+1) = (1 - 1/tau_e) * e(t) + x_pre(t) * x_post(t)

    The weight update fires when the neuromodulator M(t) arrives:
        dw = eta * e(t) * M(t)
    """

    def __init__(
        self,
        n_out: int,
        n_in: int,
        tau_e: float = 20.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        """
        Parameters
        ----------
        n_out : output (post-synaptic) dimension
        n_in  : input (pre-synaptic) dimension
        tau_e : eligibility trace time constant (steps).
                Biologically ~100-500ms; here 1 step = one forward pass.
        """
        self.n_out = n_out
        self.n_in = n_in
        self.tau_e = float(tau_e)
        self._decay = 1.0 - 1.0 / tau_e
        self.traces = np.zeros((n_out, n_in), dtype=dtype)

    def update(self, x_pre: np.ndarray, x_post: np.ndarray) -> None:
        """
        Record pre-post coincidence and decay existing trace.

        Parameters
        ----------
        x_pre  : pre-synaptic activity vector, shape (n_in,)
        x_post : post-synaptic activity vector, shape (n_out,)
        """
        # Outer product: correlation[i,j] = x_post[i] * x_pre[j]
        correlation = np.outer(x_post, x_pre)
        self.traces = self._decay * self.traces + correlation

    def apply_modulation(
        self,
        M: float,
        eta: float,
        weight_min: float = -5.0,
        weight_max: float = 5.0,
        W_current: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the weight delta from the three-factor rule:
            dw_ij = eta * e_ij(t) * M(t)

        Parameters
        ----------
        M         : scalar neuromodulatory signal at this timestep
        eta       : base learning rate
        weight_min, weight_max : hard clips on the returned delta
        W_current : if provided, clip so resulting weights stay in bounds

        Returns
        -------
        dw : weight update matrix, same shape as W
        """
        dw = eta * self.traces * M
        if W_current is not None:
            # Clip so w + dw stays in [weight_min, weight_max]
            dw = np.clip(dw, weight_min - W_current, weight_max - W_current)
        return dw

    def reset(self) -> None:
        """Zero-out all traces (e.g. after a context switch)."""
        self.traces.fill(0.0)

    @property
    def mean_abs(self) -> float:
        return float(np.mean(np.abs(self.traces)))

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.traces)))


# ---------------------------------------------------------------------------
# Network layer with three-factor learning
# ---------------------------------------------------------------------------

class ThreeFactorLayer:
    """
    A single fully-connected hidden layer trained by the three-factor rule.

    The layer maintains:
      - W : weight matrix [n_out, n_in]
      - b : bias vector   [n_out]
      - E : EligibilityTraceBuffer for W
      - activity history for neurogenesis contribution scoring

    Activation: tanh (smooth, bounded, biologically plausible).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_e: float = 20.0,
        eta: float = 0.01,
        weight_scale: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.eta = eta
        self.rng = rng or np.random.default_rng()

        sigma = weight_scale / np.sqrt(n_in)
        self.W = self.rng.normal(0.0, sigma, (n_out, n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.E = EligibilityTraceBuffer(n_out, n_in, tau_e=tau_e)

        # Track unit contribution for neurogenesis targeting
        self._recent_activations: list[np.ndarray] = []
        self._activation_window = 50  # steps

        # History tracking
        self._update_count = 0
        self._weight_history: list[float] = []  # mean |W| over time

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : input, shape (n_in,) or (batch, n_in)

        Returns
        -------
        h : activations, shape (n_out,) or (batch, n_out)
        """
        z = x @ self.W.T + self.b
        h = np.tanh(z)

        # Record activity for neurogenesis scoring (single sample only)
        if h.ndim == 1:
            self._recent_activations.append(np.abs(h))
            if len(self._recent_activations) > self._activation_window:
                self._recent_activations.pop(0)
        return h

    def update_traces(self, x_pre: np.ndarray, x_post: np.ndarray) -> None:
        """Update eligibility traces given pre- and post-synaptic activities."""
        self.E.update(x_pre, x_post)

    def apply_three_factor_update(
        self,
        M: float,
        L2_decay: float = 1e-4,
    ) -> float:
        """
        Apply three-factor weight update using the current eligibility trace.

        Also applies L2 weight decay (improves generalisation).

        Returns mean |dw| for monitoring.
        """
        dw = self.E.apply_modulation(M, self.eta, W_current=self.W)
        # L2 regularization term
        dw -= L2_decay * self.W
        self.W += dw.astype(np.float32)
        self._update_count += 1
        self._weight_history.append(float(np.mean(np.abs(self.W))))
        return float(np.mean(np.abs(dw)))

    def unit_contributions(self) -> np.ndarray:
        """
        Compute per-unit contribution score for neurogenesis targeting.

        Score = mean recent |activation| * L1 norm of outgoing weights.
        Units with low score are candidates for re-initialization.

        Returns
        -------
        scores : shape (n_out,), higher = more contributing
        """
        if self._recent_activations:
            mean_act = np.mean(self._recent_activations, axis=0)
        else:
            mean_act = np.ones(self.n_out)

        outgoing_l1 = np.sum(np.abs(self.W), axis=1)  # (n_out,)
        return mean_act * outgoing_l1

    def snapshot_weights(self) -> np.ndarray:
        """Return a copy of the current weight matrix."""
        return self.W.copy()

    def restore_weights(self, W_snapshot: np.ndarray) -> None:
        """Restore weights from a snapshot (used in sleep-replay)."""
        self.W = W_snapshot.astype(np.float32)


# ---------------------------------------------------------------------------
# Output layer (linear + softmax for classification)
# ---------------------------------------------------------------------------

class OutputLayer:
    """Linear output layer with cross-entropy loss for classification."""

    def __init__(
        self,
        n_in: int,
        n_classes: int,
        eta: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_in = n_in
        self.n_classes = n_classes
        self.eta = eta
        self.rng = rng or np.random.default_rng()

        sigma = 0.1 / np.sqrt(n_in)
        self.W = self.rng.normal(0.0, sigma, (n_classes, n_in)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        h : hidden activations, shape (n_in,) or (batch, n_in)

        Returns
        -------
        probs : softmax probabilities, same batch shape
        """
        z = h @ self.W.T + self.b
        z = z - np.max(z, axis=-1, keepdims=True)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def backward(self, h: np.ndarray, y_true: int) -> tuple[np.ndarray, float]:
        """
        Compute gradient and loss for a single sample.

        Returns
        -------
        dh    : gradient w.r.t. hidden activation h
        loss  : scalar cross-entropy loss
        """
        probs = self.forward(h)
        # Cross-entropy loss
        loss = -np.log(np.clip(probs[y_true], 1e-12, 1.0))

        # Gradient w.r.t. logits (softmax + CE combined)
        dz = probs.copy()
        dz[y_true] -= 1.0  # shape (n_classes,)

        # Gradient w.r.t. W
        dW = np.outer(dz, h)
        self.W -= self.eta * dW.astype(np.float32)
        self.b -= self.eta * dz.astype(np.float32)

        # Gradient w.r.t. h
        dh = dz @ self.W  # shape (n_in,)
        return dh, float(loss)

    def predict(self, h: np.ndarray) -> int:
        return int(np.argmax(self.forward(h)))


# ---------------------------------------------------------------------------
# Sleep-replay consolidation  (Tadros et al. 2022)
# ---------------------------------------------------------------------------

class SleepReplayConsolidator:
    """
    Offline Hebbian consolidation that prevents catastrophic forgetting.

    During a "sleep phase" the network is driven by spontaneous noisy inputs
    (no task-specific data).  Weights are updated by a local Hebbian rule:

        dw_ij = alpha * (x_i * x_j - lambda * w_ij)

    where x_i is the binarized (Heaviside) activation of unit i under noisy
    input.  The decay term (lambda * w_ij) is an Oja-like weight normalization
    that prevents unbounded growth and corresponds to synaptic downscaling
    during sleep (Tononi & Cirelli, 2003).

    Key design choices:
      - Activations are BINARIZED (sign) to abstract away task-specific
        magnitude differences — the network stores patterns, not values
      - Noise is Gaussian N(0, sigma_noise) applied at the input layer
      - Only the hidden layer weights are updated (output layer is frozen)
      - The consolidated weights remain close to the pre-sleep weights
        because the Oja decay keeps them bounded

    Parameters
    ----------
    alpha       : Hebbian learning rate during sleep (smaller than eta)
    lambda_decay: Oja weight decay coefficient
    sigma_noise : std of spontaneous input noise
    n_steps     : number of spontaneous replay steps per sleep phase
    """

    def __init__(
        self,
        alpha: float = 0.002,
        lambda_decay: float = 0.01,
        sigma_noise: float = 0.3,
        n_steps: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.alpha = alpha
        self.lambda_decay = lambda_decay
        self.sigma_noise = sigma_noise
        self.n_steps = n_steps
        self.rng = rng or np.random.default_rng()

        self._consolidation_history: list[dict] = []

    def consolidate(
        self,
        layer: ThreeFactorLayer,
        n_in: int,
    ) -> dict:
        """
        Run one sleep-phase consolidation on a ThreeFactorLayer.

        1. Snapshot weights before consolidation
        2. Iterate n_steps with noisy inputs:
           a. Forward pass
           b. Binarize activations
           c. Hebbian update: dw += alpha * (h_bin outer x_noise - lambda * W)
        3. Return weight drift statistics

        Parameters
        ----------
        layer : the ThreeFactorLayer to consolidate
        n_in  : input dimensionality (for generating noise)

        Returns
        -------
        stats : dict with weight drift and trace norms
        """
        W_before = layer.snapshot_weights()

        for _ in range(self.n_steps):
            # Spontaneous noisy input (no real task data)
            x_noise = self.rng.normal(0.0, self.sigma_noise, n_in).astype(np.float32)

            # Forward pass
            z = x_noise @ layer.W.T + layer.b
            h_continuous = np.tanh(z)

            # Binarize activations: Heaviside (sign)
            h_binary = np.sign(h_continuous).astype(np.float32)
            # Handle zero case: treat as +1 (Heaviside convention)
            h_binary[h_binary == 0] = 1.0

            # Local Hebbian update with Oja-style decay
            dw = self.alpha * (
                np.outer(h_binary, x_noise) - self.lambda_decay * layer.W
            )
            layer.W += dw.astype(np.float32)

        W_after = layer.snapshot_weights()
        drift = float(np.mean(np.abs(W_after - W_before)))
        max_drift = float(np.max(np.abs(W_after - W_before)))

        stats = {
            "weight_drift_mean": drift,
            "weight_drift_max": max_drift,
            "n_steps": self.n_steps,
            "mean_abs_W_before": float(np.mean(np.abs(W_before))),
            "mean_abs_W_after": float(np.mean(np.abs(W_after))),
        }
        self._consolidation_history.append(stats)
        return stats

    @property
    def history(self) -> list[dict]:
        return list(self._consolidation_history)


# ---------------------------------------------------------------------------
# Neurogenesis (Sandia 2017)
# ---------------------------------------------------------------------------

class NeurogenesisRegularizer:
    """
    Structured unit re-initialization as regularization.

    Every `period` update steps:
      1. Compute contribution score for each hidden unit
         score_i = mean_recent_activation_i * L1_norm_outgoing_weights_i
      2. Select the bottom `fraction` of units (lowest score = least useful)
      3. Re-initialize their incoming and outgoing weights from N(0, sigma_init)
      4. Zero their biases

    This acts as structured dropout:
      - Preferentially removes dead/saturated units
      - Maintains total capacity at N_hidden units
      - Better than unstructured dropout for continual learning because
        it targets units that are genuinely not contributing

    Wiring: neurogenesis should run AFTER three-factor updates and BEFORE
    sleep-replay, so the new units can be stabilized during sleep.
    """

    def __init__(
        self,
        period: int = 500,
        fraction: float = 0.05,
        sigma_init: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        period     : number of weight update steps between neurogenesis events
        fraction   : fraction of units replaced per event (Sandia: 5%)
        sigma_init : std for re-initialized weights
        """
        self.period = period
        self.fraction = fraction
        self.sigma_init = sigma_init
        self.rng = rng or np.random.default_rng()

        self._events: list[dict] = []  # log of all neurogenesis events

    def maybe_apply(
        self,
        layer: ThreeFactorLayer,
        step: int,
    ) -> Optional[dict]:
        """
        Conditionally apply neurogenesis if `step` is a multiple of `period`.

        Parameters
        ----------
        layer : ThreeFactorLayer to apply neurogenesis to
        step  : current global update step counter

        Returns
        -------
        event dict if neurogenesis fired, else None
        """
        if step == 0 or step % self.period != 0:
            return None
        return self._apply(layer, step)

    def _apply(self, layer: ThreeFactorLayer, step: int) -> dict:
        n_replace = max(1, int(layer.n_out * self.fraction))

        scores = layer.unit_contributions()
        # Select units with lowest contribution
        target_indices = np.argsort(scores)[:n_replace]

        sigma = self.sigma_init / np.sqrt(layer.n_in)

        # Re-initialize incoming weights for target units
        layer.W[target_indices, :] = self.rng.normal(
            0.0, sigma, (n_replace, layer.n_in)
        ).astype(np.float32)
        layer.b[target_indices] = 0.0

        # Reset their eligibility traces (new neuron, no history)
        layer.E.traces[target_indices, :] = 0.0

        # Clear their recent activation history contribution
        layer._recent_activations.clear()

        event = {
            "step": step,
            "n_replaced": n_replace,
            "fraction": n_replace / layer.n_out,
            "min_score_replaced": float(np.min(scores[target_indices])),
            "max_score_kept": float(np.max(scores)) if len(scores) > n_replace else 0.0,
            "target_units": target_indices.tolist(),
        }
        self._events.append(event)
        return event

    @property
    def events(self) -> list[dict]:
        return list(self._events)


# ---------------------------------------------------------------------------
# Combined three-factor + sleep + neurogenesis network
# ---------------------------------------------------------------------------

class ThreeFactorNetwork:
    """
    Full lifelong learning network combining:
      1. Three-factor neuromodulated plasticity (Fremaux & Gerstner 2016)
      2. Sleep-replay Hebbian consolidation (Tadros et al. 2022)
      3. Neurogenesis-as-regularization (Sandia 2017)

    Architecture: Input -> ThreeFactorLayer (hidden) -> OutputLayer

    Training loop
    -------------
    For each sample (x, y):
      1. Forward: h = layer.forward(x)
      2. Output backward: dh, loss = out.backward(h, y)  (gradient flows down)
      3. Compute reward prediction error (RPE) from loss signal
      4. Update neuromodulators: DA = f(RPE), ACh = f(task novelty), NE = f(loss)
      5. Update eligibility trace: E.update(x, h)
      6. Three-factor weight update: dw = eta * E * M(t)
      7. Every `sleep_period` steps: run sleep-replay consolidation
      8. Every `neuro_period` steps: run neurogenesis

    The output layer is trained with standard gradient descent (the "teacher"
    signal that drives RPE).  The hidden layer is trained only via the
    three-factor rule — no backprop gradient reaches W_hidden.

    Parameters
    ----------
    n_input         : input dimensionality
    n_hidden        : hidden layer size
    n_classes       : number of output classes
    eta_hidden      : learning rate for hidden three-factor layer
    eta_output      : learning rate for output layer
    tau_e           : eligibility trace time constant
    sleep_period    : steps between sleep-replay phases
    neuro_period    : steps between neurogenesis events
    neuro_fraction  : fraction of hidden units replaced per event
    seed            : random seed for reproducibility
    """

    def __init__(
        self,
        n_input: int = 784,
        n_hidden: int = 256,
        n_classes: int = 10,
        eta_hidden: float = 0.005,
        eta_output: float = 0.01,
        tau_e: float = 20.0,
        sleep_period: int = 500,
        neuro_period: int = 500,
        neuro_fraction: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.sleep_period = sleep_period
        self.neuro_period = neuro_period

        self.rng = np.random.default_rng(seed)

        self.hidden = ThreeFactorLayer(
            n_in=n_input,
            n_out=n_hidden,
            tau_e=tau_e,
            eta=eta_hidden,
            rng=self.rng,
        )
        self.output = OutputLayer(
            n_in=n_hidden,
            n_classes=n_classes,
            eta=eta_output,
            rng=self.rng,
        )
        self.sleep = SleepReplayConsolidator(rng=self.rng)
        self.neurogenesis = NeurogenesisRegularizer(
            period=neuro_period,
            fraction=neuro_fraction,
            rng=self.rng,
        )

        # Neuromodulator state
        self._modulator = NeuromodulatorSignals()
        self._baseline_loss = 1.0  # running estimate for RPE
        self._ema_loss_alpha = 0.01

        # Metrics
        self.step = 0
        self.losses: list[float] = []
        self.accuracies: list[float] = []
        self.modulator_history: list[dict] = []
        self.sleep_events: list[dict] = []
        self.neuro_events: list[dict] = []

    # ------------------------------------------------------------------
    # Neuromodulator computation
    # ------------------------------------------------------------------

    def _compute_modulators(
        self,
        loss: float,
        y_pred: int,
        y_true: int,
        task_id: int = 0,
    ) -> NeuromodulatorSignals:
        """
        Derive neuromodulatory signals from the current prediction.

        DA  (dopamine, RPE):
          Reward = 1 if correct, 0 if wrong.
          Predicted reward = 1 - baseline_loss (normalized).
          RPE = actual_reward - predicted_reward.
          DA  = 1.0 + RPE   (so DA > 1 = positive surprise)

        ACh (acetylcholine, attention/novelty):
          Higher loss relative to baseline -> more attention -> higher ACh.
          ACh = clip(loss / baseline_loss, 0.5, 3.0)

        NE  (norepinephrine, surprise/gain):
          Sudden loss spike relative to smooth baseline indicates surprise.
          NE  = clip(loss / (self._baseline_loss + 1e-6), 0.5, 2.0)
        """
        # Reward: binary correct/incorrect
        actual_reward = 1.0 if (y_pred == y_true) else 0.0
        predicted_reward = max(0.0, 1.0 - self._baseline_loss)
        rpe = actual_reward - predicted_reward

        DA = 1.0 + rpe                                # RPE signal
        ACh = float(np.clip(loss / (self._baseline_loss + 1e-6), 0.5, 3.0))
        NE = float(np.clip(loss / (self._baseline_loss + 1e-6), 0.5, 2.0))

        return NeuromodulatorSignals(DA=DA, ACh=ACh, NE=NE)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        x: np.ndarray,
        y: int,
        task_id: int = 0,
    ) -> dict:
        """
        Train on a single sample using the combined three-factor rule.

        Parameters
        ----------
        x       : input vector, shape (n_input,)
        y       : integer class label
        task_id : task identifier for modulator context

        Returns
        -------
        info : dict with loss, accuracy, modulator values, etc.
        """
        # 1. Forward pass
        h = self.hidden.forward(x)

        # 2. Output layer backward (gradient-based, teacher signal)
        dh, loss = self.output.backward(h, y)
        y_pred = self.output.predict(h)

        # 3. Update running loss baseline (for RPE)
        self._baseline_loss = (
            (1 - self._ema_loss_alpha) * self._baseline_loss
            + self._ema_loss_alpha * loss
        )

        # 4. Compute neuromodulators
        mods = self._compute_modulators(loss, y_pred, y, task_id)
        self._modulator = mods
        M = mods.combined_modulation()

        # Scale eta by ACh * NE (attention and arousal boost)
        effective_eta_gain = mods.effective_lr_gain()
        original_eta = self.hidden.eta
        self.hidden.eta = original_eta * effective_eta_gain

        # 5. Update eligibility trace with pre and post activity
        self.hidden.update_traces(x, h)

        # 6. Three-factor weight update: dw = eta * e * M
        mean_dw = self.hidden.apply_three_factor_update(M)

        self.hidden.eta = original_eta  # restore eta

        self.step += 1
        self.losses.append(loss)
        self.accuracies.append(float(y_pred == y))

        # Record modulator history (every 50 steps to save memory)
        if self.step % 50 == 0:
            self.modulator_history.append({
                "step": self.step,
                "DA": round(mods.DA, 4),
                "ACh": round(mods.ACh, 4),
                "NE": round(mods.NE, 4),
                "M": round(M, 4),
                "loss": round(loss, 4),
            })

        # 7. Neurogenesis check
        neuro_event = self.neurogenesis.maybe_apply(self.hidden, self.step)
        if neuro_event:
            self.neuro_events.append(neuro_event)

        # 8. Sleep-replay check
        if self.sleep_period > 0 and self.step % self.sleep_period == 0:
            sleep_stats = self.sleep.consolidate(self.hidden, self.n_input)
            sleep_stats["step"] = self.step
            self.sleep_events.append(sleep_stats)

        return {
            "step": self.step,
            "loss": loss,
            "correct": y_pred == y,
            "DA": mods.DA,
            "ACh": mods.ACh,
            "NE": mods.NE,
            "M": M,
            "mean_dw": mean_dw,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on a dataset without weight updates.

        Parameters
        ----------
        X : shape (n_samples, n_input)
        y : shape (n_samples,) integer labels

        Returns
        -------
        accuracy in [0, 1]
        """
        correct = 0
        for xi, yi in zip(X, y):
            h = self.hidden.forward(xi)
            pred = self.output.predict(h)
            correct += int(pred == int(yi))
        return correct / len(y)

    def train_on_task(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_id: int = 0,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> list[float]:
        """
        Train on a full dataset for one or more epochs.

        Returns list of per-sample losses.
        """
        all_losses = []
        indices = np.arange(len(X))
        for _ in range(epochs):
            if shuffle:
                self.rng.shuffle(indices)
            for i in indices:
                info = self.train_step(X[i], int(y[i]), task_id=task_id)
                all_losses.append(info["loss"])
        return all_losses

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a summary of current network state."""
        recent_n = min(200, len(self.losses))
        recent_acc = float(np.mean(self.accuracies[-recent_n:])) if self.accuracies else 0.0
        recent_loss = float(np.mean(self.losses[-recent_n:])) if self.losses else 0.0

        return {
            "step": self.step,
            "recent_accuracy": round(recent_acc, 4),
            "recent_loss": round(recent_loss, 4),
            "mean_abs_W_hidden": round(float(np.mean(np.abs(self.hidden.W))), 6),
            "mean_trace": round(self.hidden.E.mean_abs, 6),
            "max_trace": round(self.hidden.E.max_abs, 6),
            "n_sleep_events": len(self.sleep_events),
            "n_neuro_events": len(self.neuro_events),
            "DA": round(self._modulator.DA, 4),
            "ACh": round(self._modulator.ACh, 4),
            "NE": round(self._modulator.NE, 4),
        }


# ---------------------------------------------------------------------------
# Reward prediction error (RPE) signal utility
# ---------------------------------------------------------------------------

class RewardPredictionError:
    """
    Temporal-difference (TD) reward prediction error for generating DA signal.

    Implements a simple TD(0) estimator:
        V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
        RPE  = r + gamma * V(s') - V(s)
        DA   = 1.0 + clip(RPE / RPE_scale, -1.0, 1.0)

    Can be used standalone to provide a more principled DA signal than the
    simple binary-reward version in ThreeFactorNetwork.

    Parameters
    ----------
    n_states  : number of discrete states (tasks or context IDs)
    alpha     : TD learning rate
    gamma     : discount factor
    rpe_scale : normalization factor for RPE -> DA conversion
    """

    def __init__(
        self,
        n_states: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.9,
        rpe_scale: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.rpe_scale = rpe_scale
        self.V = np.zeros(n_states)  # value estimates per state
        self._history: list[dict] = []

    def step(self, state: int, reward: float, next_state: int) -> float:
        """
        Perform one TD update and return the DA signal.

        Parameters
        ----------
        state      : current state index
        reward     : observed scalar reward
        next_state : next state index

        Returns
        -------
        DA : dopamine signal (1.0 + normalized RPE)
        """
        rpe = reward + self.gamma * self.V[next_state] - self.V[state]
        self.V[state] += self.alpha * rpe
        DA = 1.0 + float(np.clip(rpe / self.rpe_scale, -1.0, 1.0))
        self._history.append({"state": state, "reward": reward, "rpe": rpe, "DA": DA})
        return DA

    @property
    def history(self) -> list[dict]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

class SyntheticTaskGenerator:
    """
    Generates a sequence of N synthetic binary classification tasks.

    Each task is a random linearly-separable dataset in R^n_features.
    Tasks are designed to have partial overlap (controlled by `overlap`)
    to create a realistic continual learning scenario.

    Parameters
    ----------
    n_tasks    : number of tasks in the sequence
    n_samples  : samples per task
    n_features : input dimensionality
    n_classes  : classes per task (shared across tasks for permuted-MNIST-like setup)
    overlap    : fraction of feature space shared between consecutive tasks
    seed       : random seed
    """

    def __init__(
        self,
        n_tasks: int = 10,
        n_samples: int = 500,
        n_features: int = 100,
        n_classes: int = 5,
        overlap: float = 0.3,
        seed: int = 0,
    ) -> None:
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.overlap = overlap
        self.rng = np.random.default_rng(seed)

        self._tasks: list[tuple[np.ndarray, np.ndarray]] = []
        self._prototypes: list[np.ndarray] = []
        self._generate()

    def _generate(self) -> None:
        """Generate all tasks."""
        # Base prototype in shared feature subspace
        shared_dim = int(self.n_features * self.overlap)
        shared_basis = self.rng.normal(0, 1, (self.n_classes, shared_dim))
        shared_basis /= np.linalg.norm(shared_basis, axis=1, keepdims=True) + 1e-8

        for task_id in range(self.n_tasks):
            # Task-specific feature subspace
            task_dim = self.n_features - shared_dim
            task_basis = self.rng.normal(0, 1, (self.n_classes, task_dim))
            task_basis /= np.linalg.norm(task_basis, axis=1, keepdims=True) + 1e-8

            # Class prototypes: shared component + task-specific
            prototypes = np.hstack([shared_basis, task_basis])  # (n_classes, n_features)

            X, y = [], []
            for _ in range(self.n_samples):
                c = self.rng.integers(0, self.n_classes)
                noise = self.rng.normal(0, 0.5, self.n_features)
                sample = prototypes[c] + noise
                X.append(sample.astype(np.float32))
                y.append(c)

            self._tasks.append((np.array(X), np.array(y)))
            self._prototypes.append(prototypes)

    def get_task(self, task_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for task_id."""
        return self._tasks[task_id]

    def get_all_tasks(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(self._tasks)

    @property
    def input_dim(self) -> int:
        return self.n_features

    @property
    def output_dim(self) -> int:
        return self.n_classes


# ---------------------------------------------------------------------------
# Benchmark: Three-factor vs naive SGD (forgetting baseline)
# ---------------------------------------------------------------------------

def run_forgetting_benchmark(
    n_tasks: int = 8,
    n_samples: int = 300,
    n_features: int = 100,
    n_hidden: int = 128,
    n_classes: int = 5,
    epochs_per_task: int = 2,
    seed: int = 7,
) -> dict:
    """
    Compare three-factor+sleep+neurogenesis vs naive SGD on sequential tasks.

    Both networks see the same task sequence.  After training on all tasks,
    we evaluate accuracy on all previous tasks to measure forgetting.

    Returns
    -------
    results : dict with per-task accuracy matrices for both methods
    """
    gen = SyntheticTaskGenerator(
        n_tasks=n_tasks,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )

    # Network A: three-factor + sleep + neurogenesis
    net_3f = ThreeFactorNetwork(
        n_input=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        eta_hidden=0.005,
        eta_output=0.01,
        tau_e=20.0,
        sleep_period=300,
        neuro_period=300,
        neuro_fraction=0.05,
        seed=seed,
    )

    # Network B: naive SGD (no sleep, no neurogenesis, no three-factor gating)
    # We simulate this by disabling sleep and neurogenesis, and using a large
    # constant modulation (M = 1) so updates are purely gradient-like
    net_naive = ThreeFactorNetwork(
        n_input=n_features,
        n_hidden=n_hidden,
        n_classes=n_classes,
        eta_hidden=0.005,
        eta_output=0.01,
        tau_e=20.0,
        sleep_period=0,       # no sleep
        neuro_period=10**9,   # never neurogenesis
        neuro_fraction=0.0,
        seed=seed,
    )

    acc_3f = np.zeros((n_tasks, n_tasks))    # acc_3f[after_task, eval_task]
    acc_naive = np.zeros((n_tasks, n_tasks))

    for task_id in range(n_tasks):
        X_train, y_train = gen.get_task(task_id)

        net_3f.train_on_task(X_train, y_train, task_id=task_id, epochs=epochs_per_task)
        net_naive.train_on_task(X_train, y_train, task_id=task_id, epochs=epochs_per_task)

        # Evaluate on all tasks seen so far
        for eval_id in range(task_id + 1):
            X_eval, y_eval = gen.get_task(eval_id)
            acc_3f[task_id, eval_id] = net_3f.evaluate(X_eval, y_eval)
            acc_naive[task_id, eval_id] = net_naive.evaluate(X_eval, y_eval)

    # Compute backward transfer (BWT): average forgetting on old tasks
    def bwt(acc_matrix: np.ndarray) -> float:
        """Average accuracy drop on task T after learning T+1...N."""
        total, count = 0.0, 0
        for t in range(n_tasks - 1):
            for t2 in range(t + 1, n_tasks):
                total += acc_matrix[t2, t] - acc_matrix[t, t]
                count += 1
        return total / count if count > 0 else 0.0

    # Forward transfer (FWT): how well new tasks benefit from old training
    def fwt(acc_matrix: np.ndarray, random_baseline: float = 1.0 / n_classes) -> float:
        total, count = 0.0, 0
        for t in range(1, n_tasks):
            total += acc_matrix[t - 1, t] - random_baseline
            count += 1
        return total / count if count > 0 else 0.0

    return {
        "acc_3f": acc_3f,
        "acc_naive": acc_naive,
        "bwt_3f": bwt(acc_3f),
        "bwt_naive": bwt(acc_naive),
        "fwt_3f": fwt(acc_3f),
        "fwt_naive": fwt(acc_naive),
        "final_acc_3f": float(np.mean(np.diag(acc_3f))),
        "final_acc_naive": float(np.mean(np.diag(acc_naive))),
        "net_3f": net_3f,
        "net_naive": net_naive,
        "n_tasks": n_tasks,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    print("=== Three-Factor Plasticity System: Smoke Test ===\n")

    rng = np.random.default_rng(42)

    # Small synthetic task
    n_in, n_hid, n_cls = 50, 64, 4
    X = rng.normal(0, 1, (200, n_in)).astype(np.float32)
    y = rng.integers(0, n_cls, 200)

    net = ThreeFactorNetwork(
        n_input=n_in,
        n_hidden=n_hid,
        n_classes=n_cls,
        eta_hidden=0.01,
        eta_output=0.02,
        tau_e=15.0,
        sleep_period=100,
        neuro_period=100,
        seed=0,
    )

    t0 = time.perf_counter()
    losses = net.train_on_task(X, y, task_id=0, epochs=3)
    elapsed = time.perf_counter() - t0

    acc = net.evaluate(X, y)
    stats = net.get_stats()

    print(f"Training: {len(losses)} steps in {elapsed:.3f}s")
    print(f"Final accuracy: {acc:.3f}")
    print("\nNetwork stats:")
    pprint.pprint(stats)

    print(f"\nSleep events: {len(net.sleep_events)}")
    print(f"Neurogenesis events: {len(net.neuro_events)}")

    # Eligibility trace smoke test
    et = EligibilityTraceBuffer(n_out=4, n_in=8, tau_e=10.0)
    x_pre = rng.normal(0, 1, 8).astype(np.float32)
    x_post = rng.normal(0, 1, 4).astype(np.float32)
    et.update(x_pre, x_post)
    print(f"\nTrace mean_abs after one update: {et.mean_abs:.6f}")

    dw = et.apply_modulation(M=0.5, eta=0.01)
    print(f"Delta-W mean_abs (M=0.5): {float(np.mean(np.abs(dw))):.6f}")

    print("\nSmoke test complete.")
