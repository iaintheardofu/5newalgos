"""
Hierarchical Active Inference / Predictive-Coding Agents
=========================================================
Faithful implementation of Friston's Free Energy Principle with:
  - Hierarchical generative model: p(o, s, pi)
  - Variational free energy minimization: F = E_q[ln q - ln p]
  - Expected free energy action selection: G(pi) decomposed into
      epistemic value (information gain) + pragmatic value (preference satisfaction)
  - Canonical microcircuit mapping (Bastos et al. 2012):
      superficial pyramidals -> prediction errors
      deep pyramidals        -> expectations (predictions)
      precision gating via neuromodulators
  - Predictive coding: L5/6 top-down predictions, L2/3 bottom-up errors
  - Configurable hierarchy depth

References
----------
Friston KJ (2010) The free-energy principle: a unified brain theory? Nat Rev Neurosci.
Friston KJ et al. (2017) Active inference and epistemic value. Cogn Neurodyn.
Bastos AM et al. (2012) Canonical microcircuits for predictive coding. Neuron.
Parr T, Friston KJ (2019) Generalised free energy and active inference. Biol Cybern.
VERSES AI / Genius benchmark (2023) — 140x sample-efficiency vs PPO on Mastermind.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

_EPS = 1e-16


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (exp_x.sum(axis=axis, keepdims=True) + _EPS)


def _log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, None))


def _kl_dirichlet_categorical(q: np.ndarray, p: np.ndarray) -> float:
    """KL[q || p] for categorical distributions."""
    q = np.clip(q, _EPS, None)
    p = np.clip(p, _EPS, None)
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * (_log(q) - _log(p))))


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, _EPS, None)
    p = p / p.sum()
    return float(-np.sum(p * _log(p)))


# ---------------------------------------------------------------------------
# Precision (neuromodulatory gating)
# ---------------------------------------------------------------------------

@dataclass
class PrecisionModule:
    """
    Implements neuromodulatory precision weighting.

    In the canonical microcircuit, precision (inverse variance / beta in
    Dirichlet sense) gates how strongly prediction errors update beliefs.
    Maps to acetylcholine (ACh) for state precision, dopamine (DA) for
    policy precision.

    Parameters
    ----------
    state_precision : float
        Initial precision on hidden-state beliefs (ACh proxy).
    policy_precision : float
        Initial precision on policy beliefs (DA proxy).
    learning_rate : float
        Rate at which precision adapts based on surprise.
    """

    state_precision: float = 2.0
    policy_precision: float = 4.0
    learning_rate: float = 0.05
    _history: List[Tuple[float, float]] = field(default_factory=list, repr=False)

    def update(self, state_surprise: float, policy_surprise: float) -> None:
        """Adapt precision inversely proportional to surprise (homeostatic)."""
        self.state_precision = max(
            0.1,
            self.state_precision - self.learning_rate * (state_surprise - 1.0),
        )
        self.policy_precision = max(
            0.1,
            self.policy_precision - self.learning_rate * (policy_surprise - 1.0),
        )
        self._history.append((self.state_precision, self.policy_precision))

    @property
    def history(self) -> np.ndarray:
        return np.array(self._history)


# ---------------------------------------------------------------------------
# Canonical Microcircuit Layer (Bastos et al. 2012)
# ---------------------------------------------------------------------------

@dataclass
class CorticalLayer:
    """
    Single cortical layer implementing the canonical microcircuit.

    Bastos et al. 2012 mapping:
      - L2/3 superficial pyramidals  -> prediction errors (epsilon)
      - L5/6 deep pyramidals         -> expectations / predictions (mu)
      - L4 stellate cells            -> receive top-down predictions
      - Interneurons                 -> precision weighting

    In predictive coding:
      epsilon  = o - A @ mu_state   (bottom-up error signal)
      mu_state updates in direction of epsilon, scaled by precision
    """

    n_obs: int
    n_states: int
    precision: float = 1.0

    def __post_init__(self) -> None:
        # A: likelihood mapping  p(o | s),  shape (n_obs, n_states)
        rng = np.random.default_rng(42)
        raw = rng.dirichlet(np.ones(self.n_obs), size=self.n_states).T
        self.A: np.ndarray = raw  # (n_obs, n_states)

        # D: prior over initial states, shape (n_states,)
        self.D: np.ndarray = np.ones(self.n_states) / self.n_states

        # mu: current expectation (deep pyramidals), shape (n_states,)
        self.mu: np.ndarray = self.D.copy()

        # epsilon: prediction error (superficial pyramidals), shape (n_obs,)
        self.epsilon: np.ndarray = np.zeros(self.n_obs)

        # History for visualization
        self._mu_history: List[np.ndarray] = []
        self._epsilon_history: List[np.ndarray] = []
        self._fe_history: List[float] = []

    # ------------------------------------------------------------------
    # L5/6: top-down prediction
    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        """Deep pyramidals generate top-down prediction of observations."""
        return self.A @ self.mu  # (n_obs,)

    # ------------------------------------------------------------------
    # L2/3: prediction error
    # ------------------------------------------------------------------
    def compute_error(self, observation: np.ndarray) -> np.ndarray:
        """Superficial pyramidals encode mismatch between prediction and input."""
        pred = self.predict()
        self.epsilon = observation - pred
        self._epsilon_history.append(self.epsilon.copy())
        return self.epsilon

    # ------------------------------------------------------------------
    # Belief update (variational Bayes, mean-field)
    # ------------------------------------------------------------------
    def update_beliefs(
        self,
        observation: np.ndarray,
        top_down: Optional[np.ndarray] = None,
        n_iter: int = 16,
    ) -> np.ndarray:
        """
        Minimise variational free energy F = E_q[ln q(s) - ln p(o,s)]
        via gradient descent on mu (mean-field VI).

        F ≈ -ln p(o | mu) + KL[q(s) || p(s)]

        With top-down: F also includes error from higher level.
        """
        obs = np.asarray(observation, dtype=float)
        lr = 0.1 * self.precision

        for _ in range(n_iter):
            # Likelihood gradient: dF/d_mu = -A^T (o - A mu)  (Gaussian approx)
            pred_error = obs - self.A @ self.mu
            grad_likelihood = -self.A.T @ pred_error  # (n_states,)

            # KL gradient: dF/d_mu = mu - D  (Gaussian prior)
            grad_prior = self.mu - self.D

            # Top-down prediction error from higher cortical level
            grad_top_down = np.zeros(self.n_states)
            if top_down is not None:
                td = np.asarray(top_down, dtype=float)
                # Map top-down prediction to state space via A^T
                grad_top_down = self.A.T @ (self.A @ self.mu - td)

            total_grad = grad_likelihood + grad_prior + grad_top_down
            self.mu = self.mu - lr * total_grad
            # Keep mu as valid probability via softmax
            self.mu = _softmax(self.mu)

        self.compute_error(obs)
        self._mu_history.append(self.mu.copy())
        fe = self._variational_free_energy(obs)
        self._fe_history.append(fe)
        return self.mu

    def _variational_free_energy(self, observation: np.ndarray) -> float:
        """
        F = E_q[ln q(s)] - E_q[ln p(o|s)] - E_q[ln p(s)]
          = KL[q(s)||p(s)] - ln p(o|s=mu)    (mean-field approximation)
        """
        likelihood = float(np.dot(_log(self.A @ self.mu), observation)
                           if observation.sum() > 0 else 0.0)
        kl_prior = _kl_dirichlet_categorical(self.mu, self.D)
        return kl_prior - likelihood

    @property
    def fe_history(self) -> np.ndarray:
        return np.array(self._fe_history)

    @property
    def mu_history(self) -> np.ndarray:
        return np.array(self._mu_history) if self._mu_history else np.empty((0, self.n_states))

    @property
    def epsilon_history(self) -> np.ndarray:
        return np.array(self._epsilon_history) if self._epsilon_history else np.empty((0, self.n_obs))


# ---------------------------------------------------------------------------
# Transition model B
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    State-transition likelihood p(s_t | s_{t-1}, a_t).

    B[a] is an (n_states, n_states) column-stochastic matrix:
      B[a][:, s] = p(s_next | s_curr=s, action=a)
    """

    def __init__(self, n_states: int, n_actions: int, rng_seed: int = 0) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        rng = np.random.default_rng(rng_seed)
        self.B: np.ndarray = np.stack(
            [rng.dirichlet(np.ones(n_states) * 2.0, size=n_states).T
             for _ in range(n_actions)],
            axis=0,
        )  # (n_actions, n_states, n_states)

    def predict_next(self, belief: np.ndarray, action: int) -> np.ndarray:
        """p(s_{t+1} | q(s_t), a_t) = B[a] @ q(s_t)."""
        return self.B[action] @ belief

    def set_transition(self, action: int, matrix: np.ndarray) -> None:
        """Override with known transition structure (e.g., grid world)."""
        assert matrix.shape == (self.n_states, self.n_states)
        self.B[action] = matrix


# ---------------------------------------------------------------------------
# Preference model C (pragmatic value)
# ---------------------------------------------------------------------------

@dataclass
class PreferenceModel:
    """
    Encodes preferences over observations: ln p̃(o) — the agent's goals.

    C is a log-preference vector over observations.  Positive entries are
    preferred outcomes; negative entries are aversive outcomes.
    """

    n_obs: int
    C: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.C = np.zeros(self.n_obs)

    def set_preference(self, obs_index: int, value: float) -> None:
        """Assign log-preference to a particular observation."""
        self.C[obs_index] = value

    def pragmatic_value(self, predicted_obs: np.ndarray) -> float:
        """
        Pragmatic value = E_q[ln p̃(o)] = sum_o q(o) * C(o).
        Higher is better (agent prefers observations aligned with C).
        """
        q_obs = np.clip(predicted_obs, _EPS, None)
        q_obs /= q_obs.sum()
        return float(np.dot(q_obs, self.C))

    def epistemic_value(
        self,
        predicted_obs: np.ndarray,
        A: np.ndarray,
        q_s: np.ndarray,
    ) -> float:
        """
        Epistemic value (information gain / salience):
          W = H[p(o|pi)] - E_{q(s|pi)}[H[p(o|s)]]
            = mutual information between o and s under pi.

        Higher epistemic value -> more informative action.
        """
        # Marginal: p(o|pi) = A @ q(s|pi)
        p_o_pi = A @ q_s
        p_o_pi = np.clip(p_o_pi, _EPS, None)
        p_o_pi /= p_o_pi.sum()

        h_marginal = _entropy(p_o_pi)

        # Expected conditional entropy: E_{q(s)}[H[p(o|s)]]
        h_conditional = 0.0
        for i, prob_s in enumerate(q_s):
            h_conditional += prob_s * _entropy(A[:, i])

        return float(h_marginal - h_conditional)


# ---------------------------------------------------------------------------
# Policy / Action selection via Expected Free Energy G(pi)
# ---------------------------------------------------------------------------

@dataclass
class PolicyModel:
    """
    Active-inference policy selection via expected free energy (EFE).

    G(pi) = E_{q(o,s|pi)}[ln q(s|pi) - ln p̃(o,s|pi)]
           = -epistemic_value(pi) - pragmatic_value(pi)

    Optimal policy: pi* = softmax(-G) / sum_pi softmax(-G)

    The decomposition makes the exploration-exploitation trade-off explicit:
      - epistemic value  -> explore (reduce uncertainty)
      - pragmatic value  -> exploit (satisfy preferences)
    """

    n_actions: int
    n_policies: int
    policy_horizon: int
    n_states: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1)
        # Each policy is a sequence of actions of length policy_horizon
        self.policies: np.ndarray = rng.integers(
            0, self.n_actions, size=(self.n_policies, self.policy_horizon)
        )
        self.G: np.ndarray = np.zeros(self.n_policies)
        self.q_pi: np.ndarray = np.ones(self.n_policies) / self.n_policies
        self._G_history: List[np.ndarray] = []
        self._q_pi_history: List[np.ndarray] = []
        self._epistemic_history: List[float] = []
        self._pragmatic_history: List[float] = []

    def evaluate_policies(
        self,
        current_belief: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        preferences: PreferenceModel,
        precision: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """
        Evaluate all policies, compute G(pi) for each.

        Returns
        -------
        action : int
            Selected action index for the first step of the optimal policy.
        q_pi : np.ndarray
            Posterior belief over policies.
        """
        epistemic_vals = np.zeros(self.n_policies)
        pragmatic_vals = np.zeros(self.n_policies)

        for i, policy in enumerate(self.policies):
            q_s = current_belief.copy()
            G_pi = 0.0
            epi = 0.0
            prag = 0.0

            for action in policy:
                # Predicted next state
                q_s_next = B[action] @ q_s
                # Predicted observation under next state
                pred_obs = A @ q_s_next

                # Epistemic value (information gain)
                e_val = preferences.epistemic_value(pred_obs, A, q_s_next)
                # Pragmatic value (preference satisfaction)
                p_val = preferences.pragmatic_value(pred_obs)

                G_pi += -e_val - p_val  # EFE is negative of both values
                epi += e_val
                prag += p_val
                q_s = q_s_next

            self.G[i] = G_pi
            epistemic_vals[i] = epi
            pragmatic_vals[i] = prag

        # Record decomposition
        self._epistemic_history.append(float(epistemic_vals.mean()))
        self._pragmatic_history.append(float(pragmatic_vals.mean()))

        # Policy posterior: q(pi) proportional to exp(-precision * G(pi))
        self.q_pi = _softmax(-precision * self.G)
        self._G_history.append(self.G.copy())
        self._q_pi_history.append(self.q_pi.copy())

        # Select best policy
        best_policy_idx = int(np.argmax(self.q_pi))
        action = int(self.policies[best_policy_idx, 0])
        return action, self.q_pi

    @property
    def G_history(self) -> np.ndarray:
        return np.array(self._G_history) if self._G_history else np.empty((0, self.n_policies))

    @property
    def q_pi_history(self) -> np.ndarray:
        return np.array(self._q_pi_history) if self._q_pi_history else np.empty((0, self.n_policies))

    @property
    def epistemic_history(self) -> np.ndarray:
        return np.array(self._epistemic_history)

    @property
    def pragmatic_history(self) -> np.ndarray:
        return np.array(self._pragmatic_history)


# ---------------------------------------------------------------------------
# Hierarchical Generative Model
# ---------------------------------------------------------------------------

class HierarchicalGenerativeModel:
    """
    Multi-level hierarchical generative model.

    Level 0 (sensory): maps sensory states to raw observations
    Level 1 (hidden):  hidden causes at intermediate timescale
    Level 2 (context): slow contextual / attentional states
    ...

    At each level l:
      p(o^l | s^l)         -- likelihood (A matrix)
      p(s^l | s^{l-1}, a)  -- transition (B matrix)
      p(s^l_0)             -- prior (D vector)

    Top-down: level l+1 provides prior for level l
    Bottom-up: level l sends prediction errors to level l+1
    """

    def __init__(
        self,
        obs_dims: Sequence[int],
        state_dims: Sequence[int],
        n_actions: int,
        n_policies: int = 8,
        policy_horizon: int = 3,
        precisions: Optional[Sequence[float]] = None,
    ) -> None:
        assert len(obs_dims) == len(state_dims), "Each level needs obs_dim and state_dim."
        self.depth = len(obs_dims)
        self.n_actions = n_actions

        if precisions is None:
            # Higher levels have lower precision (slower timescales)
            precisions = [2.0 ** (self.depth - i) for i in range(self.depth)]

        # Cortical layers (one per hierarchical level)
        self.layers: List[CorticalLayer] = [
            CorticalLayer(obs_dims[i], state_dims[i], precisions[i])
            for i in range(self.depth)
        ]

        # Transition model (shared or per-level; here one per level)
        self.transitions: List[TransitionModel] = [
            TransitionModel(state_dims[i], n_actions, rng_seed=i)
            for i in range(self.depth)
        ]

        # Preferences (only on lowest level where observations live)
        self.preferences = PreferenceModel(obs_dims[0])

        # Policy model
        self.policy_model = PolicyModel(
            n_actions=n_actions,
            n_policies=n_policies,
            policy_horizon=policy_horizon,
            n_states=state_dims[0],
        )

        # Neuromodulatory precision
        self.precision_module = PrecisionModule()

        # Episode counters
        self.t: int = 0
        self._action_history: List[int] = []
        self._total_fe_history: List[float] = []

    # ------------------------------------------------------------------
    # Perception: bottom-up pass (prediction-error minimization)
    # ------------------------------------------------------------------
    def perceive(self, observation: np.ndarray, n_iter: int = 16) -> List[np.ndarray]:
        """
        Hierarchical belief updating via predictive coding.

        Bottom-up: error from level l serves as observation for level l+1.
        Top-down:  belief from level l+1 constrains level l.

        Returns list of posterior beliefs [q_s^0, q_s^1, ..., q_s^L].
        """
        obs = np.asarray(observation, dtype=float)

        # --- Bottom-up pass: seed errors from sensory input ---
        # Level 0: observe raw sensory signal
        inputs: List[np.ndarray] = [obs]
        for l in range(1, self.depth):
            # Higher level observes the L2 prediction error from level below
            # (approximated as absolute error signal, normalised)
            prev_layer = self.layers[l - 1]
            err = prev_layer.compute_error(inputs[l - 1])
            err_obs = np.abs(err)
            # Resize to next level's obs_dim via uniform aggregation
            n_obs_up = self.layers[l].n_obs
            if err_obs.shape[0] != n_obs_up:
                # Simple resampling: nearest-neighbour
                indices = np.linspace(0, len(err_obs) - 1, n_obs_up).astype(int)
                err_obs = err_obs[indices]
            err_obs = np.clip(err_obs, 0, None)
            if err_obs.sum() > 0:
                err_obs /= err_obs.sum()
            else:
                err_obs = np.ones(n_obs_up) / n_obs_up
            inputs.append(err_obs)

        # --- Top-down modulated belief update ---
        beliefs: List[np.ndarray] = [np.empty(0)] * self.depth
        top_down: Optional[np.ndarray] = None

        # Update highest level first (no top-down constraint)
        for l in range(self.depth - 1, -1, -1):
            beliefs[l] = self.layers[l].update_beliefs(inputs[l], top_down, n_iter)
            # Propagate as top-down for next (lower) level
            # Map belief through A to get observation-space prediction
            top_down = self.layers[l].predict()
            # Resize if needed for lower level
            n_obs_down = self.layers[l - 1].n_obs if l > 0 else self.layers[l].n_obs
            if top_down.shape[0] != n_obs_down and l > 0:
                indices = np.linspace(0, len(top_down) - 1, n_obs_down).astype(int)
                top_down = top_down[indices]

        # Update precision module based on overall surprise
        total_fe = sum(layer.fe_history[-1] for layer in self.layers if len(layer.fe_history) > 0)
        self._total_fe_history.append(total_fe)
        state_surprise = abs(total_fe) / max(1.0, abs(total_fe))
        self.precision_module.update(state_surprise, state_surprise * 0.8)

        return beliefs

    # ------------------------------------------------------------------
    # Action: select via EFE
    # ------------------------------------------------------------------
    def act(self, belief: np.ndarray) -> int:
        """
        Select action by evaluating policies under current belief.

        Uses level-0 generative model (A, B) for prospective simulation.
        """
        action, q_pi = self.policy_model.evaluate_policies(
            current_belief=belief,
            A=self.layers[0].A,
            B=self.transitions[0].B,
            preferences=self.preferences,
            precision=self.precision_module.policy_precision,
        )
        self._action_history.append(action)
        self.t += 1
        return action

    # ------------------------------------------------------------------
    # Full step: perceive -> act
    # ------------------------------------------------------------------
    def step(self, observation: np.ndarray) -> Tuple[int, List[np.ndarray]]:
        """Single perception-action cycle."""
        beliefs = self.perceive(observation)
        action = self.act(beliefs[0])
        return action, beliefs

    # ------------------------------------------------------------------
    # State update (environment feedback)
    # ------------------------------------------------------------------
    def update_state(self, action: int, level: int = 0) -> np.ndarray:
        """Apply selected action to current belief, return predicted next state."""
        current_belief = self.layers[level].mu
        next_belief = self.transitions[level].predict_next(current_belief, action)
        self.layers[level].mu = next_belief
        return next_belief

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def total_fe_history(self) -> np.ndarray:
        return np.array(self._total_fe_history)

    @property
    def action_history(self) -> List[int]:
        return self._action_history.copy()

    def get_layer_fe_histories(self) -> List[np.ndarray]:
        return [layer.fe_history for layer in self.layers]

    def get_epistemic_pragmatic_histories(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.policy_model.epistemic_history,
            self.policy_model.pragmatic_history,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_default_agent(
    n_obs: int = 8,
    n_states: int = 6,
    n_actions: int = 4,
    depth: int = 3,
    n_policies: int = 12,
    policy_horizon: int = 4,
) -> HierarchicalGenerativeModel:
    """
    Create a default hierarchical active inference agent.

    Parameters
    ----------
    n_obs : int
        Number of observations at the lowest sensory level.
    n_states : int
        Number of hidden states at the lowest level.
    n_actions : int
        Number of available actions.
    depth : int
        Number of hierarchical levels.
    n_policies : int
        Number of policies evaluated per action selection.
    policy_horizon : int
        Number of time steps to evaluate each policy.

    Returns
    -------
    HierarchicalGenerativeModel
    """
    obs_dims = [max(2, n_obs // (2 ** i)) for i in range(depth)]
    state_dims = [max(2, n_states // (2 ** i)) for i in range(depth)]

    agent = HierarchicalGenerativeModel(
        obs_dims=obs_dims,
        state_dims=state_dims,
        n_actions=n_actions,
        n_policies=n_policies,
        policy_horizon=policy_horizon,
    )
    return agent


# ---------------------------------------------------------------------------
# PPO baseline (minimal, for comparison)
# ---------------------------------------------------------------------------

class MinimalPPOBaseline:
    """
    Stripped-down PPO-like agent for sample-efficiency comparison.

    Uses softmax policy with gradient-free updates (score function estimator)
    to approximate the sample inefficiency of gradient-based RL.

    This is intentionally minimal — the key comparison is the number of
    environment interactions needed to converge.
    """

    def __init__(self, n_obs: int, n_actions: int, lr: float = 0.01) -> None:
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.lr = lr
        rng = np.random.default_rng(99)
        self.theta: np.ndarray = rng.normal(0, 0.1, (n_actions, n_obs))
        self.value: np.ndarray = rng.normal(0, 0.1, n_obs)
        self._reward_history: List[float] = []
        self._step_count: int = 0

    def act(self, observation: np.ndarray) -> int:
        obs = np.asarray(observation, dtype=float)
        if obs.shape[0] != self.n_obs:
            indices = np.linspace(0, len(obs) - 1, self.n_obs).astype(int)
            obs = obs[indices]
        logits = self.theta @ obs
        probs = _softmax(logits)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, obs: np.ndarray, action: int, reward: float) -> None:
        """Score function (REINFORCE) update."""
        o = np.asarray(obs, dtype=float)
        if o.shape[0] != self.n_obs:
            indices = np.linspace(0, len(o) - 1, self.n_obs).astype(int)
            o = o[indices]
        baseline = float(self.value @ o)
        advantage = reward - baseline
        logits = self.theta @ o
        probs = _softmax(logits)
        grad = -advantage * o
        # Policy gradient update
        for a in range(self.n_actions):
            if a == action:
                self.theta[a] -= self.lr * grad * (1 - probs[a])
            else:
                self.theta[a] += self.lr * grad * probs[a]
        # Baseline update
        self.value += self.lr * advantage * o
        self._reward_history.append(reward)
        self._step_count += 1

    @property
    def reward_history(self) -> np.ndarray:
        return np.array(self._reward_history)

    @property
    def step_count(self) -> int:
        return self._step_count


# ---------------------------------------------------------------------------
# Utility: run a perception-action loop on a simple environment
# ---------------------------------------------------------------------------

def run_perception_action_loop(
    agent: HierarchicalGenerativeModel,
    env_fn: Callable[[int], np.ndarray],
    n_steps: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Run the agent for n_steps steps in an environment defined by env_fn.

    Parameters
    ----------
    agent : HierarchicalGenerativeModel
    env_fn : Callable[[int], np.ndarray]
        Given the action taken, returns the next observation vector.
    n_steps : int
    verbose : bool

    Returns
    -------
    dict with keys: actions, total_fe, beliefs_l0
    """
    n_obs = agent.layers[0].n_obs
    obs = np.ones(n_obs) / n_obs  # Uniform start

    results: dict = {"actions": [], "total_fe": [], "beliefs_l0": []}

    for t in range(n_steps):
        action, beliefs = agent.step(obs)
        obs = env_fn(action)
        results["actions"].append(action)
        results["total_fe"].append(agent.total_fe_history[-1] if len(agent.total_fe_history) > 0 else 0.0)
        results["beliefs_l0"].append(beliefs[0].copy())
        if verbose:
            print(f"  t={t:3d}  action={action}  fe={results['total_fe'][-1]:.4f}")

    return results
