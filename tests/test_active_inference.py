"""
Tests for Algorithm #4: Hierarchical Active Inference / Predictive-Coding Agents
"""
import numpy as np
import pytest

from active_inference import (
    _softmax, _log, _kl_dirichlet_categorical, _entropy,
    PrecisionModule, CorticalLayer, TransitionModel,
    HierarchicalGenerativeModel
)


class TestNumerics:
    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        s = _softmax(x)
        assert np.allclose(s.sum(), 1.0)
        assert s[2] > s[1] > s[0]

    def test_softmax_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        s = _softmax(x)
        assert np.allclose(s.sum(), 1.0)
        assert np.all(np.isfinite(s))

    def test_softmax_negative(self):
        x = np.array([-1000.0, -999.0, -998.0])
        s = _softmax(x)
        assert np.allclose(s.sum(), 1.0)

    def test_log_safe(self):
        result = _log(np.array([0.0]))
        assert np.isfinite(result[0])

    def test_kl_identical(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = _kl_dirichlet_categorical(p, p)
        assert abs(kl) < 1e-10

    def test_kl_positive(self):
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.33, 0.33, 0.34])
        kl = _kl_dirichlet_categorical(p, q)
        assert kl > 0

    def test_entropy_ordering(self):
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        peaked = np.array([0.97, 0.01, 0.01, 0.01])
        assert _entropy(uniform) > _entropy(peaked)


class TestPrecisionModule:
    def test_creation(self):
        pm = PrecisionModule()
        assert pm.state_precision > 0
        assert pm.policy_precision > 0

    def test_update(self):
        pm = PrecisionModule(state_precision=2.0, policy_precision=4.0)
        sp_before = pm.state_precision
        pm.update(state_surprise=5.0, policy_surprise=0.5)
        # Should adapt


class TestCorticalLayer:
    def test_creation(self):
        layer = CorticalLayer(n_obs=5, n_states=3)
        assert layer.A.shape == (5, 3)
        assert layer.mu.shape == (3,)

    def test_predict(self):
        layer = CorticalLayer(n_obs=5, n_states=3)
        pred = layer.predict()
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_compute_error(self):
        layer = CorticalLayer(n_obs=5, n_states=3)
        obs = np.zeros(5)
        obs[0] = 1.0
        epsilon = layer.compute_error(obs)
        assert epsilon.shape == (5,)
        assert np.all(np.isfinite(epsilon))

    def test_update_beliefs(self):
        layer = CorticalLayer(n_obs=5, n_states=3)
        obs = np.zeros(5)
        obs[2] = 1.0
        mu_before = layer.mu.copy()
        layer.update_beliefs(obs)
        assert not np.allclose(layer.mu, mu_before, atol=1e-10)


class TestTransitionModel:
    def test_creation(self):
        tm = TransitionModel(n_states=4, n_actions=3)
        assert len(tm.B) == 3
        for b in tm.B:
            assert b.shape == (4, 4)
            assert np.allclose(b.sum(axis=0), 1.0, atol=0.01)


class TestHierarchicalGenerativeModel:
    def test_construction(self):
        model = HierarchicalGenerativeModel(
            obs_dims=[6], state_dims=[4], n_actions=3
        )
        assert model is not None

    def test_step(self):
        model = HierarchicalGenerativeModel(
            obs_dims=[4], state_dims=[3], n_actions=2
        )
        obs = np.array([1.0, 0.0, 0.0, 0.0])
        action, beliefs = model.step(obs)
        assert 0 <= action < 2

    def test_free_energy_finite(self):
        model = HierarchicalGenerativeModel(
            obs_dims=[4], state_dims=[3], n_actions=2
        )
        obs = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(5):
            model.step(obs)
        fe_history = model.total_fe_history
        assert all(np.isfinite(f) for f in fe_history)


class TestMastermindAgent:
    def test_import(self):
        from mastermind_demo import ActiveInferenceMastermindAgent
        agent = ActiveInferenceMastermindAgent(n_pegs=4, n_colors=6)
        assert agent is not None

    def test_solve_small(self):
        from mastermind_demo import ActiveInferenceMastermindAgent
        agent = ActiveInferenceMastermindAgent(n_pegs=4, n_colors=6)
        secret = (0, 1, 2, 3)
        result = agent.solve(secret, max_guesses=10)
        assert isinstance(result, dict)
