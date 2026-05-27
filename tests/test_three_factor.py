"""
Tests for Algorithm #5: Three-Factor Neuromodulated Plasticity + Sleep-Replay + Neurogenesis
"""
import numpy as np
import pytest

from three_factor_system import (
    NeuromodulatorSignals, EligibilityTraceBuffer, ThreeFactorLayer,
    SleepReplayConsolidator, NeurogenesisRegularizer, ThreeFactorNetwork,
    RewardPredictionError, SyntheticTaskGenerator, OutputLayer
)


class TestNeuromodulatorSignals:
    def test_defaults(self):
        nm = NeuromodulatorSignals()
        assert nm.DA == 1.0
        assert nm.ACh == 1.0
        assert nm.NE == 1.0

    def test_combined_neutral(self):
        nm = NeuromodulatorSignals(DA=1.0, ACh=1.0, NE=1.0)
        assert nm.combined_modulation() == 0.0

    def test_combined_positive(self):
        nm = NeuromodulatorSignals(DA=2.0, ACh=1.5, NE=1.0)
        assert nm.combined_modulation() > 0

    def test_combined_negative(self):
        nm = NeuromodulatorSignals(DA=0.5, ACh=1.0, NE=1.0)
        assert nm.combined_modulation() < 0

    def test_effective_lr_gain(self):
        nm = NeuromodulatorSignals(DA=1.5, ACh=2.0, NE=1.0)
        gain = nm.effective_lr_gain()
        assert gain > 0


class TestEligibilityTraceBuffer:
    def test_creation(self):
        buf = EligibilityTraceBuffer(n_out=5, n_in=10, tau_e=20.0)
        assert buf.traces.shape == (5, 10)

    def test_update_accumulates(self):
        buf = EligibilityTraceBuffer(n_out=5, n_in=10, tau_e=20.0)
        buf.update(np.ones(10) * 0.5, np.ones(5) * 0.8)
        assert np.any(buf.traces > 0)

    def test_decay(self):
        buf = EligibilityTraceBuffer(n_out=5, n_in=10, tau_e=20.0)
        buf.update(np.ones(10), np.ones(5))
        val = buf.traces.copy()
        buf.update(np.zeros(10), np.zeros(5))
        assert np.all(buf.traces <= val + 1e-10)

    def test_reset(self):
        buf = EligibilityTraceBuffer(n_out=5, n_in=10)
        buf.update(np.ones(10), np.ones(5))
        buf.reset()
        assert np.allclose(buf.traces, 0.0)


class TestThreeFactorLayer:
    def test_creation(self):
        layer = ThreeFactorLayer(n_in=20, n_out=10)
        assert layer is not None

    def test_forward(self):
        layer = ThreeFactorLayer(n_in=20, n_out=10, rng=np.random.default_rng(42))
        x = np.random.default_rng(42).random(20)
        out = layer.forward(x)
        assert out.shape == (10,)
        assert np.all(np.isfinite(out))

    def test_snapshot_restore(self):
        layer = ThreeFactorLayer(n_in=20, n_out=10, rng=np.random.default_rng(42))
        w_snap = layer.snapshot_weights()
        layer.W += 1.0
        layer.restore_weights(w_snap)
        assert np.allclose(layer.W, w_snap)


class TestOutputLayer:
    def test_creation(self):
        out = OutputLayer(n_in=10, n_classes=3)
        assert out is not None

    def test_forward(self):
        out = OutputLayer(n_in=10, n_classes=3, rng=np.random.default_rng(42))
        h = np.random.default_rng(42).random(10)
        probs = out.forward(h)
        assert probs.shape == (3,)
        assert np.allclose(probs.sum(), 1.0, atol=0.01)


class TestSleepReplayConsolidator:
    def test_creation(self):
        src = SleepReplayConsolidator()
        assert src is not None

    def test_consolidate(self):
        layer = ThreeFactorLayer(n_in=20, n_out=10, rng=np.random.default_rng(42))
        src = SleepReplayConsolidator(rng=np.random.default_rng(42))
        w_before = layer.W.copy()
        src.consolidate(layer, n_in=20)
        assert not np.allclose(w_before, layer.W)


class TestNeurogenesisRegularizer:
    def test_creation(self):
        nr = NeurogenesisRegularizer(fraction=0.05)
        assert nr.fraction == 0.05

    def test_turnover(self):
        layer = ThreeFactorLayer(n_in=20, n_out=100, rng=np.random.default_rng(42))
        nr = NeurogenesisRegularizer(period=1, fraction=0.1, rng=np.random.default_rng(42))
        w_before = layer.W.copy()
        nr.maybe_apply(layer, step=1)
        changed = np.any(layer.W != w_before, axis=1)
        assert changed.sum() >= 5


class TestThreeFactorNetwork:
    def test_construction(self):
        net = ThreeFactorNetwork(n_input=32, n_hidden=20, n_classes=5, seed=42)
        assert net is not None

    def test_train_step(self):
        net = ThreeFactorNetwork(n_input=32, n_hidden=20, n_classes=3, seed=42)
        x = np.random.default_rng(42).random(32)
        result = net.train_step(x, 1)
        assert isinstance(result, dict)

    def test_evaluate(self):
        net = ThreeFactorNetwork(n_input=32, n_hidden=20, n_classes=3, seed=42)
        rng = np.random.default_rng(42)
        X = rng.random((20, 32))
        y = rng.integers(0, 3, 20)
        acc = net.evaluate(X, y)
        assert 0.0 <= acc <= 1.0


class TestRewardPredictionError:
    def test_creation(self):
        rpe = RewardPredictionError()
        assert rpe is not None

    def test_step(self):
        rpe = RewardPredictionError(alpha=0.1)
        error = rpe.step(state=0, reward=1.0, next_state=1)
        assert np.isfinite(error)


class TestSyntheticTaskGenerator:
    def test_creation(self):
        gen = SyntheticTaskGenerator(n_tasks=5, n_samples=100, n_features=32, n_classes=3)
        assert gen.input_dim == 32
        assert gen.output_dim == 3

    def test_get_task(self):
        gen = SyntheticTaskGenerator(n_tasks=5, n_samples=100, n_features=32, n_classes=3)
        X, y = gen.get_task(0)
        assert X.shape == (100, 32)
        assert y.shape == (100,)

    def test_different_tasks(self):
        gen = SyntheticTaskGenerator(n_tasks=5, n_samples=50, n_features=32, n_classes=3)
        X1, _ = gen.get_task(0)
        X2, _ = gen.get_task(1)
        assert not np.allclose(X1.mean(axis=0), X2.mean(axis=0), atol=0.5)
