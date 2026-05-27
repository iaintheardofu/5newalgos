"""
Tests for Algorithm #2: Active-Dendrite NMDA Sub-unit Networks
"""
import numpy as np
import pytest

from dendrite_network import (
    BranchConfig, NeuronConfig, NetworkConfig, ActiveDendriteNetwork
)


class TestBranchConfig:
    def test_defaults(self):
        cfg = BranchConfig()
        assert cfg.n_synapses == 64
        assert cfg.nmda_threshold > 0
        assert cfg.nonlinearity in ("sigmoid", "threshold", "relu_plateau")

    def test_ewc_lambda(self):
        cfg = BranchConfig()
        assert cfg.ewc_lambda > 0


class TestNeuronConfig:
    def test_defaults(self):
        cfg = NeuronConfig()
        assert cfg.n_branches == 8

    def test_custom_branches(self):
        cfg = NeuronConfig(n_branches=16)
        assert cfg.n_branches == 16


class TestActiveDendriteNetwork:
    def test_construction(self):
        cfg = NetworkConfig(
            n_neurons=10, n_input=32, n_output=3, n_context=8,
            n_branches=4, synapses_per_branch=8, random_seed=42
        )
        net = ActiveDendriteNetwork(cfg)
        assert net is not None

    def test_forward_with_task(self):
        cfg = NetworkConfig(
            n_neurons=10, n_input=32, n_output=3, n_context=8,
            n_branches=4, synapses_per_branch=8, random_seed=42
        )
        net = ActiveDendriteNetwork(cfg)
        # register_task(task_id) returns context vector
        net.register_task(0)
        x = np.random.default_rng(42).random(32)
        out = net.forward(x, task_id=0)
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))

    def test_register_and_train_task(self):
        cfg = NetworkConfig(
            n_neurons=10, n_input=32, n_output=3, n_context=8,
            n_branches=4, synapses_per_branch=8, random_seed=42
        )
        net = ActiveDendriteNetwork(cfg)
        rng = np.random.default_rng(42)
        net.register_task(0)
        X = rng.random((50, 32))
        y = rng.integers(0, 3, 50)
        result = net.train_task(0, X, y, n_epochs=3)
        assert isinstance(result, dict)
        acc = net.evaluate(X[:10], y[:10], task_id=0)
        assert 0.0 <= acc <= 1.0

    def test_continual_two_tasks(self):
        cfg = NetworkConfig(
            n_neurons=10, n_input=32, n_output=3, n_context=8,
            n_branches=4, synapses_per_branch=8, random_seed=42
        )
        net = ActiveDendriteNetwork(cfg)
        rng = np.random.default_rng(42)

        net.register_task(0)
        X0, y0 = rng.random((50, 32)), rng.integers(0, 3, 50)
        net.train_task(0, X0, y0, n_epochs=3)

        net.register_task(1)
        X1, y1 = rng.random((50, 32)), rng.integers(0, 3, 50)
        net.train_task(1, X1, y1, n_epochs=3)

        acc0 = net.evaluate(X0[:10], y0[:10], task_id=0)
        acc1 = net.evaluate(X1[:10], y1[:10], task_id=1)
        assert 0.0 <= acc0 <= 1.0
        assert 0.0 <= acc1 <= 1.0

    def test_branch_summary(self):
        cfg = NetworkConfig(
            n_neurons=10, n_input=32, n_output=3, n_context=8,
            n_branches=4, synapses_per_branch=8, random_seed=42
        )
        net = ActiveDendriteNetwork(cfg)
        summary = net.branch_summary()
        assert isinstance(summary, dict)
