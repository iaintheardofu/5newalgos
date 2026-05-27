"""
Tests for Algorithm #1: Astrocyte-Modulated Tripartite-Synapse Networks (NALSM)
"""
import numpy as np
import pytest

from astrocyte_network import (
    LIFConfig, AstrocyteConfig, STDPConfig, NetworkConfig,
    LIFLayer, AstrocyteLayer, TripartiteNetwork, make_synthetic_mnist
)


class TestLIFConfig:
    def test_defaults(self):
        cfg = LIFConfig()
        assert cfg.n_neurons == 256
        assert cfg.tau_m == 20.0
        assert cfg.v_thresh > cfg.v_reset

    def test_custom(self):
        cfg = LIFConfig(n_neurons=100, tau_m=10.0)
        assert cfg.n_neurons == 100


class TestAstrocyteConfig:
    def test_defaults(self):
        cfg = AstrocyteConfig()
        assert cfg.coverage_k == 8
        assert cfg.g_max > cfg.g_min

    def test_slow_timescale(self):
        cfg = AstrocyteConfig()
        assert cfg.tau_ca >= 100.0, "Ca2+ timescale should be slow (100ms+)"


class TestLIFLayer:
    def test_creation_and_reset(self):
        cfg = LIFConfig(n_neurons=50)
        rng = np.random.default_rng(42)
        layer = LIFLayer(cfg, rng=rng)
        layer.reset_state()
        assert layer.v.shape == (50,)

    def test_step_produces_spikes(self):
        cfg = LIFConfig(n_neurons=50, v_thresh=-50.0)
        rng = np.random.default_rng(42)
        layer = LIFLayer(cfg, rng=rng)
        layer.reset_state()
        I_syn = np.ones(50) * 100.0
        total = 0
        for _ in range(100):
            layer.step(I_syn)
            total += layer.spike.sum()
        assert total > 0, "Strong current should produce spikes"

    def test_no_spikes_without_input(self):
        cfg = LIFConfig(n_neurons=20)
        rng = np.random.default_rng(42)
        layer = LIFLayer(cfg, rng=rng)
        layer.reset_state()
        layer.step(np.zeros(20))
        assert not np.any(layer.spike)


class TestAstrocyteLayer:
    def test_creation(self):
        cfg = AstrocyteConfig(coverage_k=4)
        rng = np.random.default_rng(42)
        layer = AstrocyteLayer(n_pre=20, n_astrocytes=5, config=cfg, rng=rng)
        assert layer.ca is not None

    def test_calcium_changes(self):
        cfg = AstrocyteConfig(coverage_k=4)
        rng = np.random.default_rng(42)
        layer = AstrocyteLayer(n_pre=20, n_astrocytes=5, config=cfg, rng=rng)
        layer.reset_state()
        ca_before = layer.ca.copy()
        pre_rates = np.random.default_rng(42).random(20) * 0.5
        for _ in range(100):
            layer.step(pre_rates)
        assert not np.allclose(layer.ca, ca_before, atol=1e-10)


class TestTripartiteNetwork:
    def test_construction(self):
        cfg = NetworkConfig(
            n_input=50, n_hidden=30, n_output=5,
            presentation_time=10, seed=42
        )
        net = TripartiteNetwork(cfg)
        assert net is not None

    def test_run_sample(self):
        cfg = NetworkConfig(
            n_input=50, n_hidden=30, n_output=5,
            presentation_time=10, seed=42
        )
        net = TripartiteNetwork(cfg)
        x = np.random.default_rng(42).random(50)
        result = net.run_sample(x)
        assert result is not None

    def test_parameter_count(self):
        cfg = NetworkConfig(
            n_input=50, n_hidden=30, n_output=5,
            presentation_time=10, seed=42
        )
        net = TripartiteNetwork(cfg)
        count = net.total_parameters()
        assert count > 0

    def test_train_and_score(self):
        n_in = 50
        X_train, y_train, X_test, y_test = make_synthetic_mnist(
            n_train=60, n_test=20, n_classes=3, image_size=n_in
        )
        cfg = NetworkConfig(
            n_input=n_in, n_hidden=30, n_output=3,
            presentation_time=5, seed=42
        )
        net = TripartiteNetwork(cfg)
        # fit_readout takes raw data internally
        net.fit_readout(X_train[:30], y_train[:30])
        acc = net.score(X_test[:10], y_test[:10])
        assert 0.0 <= acc <= 1.0


class TestSyntheticData:
    def test_make_synthetic_mnist(self):
        X_tr, y_tr, X_te, y_te = make_synthetic_mnist(
            n_train=200, n_test=50, n_classes=5, image_size=64
        )
        assert X_tr.shape == (200, 64)
        assert y_tr.shape == (200,)
        assert X_te.shape == (50, 64)
        assert len(np.unique(y_tr)) == 5

    def test_values_in_range(self):
        X_tr, _, _, _ = make_synthetic_mnist(n_train=50, n_test=10)
        assert X_tr.min() >= 0.0
