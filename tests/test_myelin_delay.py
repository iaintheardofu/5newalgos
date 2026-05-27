"""
Tests for Algorithm #3: Oligodendrocyte/Myelin-Plastic Polychronous SNNs
"""
import numpy as np
import pytest

from polychronous_snn import (
    LIFParams, LIFNeuron, DCLSDelay, OligodendrocyteMod,
    PolychronousSNN, NetworkConfig,
    detect_polychronous_groups, generate_temporal_patterns
)


class TestLIFParams:
    def test_defaults(self):
        p = LIFParams()
        assert p.tau_m == 20.0
        assert p.v_thresh > p.v_reset

    def test_custom(self):
        p = LIFParams(tau_m=10.0, v_thresh=-40.0)
        assert p.tau_m == 10.0


class TestLIFNeuron:
    def test_creation(self):
        neuron = LIFNeuron(n_neurons=50)
        assert neuron.n == 50

    def test_reset(self):
        neuron = LIFNeuron(n_neurons=30)
        neuron.reset()
        assert neuron.v.shape == (30,)

    def test_step_with_current(self):
        neuron = LIFNeuron(n_neurons=20)
        neuron.reset()
        I = np.ones(20) * 50.0
        total_spikes = 0
        for _ in range(100):
            spikes = neuron.step(I)
            total_spikes += spikes.sum()
        assert total_spikes > 0


class TestDCLSDelay:
    def test_creation(self):
        dcls = DCLSDelay(n_pre=30, n_post=20, d_max=15)
        assert dcls.weights.shape == (20, 30)  # (post, pre)

    def test_mean_delay(self):
        dcls = DCLSDelay(n_pre=30, n_post=20, d_max=15)
        md = dcls.mean_delay_ms  # property, not callable
        assert isinstance(md, np.ndarray)

    def test_forward(self):
        dcls = DCLSDelay(n_pre=10, n_post=5, d_max=10)
        # forward takes spike_buffer, not (spikes, t)
        spike_buffer = np.zeros((10, 10))  # (d_max, n_pre)
        spike_buffer[0, 0] = 1.0
        spike_buffer[2, 3] = 1.0
        out = dcls.forward(spike_buffer)
        assert out.shape == (5,)
        assert np.all(np.isfinite(out))


class TestOligodendrocyteMod:
    def test_creation(self):
        omp = OligodendrocyteMod()
        assert omp is not None

    def test_record_and_compute(self):
        omp = OligodendrocyteMod()
        dcls = DCLSDelay(n_pre=10, n_post=5, d_max=10)
        for t in range(60):
            spikes = (np.random.default_rng(42 + t).random(5) > 0.8).astype(float)
            omp.record_spikes(spikes)
        # compute_delta_delays takes the delay array, not the DCLSDelay object
        delta = omp.compute_delta_delays(dcls.delays)
        assert isinstance(delta, np.ndarray)


class TestPolychronousSNN:
    def test_construction(self):
        cfg = NetworkConfig(n_excit=40, n_inhib=10, d_max=10)
        snn = PolychronousSNN(cfg, rng=np.random.default_rng(42))
        assert snn is not None

    def test_step(self):
        cfg = NetworkConfig(n_excit=30, n_inhib=8, d_max=8, n_input=30)
        snn = PolychronousSNN(cfg, rng=np.random.default_rng(42))
        I_ext = np.random.default_rng(42).random(cfg.n_input) * 5.0
        result = snn.step(I_ext)
        assert result is not None

    def test_summary(self):
        cfg = NetworkConfig(n_excit=30, n_inhib=8, d_max=8)
        snn = PolychronousSNN(cfg, rng=np.random.default_rng(42))
        s = snn.summary()
        assert isinstance(s, str)  # returns formatted string


class TestTemporalPatterns:
    def test_generation(self):
        patterns, labels = generate_temporal_patterns(
            n_neurons=30, n_patterns=3, pattern_duration=20
        )
        assert patterns is not None
        assert labels is not None


class TestPolychronousGroupDetection:
    def test_detection(self):
        dcls = DCLSDelay(n_pre=20, n_post=20, d_max=10,
                         rng=np.random.default_rng(42))
        groups = detect_polychronous_groups(dcls, min_group_size=2)
        assert isinstance(groups, list)
