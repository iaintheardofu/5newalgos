"""
Benchmark Suite: Astrocyte-Modulated Tripartite-Synapse Network (NALSM)
========================================================================

Measures:
  1. MNIST accuracy  — target ~85–92% on synthetic data (full MNIST ~97%)
  2. Parameter count comparison vs. equivalent Transformer / MLP
  3. Energy estimation (synaptic-op counts, pJ model)
  4. Memory footprint comparison
  5. Training throughput (samples/sec)
  6. Ablation study: effect of Ca2+ gating on accuracy

References
----------
  Kozachkov, Kastanenka & Krotov (PNAS 2023)
  Ivanov & Michmizos (NeurIPS 2021) — NALSM baseline ~97% on MNIST
  Davies et al. (2018) — Loihi energy model
  Strubell et al. (2019) — Transformer energy model
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np

from astrocyte_network import (
    AstrocyteConfig,
    LIFConfig,
    NetworkConfig,
    STDPConfig,
    TripartiteNetwork,
    make_synthetic_mnist,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for a single benchmark measurement."""

    name: str
    value: float
    unit: str
    notes: str = ""

    def __str__(self) -> str:
        note = f"  ({self.notes})" if self.notes else ""
        return f"  {self.name:<45} {self.value:>12.4g} {self.unit}{note}"


# ---------------------------------------------------------------------------
# 1. MNIST accuracy benchmark
# ---------------------------------------------------------------------------

def benchmark_accuracy(
    n_train: int = 3000,
    n_test: int = 600,
    n_hidden: int = 400,
    presentation_time: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Train NALSM on synthetic MNIST-like data and measure classification accuracy.

    Parameters
    ----------
    n_train : int
        Training set size.
    n_test : int
        Test set size.
    n_hidden : int
        Hidden LIF neuron count.
    presentation_time : int
        Timesteps per sample.
    seed : int
    verbose : bool

    Returns
    -------
    list[BenchmarkResult]
    """
    if verbose:
        print("\n[1/6] MNIST-like Accuracy Benchmark")
        print(f"      n_train={n_train}, n_test={n_test}, n_hidden={n_hidden}")

    cfg = NetworkConfig(
        n_input=784,
        n_hidden=n_hidden,
        n_output=10,
        lif=LIFConfig(n_neurons=n_hidden, adaptive_thresh=True, sparsity_target=0.05),
        astrocyte=AstrocyteConfig(coverage_k=8, tau_ca=200.0, tau_glutamate=30.0),
        stdp=STDPConfig(learning_rate=0.006, a_plus=0.012, a_minus=0.013),
        presentation_time=presentation_time,
        seed=seed,
    )
    net = TripartiteNetwork(cfg)

    X_train, y_train, X_test, y_test = make_synthetic_mnist(
        n_train, n_test, seed=seed
    )

    # --- Unsupervised phase ---
    t0 = time.perf_counter()
    reps_train = net.train_unsupervised(X_train, verbose=verbose)
    t_unsup = time.perf_counter() - t0

    # --- Supervised readout fitting ---
    net.readout.fit(reps_train, y_train)

    # --- Evaluation ---
    t_inf_start = time.perf_counter()
    reps_test = net.extract_representations(X_test)
    t_inference = time.perf_counter() - t_inf_start

    acc = net.readout.score(reps_test, y_test)
    throughput = n_train / max(t_unsup, 1e-6)
    inf_throughput = n_test / max(t_inference, 1e-6)

    if verbose:
        print(f"      Train accuracy: {net.readout.score(reps_train, y_train):.3f}")
        print(f"      Test  accuracy: {acc:.3f}")
        print(f"      Training time : {t_unsup:.2f}s ({throughput:.0f} samples/s)")
        print(f"      Inference time: {t_inference:.2f}s ({inf_throughput:.0f} samples/s)")

    return [
        BenchmarkResult("Test accuracy", acc * 100, "%",
                        f"synthetic MNIST, n_hidden={n_hidden}"),
        BenchmarkResult("Train accuracy", net.readout.score(reps_train, y_train) * 100, "%"),
        BenchmarkResult("Training throughput", throughput, "samples/s"),
        BenchmarkResult("Inference throughput", inf_throughput, "samples/s"),
        BenchmarkResult("Training wall time", t_unsup, "s",
                        f"{n_train} samples × {presentation_time} steps"),
    ]


# ---------------------------------------------------------------------------
# 2. Parameter count comparison
# ---------------------------------------------------------------------------

def benchmark_parameters(
    hidden_sizes: tuple[int, ...] = (200, 500, 1000),
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Compare parameter counts: NALSM vs. Transformer vs. MLP.

    Transformer is sized to have roughly the same hidden dimensionality.
    MLP has two hidden layers of size `h`.
    """
    if verbose:
        print("\n[2/6] Parameter Count Comparison")

    results: list[BenchmarkResult] = []
    n_input, n_output = 784, 10

    for h in hidden_sizes:
        cfg = NetworkConfig(
            n_input=n_input, n_hidden=h, n_output=n_output,
            lif=LIFConfig(n_neurons=h),
            astrocyte=AstrocyteConfig(coverage_k=8),
            stdp=STDPConfig(),
            presentation_time=30, seed=0,
        )
        net = TripartiteNetwork(cfg)
        param_snn = net.total_parameters()

        # Transformer (1-layer, d_model=h, 4 heads, FFN=4h)
        param_transformer = (
            3 * h * h + h * h +  # QKV + output projection
            2 * h * 4 * h +      # FFN
            h * n_output +       # classification head
            2 * h                # layer norms (minimal)
        )

        # Dense MLP (2 hidden layers, size h)
        param_mlp = (
            n_input * h + h +    # layer 1
            h * h + h +          # layer 2
            h * n_output + n_output  # output
        )

        ratio_vs_transformer = param_snn / max(param_transformer, 1)
        ratio_vs_mlp = param_snn / max(param_mlp, 1)

        if verbose:
            print(f"\n      h={h:>5}:")
            print(f"        NALSM SNN  : {param_snn:>10,} params")
            print(f"        Transformer: {param_transformer:>10,} params  ({1/ratio_vs_transformer:.1f}x more)")
            print(f"        Dense MLP  : {param_mlp:>10,} params  ({1/ratio_vs_mlp:.1f}x more)")

        results.extend([
            BenchmarkResult(f"NALSM params (h={h})", param_snn, "params"),
            BenchmarkResult(f"Transformer params (h={h})", param_transformer, "params"),
            BenchmarkResult(f"MLP params (h={h})", param_mlp, "params"),
            BenchmarkResult(f"Transformer/SNN ratio (h={h})", 1.0 / ratio_vs_transformer, "x"),
        ])

    return results


# ---------------------------------------------------------------------------
# 3. Energy estimation
# ---------------------------------------------------------------------------

def benchmark_energy(
    hidden_sizes: tuple[int, ...] = (200, 500, 1000),
    presentation_time: int = 30,
    sparsity: float = 0.05,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Estimate energy per inference for NALSM vs. Transformer.

    Energy model:
      SNN  : synaptice event = 23.6 pJ   (Loihi, Davies et al. 2018)
      Dense: 32-bit FP MAC   = 3.7 pJ    (GPU A100, Choquette 2021)
      INT8 : 8-bit MAC        = 0.5 pJ    (GPU tensor core)
    """
    if verbose:
        print("\n[3/6] Energy Estimation (per single inference)")

    E_snn_pJ = 23.6     # pJ per synaptic event (spike)
    E_fp32_pJ = 3.7     # pJ per FP32 MAC
    E_int8_pJ = 0.5     # pJ per INT8 MAC

    n_input, n_output = 784, 10
    conn_prob = 0.15

    results: list[BenchmarkResult] = []

    for h in hidden_sizes:
        # SNN ops per sample
        n_spikes_per_step = int(h * sparsity)
        # Each spike activates conn_prob * h post-synaptic neurons
        snn_syn_events = n_spikes_per_step * int(h * conn_prob) * presentation_time
        # Astrocyte ops (vector dot products, cheap)
        n_astro = h // 8
        astro_ops = n_astro * 8 * presentation_time  # coverage_k=8
        total_snn_ops = snn_syn_events + astro_ops
        snn_energy_pJ = total_snn_ops * E_snn_pJ

        # Transformer ops (1-layer, seq_len = presentation_time)
        seq = presentation_time
        qkv_ops = 3 * seq * h * h
        attn_ops = seq * seq * h
        ffn_ops = 2 * seq * h * 4 * h
        total_transf_ops = qkv_ops + attn_ops + ffn_ops
        transf_energy_fp32 = total_transf_ops * E_fp32_pJ
        transf_energy_int8 = total_transf_ops * E_int8_pJ

        speedup_vs_fp32 = transf_energy_fp32 / max(snn_energy_pJ, 1e-3)
        speedup_vs_int8 = transf_energy_int8 / max(snn_energy_pJ, 1e-3)

        if verbose:
            print(f"\n      h={h:>5}:")
            print(f"        SNN  synaptic events : {total_snn_ops:>12,}")
            print(f"        Transformer FP32 MACs: {total_transf_ops:>12,}")
            print(f"        SNN  energy : {snn_energy_pJ:>10.1f} pJ")
            print(f"        Transf (FP32): {transf_energy_fp32:>9.1f} pJ  ({speedup_vs_fp32:.1f}x)")
            print(f"        Transf (INT8): {transf_energy_int8:>9.1f} pJ  ({speedup_vs_int8:.1f}x)")

        results.extend([
            BenchmarkResult(f"SNN energy/inference (h={h})", snn_energy_pJ, "pJ",
                            f"sparsity={sparsity:.0%}"),
            BenchmarkResult(f"Transformer FP32 energy (h={h})", transf_energy_fp32, "pJ"),
            BenchmarkResult(f"Transformer INT8 energy (h={h})", transf_energy_int8, "pJ"),
            BenchmarkResult(f"SNN speedup vs FP32 (h={h})", speedup_vs_fp32, "x"),
            BenchmarkResult(f"SNN speedup vs INT8 (h={h})", speedup_vs_int8, "x"),
        ])

    return results


# ---------------------------------------------------------------------------
# 4. Memory footprint comparison
# ---------------------------------------------------------------------------

def benchmark_memory(
    n_hidden: int = 500,
    n_train_samples: int = 200,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Measure peak RSS memory usage via tracemalloc.

    Compares:
      - NALSM forward pass (unsupervised)
      - Equivalent dense MLP allocation (estimated)
    """
    if verbose:
        print("\n[4/6] Memory Footprint Benchmark")

    cfg = NetworkConfig(
        n_input=784, n_hidden=n_hidden, n_output=10,
        lif=LIFConfig(n_neurons=n_hidden),
        astrocyte=AstrocyteConfig(coverage_k=8),
        stdp=STDPConfig(),
        presentation_time=20, seed=0,
    )

    # --- SNN memory ---
    tracemalloc.start()
    net = TripartiteNetwork(cfg)
    X_bench, _, _, _ = make_synthetic_mnist(n_train_samples, 50, seed=0)
    _ = net.train_unsupervised(X_bench, verbose=False)
    current, peak_snn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Dense MLP equivalent memory (estimated analytically) ---
    h = n_hidden
    n_input, n_output = 784, 10
    # Weights + gradients + activations (float64 = 8 bytes)
    bytes_per_float = 8
    mlp_weights = (n_input * h + h + h * h + h + h * n_output + n_output) * bytes_per_float
    # Activations for forward + backward: batch=1, seq=20
    mlp_activations = (n_input + h + h + n_output) * 20 * bytes_per_float * 2
    peak_mlp_est = mlp_weights + mlp_activations

    ratio = peak_snn / max(peak_mlp_est, 1)

    if verbose:
        print(f"      SNN peak memory : {peak_snn / 1024**2:.2f} MB")
        print(f"      MLP est. memory : {peak_mlp_est / 1024**2:.2f} MB")
        print(f"      SNN/MLP ratio   : {ratio:.2f}x")
        print(f"      SNN active params weight: {net.total_parameters()} entries")

    return [
        BenchmarkResult("SNN peak memory", peak_snn / 1024**2, "MB",
                        f"n_hidden={n_hidden}, {n_train_samples} samples"),
        BenchmarkResult("MLP estimated memory", peak_mlp_est / 1024**2, "MB"),
        BenchmarkResult("SNN/MLP memory ratio", ratio, "x"),
    ]


# ---------------------------------------------------------------------------
# 5. Training throughput
# ---------------------------------------------------------------------------

def benchmark_throughput(
    hidden_sizes: tuple[int, ...] = (100, 200, 500),
    n_samples: int = 100,
    presentation_time: int = 20,
    seed: int = 1,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Measure training throughput (samples/second) across network sizes.
    """
    if verbose:
        print("\n[5/6] Training Throughput Benchmark")

    results: list[BenchmarkResult] = []
    X_bench, _, _, _ = make_synthetic_mnist(n_samples, 10, seed=seed)

    for h in hidden_sizes:
        cfg = NetworkConfig(
            n_input=784, n_hidden=h, n_output=10,
            lif=LIFConfig(n_neurons=h),
            astrocyte=AstrocyteConfig(coverage_k=8),
            stdp=STDPConfig(),
            presentation_time=presentation_time, seed=seed,
        )
        net = TripartiteNetwork(cfg)

        t0 = time.perf_counter()
        net.train_unsupervised(X_bench, verbose=False)
        elapsed = time.perf_counter() - t0

        throughput = n_samples / max(elapsed, 1e-6)
        ms_per_sample = elapsed / n_samples * 1000

        if verbose:
            print(f"      h={h:>5}: {throughput:>7.1f} samples/s  ({ms_per_sample:.1f} ms/sample)")

        results.extend([
            BenchmarkResult(f"Throughput (h={h})", throughput, "samples/s"),
            BenchmarkResult(f"Latency (h={h})", ms_per_sample, "ms/sample"),
        ])

    return results


# ---------------------------------------------------------------------------
# 6. Ablation study: effect of Ca2+ gating
# ---------------------------------------------------------------------------

def benchmark_ablation(
    n_train: int = 1000,
    n_test: int = 200,
    n_hidden: int = 200,
    presentation_time: int = 25,
    seed: int = 5,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Ablation: compare accuracy with and without astrocyte Ca2+ gating.

    Conditions:
      A. Full NALSM — tripartite STDP with Ca2+ gate
      B. Classical STDP — gating always 1.0 (no astrocyte modulation)
      C. No STDP — frozen random weights, only readout trained
    """
    if verbose:
        print("\n[6/6] Ablation Study: Effect of Ca2+ Gating")

    X_train, y_train, X_test, y_test = make_synthetic_mnist(
        n_train, n_test, seed=seed
    )
    results: list[BenchmarkResult] = []

    # --- Condition A: Full NALSM ---
    cfg = NetworkConfig(
        n_input=784, n_hidden=n_hidden, n_output=10,
        lif=LIFConfig(n_neurons=n_hidden, adaptive_thresh=True),
        astrocyte=AstrocyteConfig(coverage_k=8),
        stdp=STDPConfig(learning_rate=0.007),
        presentation_time=presentation_time, seed=seed,
    )
    net_A = TripartiteNetwork(cfg)
    reps_A_train = net_A.train_unsupervised(X_train, verbose=False)
    net_A.readout.fit(reps_A_train, y_train)
    reps_A_test = net_A.extract_representations(X_test)
    acc_A = net_A.readout.score(reps_A_test, y_test)

    # --- Condition B: Classical STDP (no Ca2+ gate) ---
    net_B = TripartiteNetwork(cfg)
    # Override gating to return constant 1.0
    net_B.astro_fwd.gating_values = lambda ca=None: np.ones(net_B.astro_fwd.n_astro)  # type: ignore[method-assign]
    net_B.astro_rec.gating_values = lambda ca=None: np.ones(net_B.astro_rec.n_astro)  # type: ignore[method-assign]
    reps_B_train = net_B.train_unsupervised(X_train, verbose=False)
    net_B.readout.fit(reps_B_train, y_train)
    reps_B_test = net_B.extract_representations(X_test)
    acc_B = net_B.readout.score(reps_B_test, y_test)

    # --- Condition C: No STDP (frozen weights) ---
    net_C = TripartiteNetwork(cfg)
    reps_C_train = net_C.extract_representations(X_train)  # learn=False
    net_C.readout.fit(reps_C_train, y_train)
    reps_C_test = net_C.extract_representations(X_test)
    acc_C = net_C.readout.score(reps_C_test, y_test)

    delta_AB = acc_A - acc_B
    delta_AC = acc_A - acc_C

    if verbose:
        print(f"      A. Full NALSM (tripartite STDP + Ca2+ gate): {acc_A:.3f}")
        print(f"      B. Classical STDP (no Ca2+ gate)           : {acc_B:.3f}  (Δ={delta_AB:+.3f})")
        print(f"      C. No STDP (frozen random weights)         : {acc_C:.3f}  (Δ={delta_AC:+.3f})")
        print(f"      Ca2+ gating benefit: +{delta_AB*100:.1f} pp over classical STDP")

    results.extend([
        BenchmarkResult("Accuracy: Full NALSM (tripartite STDP)", acc_A * 100, "%"),
        BenchmarkResult("Accuracy: Classical STDP (no Ca2+ gate)", acc_B * 100, "%"),
        BenchmarkResult("Accuracy: Frozen weights (no STDP)", acc_C * 100, "%"),
        BenchmarkResult("Ca2+ gating gain over classical STDP", delta_AB * 100, "pp"),
        BenchmarkResult("STDP gain over frozen weights", delta_AC * 100, "pp"),
    ])

    return results


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    fast: bool = False,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """
    Run the full benchmark suite.

    Parameters
    ----------
    fast : bool
        Use reduced dataset sizes for quick testing.
    verbose : bool

    Returns
    -------
    all_results : list[BenchmarkResult]
    """
    print("=" * 65)
    print("NALSM Tripartite SNN — Full Benchmark Suite")
    print("=" * 65)

    scale = 0.3 if fast else 1.0

    all_results: list[BenchmarkResult] = []

    all_results += benchmark_accuracy(
        n_train=int(2000 * scale),
        n_test=int(400 * scale),
        n_hidden=int(300 * scale) or 100,
        presentation_time=25,
        verbose=verbose,
    )

    all_results += benchmark_parameters(
        hidden_sizes=(200, 500, 1000) if not fast else (200, 500),
        verbose=verbose,
    )

    all_results += benchmark_energy(
        hidden_sizes=(200, 500, 1000) if not fast else (200, 500),
        verbose=verbose,
    )

    all_results += benchmark_memory(
        n_hidden=int(400 * scale) or 100,
        n_train_samples=int(200 * scale) or 50,
        verbose=verbose,
    )

    all_results += benchmark_throughput(
        hidden_sizes=(100, 200) if fast else (100, 200, 500),
        n_samples=int(80 * scale) or 40,
        presentation_time=20,
        verbose=verbose,
    )

    all_results += benchmark_ablation(
        n_train=int(800 * scale) or 300,
        n_test=int(200 * scale) or 80,
        n_hidden=int(200 * scale) or 100,
        presentation_time=20,
        verbose=verbose,
    )

    # --- Summary table ---
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    for r in all_results:
        print(r)
    print("=" * 65)

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        print("[fast mode enabled — reduced dataset sizes]\n")
    run_all_benchmarks(fast=fast_mode, verbose=True)
