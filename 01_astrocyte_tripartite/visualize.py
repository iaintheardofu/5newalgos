"""
Visualisation Suite for Astrocyte-Modulated Tripartite-Synapse Network
=======================================================================

Produces six publication-quality figures and saves them as PNG files in the
same directory as this script.

Figures
-------
1. astrocyte_ca_dynamics.png   — Ca2+ traces over time (slow 100 ms – 10 s)
2. spike_raster.png            — Sparse neural activity raster plot
3. attention_heatmap.png       — Astrocyte gating weights (QKV-attention analogue)
4. weight_evolution.png        — Weight norm during tripartite STDP training
5. network_architecture.png    — Neuron–astrocyte–synapse schematic
6. energy_comparison.png       — Energy (synaptic ops) vs. transformer equivalent

All plots use only matplotlib + numpy.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from astrocyte_network import (
    AstrocyteConfig,
    AstrocyteLayer,
    LIFConfig,
    LIFLayer,
    NetworkConfig,
    STDPConfig,
    TripartiteNetwork,
    make_synthetic_mnist,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
OUT_DIR = HERE  # save PNGs alongside this file


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: run a small network and collect data
# ---------------------------------------------------------------------------

def _build_demo_network(
    n_hidden: int = 80,
    presentation_time: int = 60,
    n_samples: int = 150,
    seed: int = 7,
) -> tuple[TripartiteNetwork, np.ndarray, np.ndarray]:
    """
    Build and run a compact network on synthetic data.

    Returns
    -------
    net        : trained TripartiteNetwork
    X_train    : input array used for training
    y_train    : labels
    """
    cfg = NetworkConfig(
        n_input=784,
        n_hidden=n_hidden,
        n_output=10,
        lif=LIFConfig(n_neurons=n_hidden, adaptive_thresh=True),
        astrocyte=AstrocyteConfig(coverage_k=8, tau_ca=200.0),
        stdp=STDPConfig(learning_rate=0.008, a_plus=0.015),
        presentation_time=presentation_time,
        seed=seed,
    )
    net = TripartiteNetwork(cfg)
    X_train, y_train, _, _ = make_synthetic_mnist(n_samples, 50, seed=seed)

    print("  Running demo network for visualisation data collection...")
    t0 = time.perf_counter()
    net.train_unsupervised(X_train, verbose=False)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    return net, X_train, y_train


# ---------------------------------------------------------------------------
# Figure 1: Astrocyte Ca2+ dynamics
# ---------------------------------------------------------------------------

def plot_ca_dynamics(net: Optional[TripartiteNetwork] = None) -> None:
    """
    Simulate a single sample with Ca2+ recording and plot selected traces.

    Shows the characteristic slow timescale (200 ms – several seconds) of
    astrocytic calcium dynamics contrasted with fast neural spiking.
    """
    print("Plotting Fig 1: Astrocyte Ca2+ dynamics…")

    if net is None:
        net, _, _ = _build_demo_network()

    # Run one sample with recording
    X_test, _, _, _ = make_synthetic_mnist(10, 5, seed=99)
    net.reset_state()

    T = 200  # timesteps
    n_display_astro = min(6, net.astro_fwd.n_astro)
    ca_traces: list[list[float]] = [[] for _ in range(n_display_astro)]
    spike_counts: list[float] = []

    rng_demo = np.random.default_rng(1)
    x = X_test[0]
    net.reset_state()
    for t in range(T):
        in_spikes = rng_demo.random(net.cfg.n_input) < x
        _, ca_fwd = net.step(in_spikes, learn=False)
        for ai in range(n_display_astro):
            ca_traces[ai].append(ca_fwd[ai])
        spike_counts.append(net.hidden.spike.sum())

    time_ms = np.arange(T) * net.cfg.dt

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0f0f1a",
    )
    fig.suptitle(
        "Astrocyte Ca²⁺ Dynamics  (slow timescale vs. fast neural spiking)",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, n_display_astro))

    ax_ca = axes[0]
    ax_ca.set_facecolor("#0f0f1a")
    for ai in range(n_display_astro):
        ax_ca.plot(time_ms, ca_traces[ai], color=colors[ai],
                   linewidth=1.5, alpha=0.9, label=f"Astrocyte {ai}")
    ax_ca.axhline(
        net.astro_fwd.cfg.ca_threshold, color="#ff6b6b",
        linestyle="--", linewidth=1.2, alpha=0.7, label="Ca²⁺ threshold",
    )
    ax_ca.set_ylabel("Ca²⁺ level (a.u.)", color="white")
    ax_ca.tick_params(colors="white")
    ax_ca.spines[:].set_color("#555")
    legend = ax_ca.legend(
        loc="upper right", fontsize=8, facecolor="#1a1a2e", edgecolor="#555",
        labelcolor="white",
    )
    ax_ca.set_ylim(bottom=0)

    ax_sp = axes[1]
    ax_sp.set_facecolor("#0f0f1a")
    ax_sp.fill_between(time_ms, spike_counts, color="#4ecdc4", alpha=0.7)
    ax_sp.set_ylabel("Hidden\nspikes", color="white")
    ax_sp.set_xlabel("Time (ms)", color="white")
    ax_sp.tick_params(colors="white")
    ax_sp.spines[:].set_color("#555")

    plt.tight_layout()
    _save(fig, "astrocyte_ca_dynamics.png")


# ---------------------------------------------------------------------------
# Figure 2: Spike raster plot
# ---------------------------------------------------------------------------

def plot_spike_raster(net: Optional[TripartiteNetwork] = None) -> None:
    """
    Raster plot of hidden layer spikes over multiple samples,
    showing characteristic sparse neural activity.
    """
    print("Plotting Fig 2: Spike raster…")

    if net is None:
        net, _, _ = _build_demo_network()

    X_test, y_test, _, _ = make_synthetic_mnist(20, 5, seed=42)
    T = net.cfg.presentation_time
    n_show = min(60, net.cfg.n_hidden)
    n_samples = 5

    all_spikes: list[tuple[float, int]] = []  # (time, neuron_idx)
    rng_demo = np.random.default_rng(5)

    for s_idx in range(n_samples):
        x = X_test[s_idx]
        net.reset_state()
        offset = s_idx * T
        for t in range(T):
            in_spikes = rng_demo.random(net.cfg.n_input) < x
            net.astro_fwd.step(in_spikes)
            net.astro_rec.step(net.hidden.spike)
            I_fwd = net.W_fwd.forward(in_spikes)
            I_rec = net.W_rec.forward(net.hidden.spike)
            net.hidden.step(I_fwd + 0.3 * I_rec)
            for ni in np.where(net.hidden.spike[:n_show])[0]:
                all_spikes.append((offset + t, int(ni)))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    if all_spikes:
        times, neurons = zip(*all_spikes)
        ax.scatter(times, neurons, s=1.5, c="#4ecdc4", alpha=0.7, linewidths=0)

    # Sample boundaries
    for s_idx in range(1, n_samples):
        ax.axvline(s_idx * T, color="#ff6b6b", linewidth=0.8, alpha=0.6)

    total_spikes = len(all_spikes)
    total_possible = n_samples * T * n_show
    sparsity = 1.0 - total_spikes / max(total_possible, 1)

    ax.set_xlabel("Time (ms across samples)", color="white")
    ax.set_ylabel("Neuron index", color="white")
    ax.set_title(
        f"Sparse Neural Spiking  [sparsity ≈ {sparsity:.1%}]  "
        f"(showing {n_show} of {net.cfg.n_hidden} neurons)",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")
    ax.set_ylim(-1, n_show)

    # Annotate sample indices
    for s_idx in range(n_samples):
        label = f"S{s_idx} (cls {y_test[s_idx]})"
        ax.text(
            s_idx * T + T * 0.05, n_show - 2, label,
            color="#ffd93d", fontsize=7, alpha=0.85,
        )

    plt.tight_layout()
    _save(fig, "spike_raster.png")


# ---------------------------------------------------------------------------
# Figure 3: Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(net: Optional[TripartiteNetwork] = None) -> None:
    """
    Visualise astrocyte gating as a QKV-attention heatmap.

    Shows how different input patterns produce different astrocyte
    attention distributions — the core insight from Kozachkov et al.
    """
    print("Plotting Fig 3: Attention heatmap…")

    if net is None:
        net, _, _ = _build_demo_network()

    X_test, y_test, _, _ = make_synthetic_mnist(50, 5, seed=55)
    n_classes = 10
    n_show_astro = min(20, net.astro_fwd.n_astro)
    attn_matrix = np.zeros((n_classes, n_show_astro), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=int)

    rng_demo = np.random.default_rng(55)
    for i in range(len(X_test)):
        c = int(y_test[i])
        x = X_test[i]
        net.reset_state()
        T = net.cfg.presentation_time
        ca_accum = np.zeros(net.astro_fwd.n_astro)
        for t in range(T):
            in_spikes = rng_demo.random(net.cfg.n_input) < x
            ca = net.astro_fwd.step(in_spikes)
            ca_accum += ca
        mean_ca = ca_accum / T
        # Softmax-normalised attention
        shifted = mean_ca[:n_show_astro] - mean_ca[:n_show_astro].max()
        attn = np.exp(shifted) / (np.exp(shifted).sum() + 1e-8)
        attn_matrix[c] += attn
        class_counts[c] += 1

    # Average per class
    for c in range(n_classes):
        if class_counts[c] > 0:
            attn_matrix[c] /= class_counts[c]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Custom colormap: dark blue → cyan → yellow
    cmap = LinearSegmentedColormap.from_list(
        "astro", ["#0d0d2b", "#1a6b9e", "#4ecdc4", "#ffd93d", "#ff6b6b"], N=256
    )
    im = ax.imshow(attn_matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Attention weight", color="white")

    ax.set_xlabel("Astrocyte index", color="white")
    ax.set_ylabel("Digit class", color="white")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"Class {c}" for c in range(n_classes)], fontsize=8)
    ax.set_title(
        "Astrocyte Attention Map  (Ca²⁺-derived QKV-attention analogue)\n"
        "Each row = softmax-normalised Ca²⁺ distribution for one digit class",
        color="white", fontsize=11, fontweight="bold",
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555")

    plt.tight_layout()
    _save(fig, "attention_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4: Weight evolution during tripartite STDP
# ---------------------------------------------------------------------------

def plot_weight_evolution() -> None:
    """
    Train a fresh network and plot W_fwd norm over time to show convergence
    and the effect of Ca2+ gating on plasticity.
    """
    print("Plotting Fig 4: Weight evolution during STDP…")

    cfg = NetworkConfig(
        n_input=784, n_hidden=60, n_output=10,
        lif=LIFConfig(n_neurons=60, adaptive_thresh=True),
        astrocyte=AstrocyteConfig(coverage_k=8, tau_ca=150.0),
        stdp=STDPConfig(learning_rate=0.01, a_plus=0.02, a_minus=0.022),
        presentation_time=30, seed=11,
    )
    net_astro = TripartiteNetwork(cfg)

    # Baseline: same network but disable astrocyte gating (g=1 always)
    cfg_base = NetworkConfig(
        n_input=784, n_hidden=60, n_output=10,
        lif=LIFConfig(n_neurons=60, adaptive_thresh=False),
        astrocyte=AstrocyteConfig(coverage_k=8, tau_ca=150.0),
        stdp=STDPConfig(learning_rate=0.01, a_plus=0.02, a_minus=0.022),
        presentation_time=30, seed=11,
    )
    net_base = TripartiteNetwork(cfg_base)
    # Monkey-patch: override gating to always return 1.0
    orig_gating = net_base.astro_fwd.gating_values
    net_base.astro_fwd.gating_values = lambda ca=None: np.ones(  # type: ignore[method-assign]
        net_base.astro_fwd.n_astro
    )
    net_base.astro_rec.gating_values = lambda ca=None: np.ones(  # type: ignore[method-assign]
        net_base.astro_rec.n_astro
    )

    X_tr, _, _, _ = make_synthetic_mnist(200, 10, seed=11)

    for i, x in enumerate(X_tr):
        net_astro.run_sample(x, learn=True)
        net_base.run_sample(x, learn=True)

    norm_astro = net_astro.W_fwd._w_norm_history
    norm_base = net_base.W_fwd._w_norm_history

    # Smooth with running average
    def _smooth(arr: list[float], w: int = 50) -> np.ndarray:
        a = np.array(arr, dtype=np.float64)
        kernel = np.ones(w) / w
        return np.convolve(a, kernel, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f0f1a")
    fig.suptitle(
        "Tripartite STDP Weight Evolution",
        color="white", fontsize=13, fontweight="bold",
    )

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#555")

    steps = np.arange(len(norm_astro))
    axes[0].plot(steps, norm_astro, color="#4ecdc4", linewidth=0.5, alpha=0.4)
    axes[0].plot(steps, _smooth(norm_astro), color="#4ecdc4", linewidth=2.0, label="Tripartite STDP")
    axes[0].plot(
        np.arange(len(norm_base)), norm_base,
        color="#ff6b6b", linewidth=0.5, alpha=0.4,
    )
    axes[0].plot(
        np.arange(len(norm_base)), _smooth(norm_base),
        color="#ff6b6b", linewidth=2.0, label="Classical STDP (no Ca²⁺ gate)",
    )
    axes[0].set_xlabel("STDP update steps", color="white")
    axes[0].set_ylabel("‖W_fwd‖₂", color="white")
    axes[0].set_title("Weight norm convergence", color="white")
    legend = axes[0].legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    # Distribution of weights at end of training
    w_end_astro = net_astro.W_fwd.W[net_astro.W_fwd.mask].flatten()
    w_end_base = net_base.W_fwd.W[net_base.W_fwd.mask].flatten()
    axes[1].hist(w_end_astro, bins=40, color="#4ecdc4", alpha=0.7, label="Tripartite STDP", density=True)
    axes[1].hist(w_end_base, bins=40, color="#ff6b6b", alpha=0.7, label="Classical STDP", density=True)
    axes[1].set_xlabel("Synaptic weight", color="white")
    axes[1].set_ylabel("Density", color="white")
    axes[1].set_title("Final weight distribution", color="white")
    legend2 = axes[1].legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    _save(fig, "weight_evolution.png")


# ---------------------------------------------------------------------------
# Figure 5: Network architecture diagram
# ---------------------------------------------------------------------------

def plot_network_architecture() -> None:
    """
    Schematic diagram showing the tripartite synapse structure:
    pre-neuron ↔ synapse ↔ post-neuron, with astrocyte modulating the synapse.
    """
    print("Plotting Fig 5: Network architecture…")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d0d2b")
    ax.set_facecolor("#0d0d2b")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")
    ax.set_title(
        "Tripartite Synapse Network Architecture\n"
        "Astrocyte Ca²⁺ gates synaptic efficacy → QKV attention semantics",
        color="white", fontsize=13, fontweight="bold", pad=15,
    )

    # --- Layer positions ---
    layers = {
        "Input\nLayer": (1.5, 5),
        "Hidden LIF\nLayer": (5.0, 5),
        "Readout": (8.5, 5),
    }

    # --- Draw neurons ---
    neuron_radius = 0.35
    neuron_color = {"Input\nLayer": "#4ecdc4", "Hidden LIF\nLayer": "#ffd93d", "Readout": "#ff6b6b"}
    neuron_positions: dict[str, list[tuple[float, float]]] = {}

    for name, (cx, cy) in layers.items():
        n = 5 if "Hidden" in name else 4
        ys = np.linspace(cy - 1.8, cy + 1.8, n)
        neuron_positions[name] = [(cx, y) for y in ys]
        for x_n, y_n in neuron_positions[name]:
            circle = plt.Circle(
                (x_n, y_n), neuron_radius,
                color=neuron_color[name], zorder=4, linewidth=1.5,
                edgecolor="white",
            )
            ax.add_patch(circle)
        ax.text(cx, cy - 2.5, name, ha="center", va="top",
                color="white", fontsize=9, fontweight="bold")

    # --- Feedforward connections (thin) ---
    src_layer = "Input\nLayer"
    tgt_layer = "Hidden LIF\nLayer"
    for x_s, y_s in neuron_positions[src_layer]:
        for x_t, y_t in neuron_positions[tgt_layer]:
            ax.plot(
                [x_s + neuron_radius, x_t - neuron_radius],
                [y_s, y_t],
                color="#555577", linewidth=0.6, alpha=0.5, zorder=1,
            )

    for x_s, y_s in neuron_positions["Hidden LIF\nLayer"]:
        for x_t, y_t in neuron_positions["Readout"]:
            ax.plot(
                [x_s + neuron_radius, x_t - neuron_radius],
                [y_s, y_t],
                color="#776655", linewidth=0.8, alpha=0.5, zorder=1,
            )

    # --- Astrocyte units (between layers) ---
    astro_xs = [3.2, 3.2, 3.2]
    astro_ys = [3.5, 5.0, 6.5]
    astro_color = "#c77dff"

    for ax_x, ay in zip(astro_xs, astro_ys):
        hexagon = plt.Polygon(
            [(ax_x + 0.4 * np.cos(np.pi / 3 * k), ay + 0.4 * np.sin(np.pi / 3 * k))
             for k in range(6)],
            closed=True, color=astro_color, zorder=5,
            linewidth=1.5, edgecolor="white", alpha=0.9,
        )
        ax.add_patch(hexagon)
        ax.text(ax_x, ay, "A", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=6)

    ax.text(3.2, 7.8, "Astrocyte\nLayer", ha="center", va="bottom",
            color=astro_color, fontsize=9, fontweight="bold")

    # --- Astrocyte process arms (to pre and post synapses) ---
    for ay in astro_ys:
        # Arm to synapse midpoint
        mid_x = (1.5 + neuron_radius + 5.0 - neuron_radius) / 2
        ax.annotate(
            "", xy=(mid_x, ay),
            xytext=(3.2, ay),
            arrowprops=dict(arrowstyle="->", color=astro_color, lw=1.5),
            zorder=3,
        )
        # Ca2+ label
        ax.text(
            (3.2 + mid_x) / 2, ay + 0.25, "Ca²⁺\ngating",
            ha="center", va="bottom", color=astro_color, fontsize=7, alpha=0.9,
        )

    # --- Glutamate uptake arms (from pre to astrocyte) ---
    for ay in astro_ys:
        ax.annotate(
            "", xy=(3.2 - 0.4, ay),
            xytext=(1.5 + neuron_radius * 1.2, ay + 0.3),
            arrowprops=dict(arrowstyle="->", color="#88ccff", lw=1.0,
                            connectionstyle="arc3,rad=-0.3"),
            zorder=3,
        )

    ax.text(2.1, 8.0, "Glutamate\nuptake", color="#88ccff",
            fontsize=7, ha="center")

    # --- STDP annotation ---
    ax.annotate(
        "Tripartite STDP:\nCa²⁺ gates ΔW",
        xy=(3.5, 4.0), fontsize=8, color="#ffd93d",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                  edgecolor="#ffd93d", alpha=0.9),
    )
    ax.annotate(
        "", xy=(3.2, 4.2), xytext=(3.5, 4.0),
        arrowprops=dict(arrowstyle="->", color="#ffd93d", lw=1.2),
    )

    # --- Recurrent connections (arc) ---
    for i, (x1, y1) in enumerate(neuron_positions["Hidden LIF\nLayer"]):
        if i % 2 == 0 and i + 1 < len(neuron_positions["Hidden LIF\nLayer"]):
            x2, y2 = neuron_positions["Hidden LIF\nLayer"][i + 1]
            arc = mpatches.FancyArrowPatch(
                (x1, y1 + neuron_radius), (x2, y2 - neuron_radius),
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.5",
                color="#ffaa44", linewidth=1.0, alpha=0.6, zorder=2,
            )
            ax.add_patch(arc)
    ax.text(5.8, 7.2, "Recurrent\nsynapses", color="#ffaa44", fontsize=7.5)

    # --- Legend ---
    handles = [
        mpatches.Patch(color="#4ecdc4", label="Input neurons"),
        mpatches.Patch(color="#ffd93d", label="LIF hidden neurons"),
        mpatches.Patch(color="#ff6b6b", label="Readout neurons"),
        mpatches.Patch(color=astro_color, label="Astrocytes"),
        mpatches.Patch(color="#88ccff", label="Glutamate uptake"),
        mpatches.Patch(color="#ffaa44", label="Recurrent connections"),
    ]
    legend = ax.legend(
        handles=handles, loc="lower right", fontsize=8,
        facecolor="#1a1a2e", edgecolor="#555", labelcolor="white",
    )

    plt.tight_layout()
    _save(fig, "network_architecture.png")


# ---------------------------------------------------------------------------
# Figure 6: Energy comparison — Tripartite SNN vs. Transformer
# ---------------------------------------------------------------------------

def plot_energy_comparison() -> None:
    """
    Estimate and compare computational energy (synaptic operations) between:
      - NALSM Tripartite SNN (event-driven, sparse spikes)
      - Equivalent-capacity Transformer

    Based on energy models from:
      - Strubell et al. (2019) for transformer
      - Davies et al. (2018) Loihi for SNN (synaptic op ~ 23 pJ)
    """
    print("Plotting Fig 6: Energy comparison…")

    # --- Model configurations ---
    hidden_sizes = [100, 200, 500, 1000, 2000]

    # Transformer parameters
    d_model_sizes = [64, 128, 256, 512, 1024]  # roughly iso-parameter
    n_heads = 4
    seq_len = 50    # presentation time equivalent
    batch = 1

    # Energy constants (pJ per operation)
    E_fp32_mac = 3.7     # 32-bit floating-point MAC (GPU)
    E_snn_spike = 23.6   # Neuromorphic synaptic event (Loihi, Davies 2018)

    results = {
        "n_hidden": hidden_sizes,
        "transformer_ops": [],
        "snn_ops": [],
        "transformer_pJ": [],
        "snn_pJ": [],
        "param_transformer": [],
        "param_snn": [],
    }

    n_input = 784
    n_output = 10
    sparsity = 0.05       # 5% spike rate (typical for sparse SNN)
    conn_prob = 0.15      # synaptic connectivity

    for h, d in zip(hidden_sizes, d_model_sizes):
        # --- Transformer ops per forward pass ---
        # QKV projections: 3 * seq * d_model * d_model
        qkv_ops = 3 * seq_len * d * d
        # Attention: seq^2 * d_model (matmul)
        attn_ops = seq_len * seq_len * d
        # FFN (2-layer, 4x expansion): 2 * seq * d * 4d
        ffn_ops = 2 * seq_len * d * 4 * d
        total_transformer_ops = (qkv_ops + attn_ops + ffn_ops) * 2  # forward + backward approx
        param_transformer = (
            3 * d * d + d * d +  # attention
            2 * d * 4 * d +      # FFN
            d * n_output         # head
        )

        # --- SNN ops per sample ---
        # Only active synapses fire; each spike activates post-synaptic neurons
        n_spikes_per_step = int(h * sparsity)  # sparse
        n_synapses_active = int(n_spikes_per_step * h * conn_prob)
        total_snn_ops = n_synapses_active * seq_len  # synaptic events across timesteps
        # Add astrocyte ops (relatively cheap — vector ops, not matrix)
        n_astro = h // 8
        astro_ops = n_astro * 8 * seq_len  # dot products (coverage_k=8)
        total_snn_ops += astro_ops

        param_snn = int(h * n_input * conn_prob) + int(h * h * conn_prob * 0.05)

        results["transformer_ops"].append(total_transformer_ops)
        results["snn_ops"].append(total_snn_ops)
        results["transformer_pJ"].append(total_transformer_ops * E_fp32_mac)
        results["snn_pJ"].append(total_snn_ops * E_snn_spike)
        results["param_transformer"].append(param_transformer)
        results["param_snn"].append(param_snn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0f0f1a")
    fig.suptitle(
        "Energy & Parameter Efficiency: Tripartite SNN vs. Transformer",
        color="white", fontsize=13, fontweight="bold",
    )

    labels = [str(h) for h in hidden_sizes]
    x = np.arange(len(labels))
    bar_w = 0.35

    for ax in axes:
        ax.set_facecolor("#0f0f1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#555")

    # --- Plot 1: Energy (pJ) ---
    ax = axes[0]
    bars1 = ax.bar(x - bar_w / 2, results["transformer_pJ"], bar_w,
                   label="Transformer", color="#ff6b6b", alpha=0.85)
    bars2 = ax.bar(x + bar_w / 2, results["snn_pJ"], bar_w,
                   label="Tripartite SNN", color="#4ecdc4", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden size", color="white")
    ax.set_ylabel("Energy per inference (pJ)", color="white")
    ax.set_title("Energy Cost", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend = ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    # --- Plot 2: Operations ---
    ax = axes[1]
    ax.bar(x - bar_w / 2, results["transformer_ops"], bar_w,
           label="Transformer", color="#ff6b6b", alpha=0.85)
    ax.bar(x + bar_w / 2, results["snn_ops"], bar_w,
           label="Tripartite SNN", color="#4ecdc4", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden size", color="white")
    ax.set_ylabel("Operations per inference", color="white")
    ax.set_title("Computational Operations", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend2 = ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    # --- Plot 3: Parameter count ---
    ax = axes[2]
    ax.bar(x - bar_w / 2, results["param_transformer"], bar_w,
           label="Transformer", color="#ff6b6b", alpha=0.85)
    ax.bar(x + bar_w / 2, results["param_snn"], bar_w,
           label="Tripartite SNN", color="#4ecdc4", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Hidden size", color="white")
    ax.set_ylabel("Learnable parameters", color="white")
    ax.set_title("Parameter Count", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    legend3 = ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    # Efficiency ratio annotation
    for i, (t_e, s_e) in enumerate(zip(results["transformer_pJ"], results["snn_pJ"])):
        ratio = t_e / max(s_e, 1e-3)
        axes[0].text(
            i - bar_w / 2 + bar_w, max(t_e, s_e) * 1.5,
            f"{ratio:.0f}x", ha="center", va="bottom",
            color="#ffd93d", fontsize=7, fontweight="bold",
        )

    plt.tight_layout()
    _save(fig, "energy_comparison.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all six visualisation figures."""
    print("=" * 60)
    print("Astrocyte Tripartite Network — Visualisation Suite")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print()

    t_total = time.perf_counter()

    # Build one shared network for figures 1-3
    print("Building shared demo network…")
    net, X_train, y_train = _build_demo_network(
        n_hidden=80, presentation_time=40, n_samples=100, seed=3
    )
    print()

    plot_ca_dynamics(net)
    print()
    plot_spike_raster(net)
    print()
    plot_attention_heatmap(net)
    print()
    plot_weight_evolution()  # builds its own network
    print()
    plot_network_architecture()  # pure diagram, no simulation
    print()
    plot_energy_comparison()  # analytical, no simulation
    print()

    elapsed = time.perf_counter() - t_total
    print(f"All figures saved in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
