"""
Visualisations — Active-Dendrite NMDA Network
==============================================
Generates 6 publication-quality figures saved as PNG files.

Figures produced:
  1. dendritic_tree_structure.png
     Schematic of the multi-compartment neuron: soma, basal, oblique,
     and apical branches with input connectivity.

  2. nmda_plateau_traces.png
     Simulated NMDA plateau voltage traces showing the 50–200 ms
     sustained depolarisation for different co-active synapse counts.

  3. branch_task_allocation.png
     Heatmap: which branches are activated by which task contexts.
     Shows task-specific routing through dendritic arbour.

  4. catastrophic_forgetting_comparison.png
     Accuracy curves over sequential tasks for:
       - ActiveDendriteNetwork
       - EWC-MLP
       - Baseline-MLP (catastrophic forgetting)

  5. parameter_efficiency.png
     Bar chart comparing effective parameter count and accuracy-per-
     parameter for dendritic net vs point neuron MLP.

  6. context_gating_visualization.png
     How different context vectors shift branch activation thresholds
     and select task-specific computation paths.

Usage:
    python3 visualize.py

All PNGs written to ./plots/ (created if absent).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from dendrite_network import (
    ActiveDendriteNetwork,
    NetworkConfig,
    BranchConfig,
    sigmoid_plateau,
    hard_threshold_plateau,
    relu_plateau,
)
from continual_learning_demo import (
    TaskSpec,
    generate_task,
    run_dendrite_experiment,
    run_ewc_experiment,
    run_mlp_experiment,
    analyse_branch_utilisation,
)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {
    "dendrite": "#2E86AB",
    "ewc":      "#A23B72",
    "mlp":      "#F18F01",
    "apical":   "#E84855",
    "oblique":  "#3BB273",
    "basal":    "#6A4C93",
    "bg":       "#F7F7F7",
    "grid":     "#E0E0E0",
}

FONT = {
    "family": "DejaVu Sans",
    "size":   10,
}

plt.rcParams.update({
    "font.family":        FONT["family"],
    "font.size":          FONT["size"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     PALETTE["bg"],
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         PALETTE["grid"],
    "grid.linewidth":     0.8,
    "axes.axisbelow":     True,
})

PLOT_DIR = Path(__file__).parent / "plots"


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 1: Dendritic tree structure
# ---------------------------------------------------------------------------


def plot_dendritic_tree(n_branches: int = 8, n_inputs_shown: int = 6) -> Path:
    """
    Draw a schematic of the multi-compartment neuron with:
      - Soma (circle at centre)
      - Basal branches (downward)
      - Oblique branches (lateral)
      - Apical branches (upward, receiving context arrows)
      - Input lines to each branch
      - NMDA spike indicator on each branch segment
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_title(
        "Multi-Compartment Neuron — Active Dendritic Branches\n"
        "(Poirazi & Mel 2003; Cichon & Gan 2015)",
        fontsize=12, fontweight="bold", pad=12,
    )

    # --- Soma ---
    soma_pos = (0.0, 0.0)
    soma = plt.Circle(soma_pos, 0.35, color=PALETTE["dendrite"], zorder=5)
    ax.add_patch(soma)
    ax.text(0.0, 0.0, "Soma", ha="center", va="center", color="white",
            fontsize=8, fontweight="bold", zorder=6)

    # --- Branch layout ---
    compartment_defs = [
        # (label, angle_deg, length, color, n_this_comp)
        ("Basal",   -90, 2.5, PALETTE["basal"],   3),
        ("Oblique",   0, 2.0, PALETTE["oblique"],  2),
        ("Oblique", 180, 2.0, PALETTE["oblique"],  2),
        ("Apical",   90, 3.0, PALETTE["apical"],   1),
    ]

    branch_idx = 0
    nmda_spike_branches = {1, 4, 6}  # branches shown as "firing plateau"

    for label, base_angle_deg, length, color, n_comp in compartment_defs:
        for sub_i in range(n_comp):
            # Spread sub-branches around base_angle
            spread = 25 * (sub_i - (n_comp - 1) / 2)
            angle_deg = base_angle_deg + spread
            angle_rad = math.radians(angle_deg)

            # Branch endpoint
            x_end = soma_pos[0] + length * math.cos(angle_rad)
            y_end = soma_pos[1] + length * math.sin(angle_rad)

            # Draw branch segment
            lw = 3.5 if branch_idx in nmda_spike_branches else 2.0
            alpha = 1.0 if branch_idx in nmda_spike_branches else 0.7
            ax.plot(
                [soma_pos[0], x_end], [soma_pos[1], y_end],
                color=color, linewidth=lw, alpha=alpha, zorder=3,
                solid_capstyle="round",
            )

            # NMDA plateau indicator
            if branch_idx in nmda_spike_branches:
                mid_x = (soma_pos[0] + x_end) / 2
                mid_y = (soma_pos[1] + y_end) / 2
                plateau_circle = plt.Circle(
                    (mid_x, mid_y), 0.18,
                    color="#FFD700", zorder=4, linewidth=1.5,
                    edgecolor="#CC9900",
                )
                ax.add_patch(plateau_circle)
                ax.text(mid_x, mid_y, "P", ha="center", va="center",
                        fontsize=6, fontweight="bold", color="#5c4a00", zorder=5)

            # Input synapse lines
            perp_angle = angle_rad + math.pi / 2
            for s_idx in range(n_inputs_shown):
                t = 0.3 + 0.65 * s_idx / max(1, n_inputs_shown - 1)
                syn_x = soma_pos[0] + t * length * math.cos(angle_rad)
                syn_y = soma_pos[1] + t * length * math.sin(angle_rad)
                # Perpendicular input stub
                inp_x = syn_x + 0.5 * math.cos(perp_angle)
                inp_y = syn_y + 0.5 * math.sin(perp_angle)
                ax.plot(
                    [syn_x, inp_x], [syn_y, inp_y],
                    color="#555555", linewidth=0.8, alpha=0.5, zorder=2,
                )
                ax.scatter([inp_x], [inp_y], s=15, color="#333333", zorder=3, alpha=0.6)

            # Branch label at tip
            ax.text(
                x_end + 0.3 * math.cos(angle_rad),
                y_end + 0.3 * math.sin(angle_rad),
                f"b{branch_idx}\n({label[0]})",
                ha="center", va="center", fontsize=7, color=color,
                fontweight="bold",
            )

            branch_idx += 1

    # --- Apical context arrow ---
    ax.annotate(
        "Context (c)\napical tuft input",
        xy=(0, 3.4), xytext=(2.0, 4.8),
        arrowprops=dict(arrowstyle="->", color=PALETTE["apical"], lw=2),
        color=PALETTE["apical"], fontsize=9, fontweight="bold",
        ha="center",
    )

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(color=PALETTE["basal"],   label="Basal branches (feedforward)"),
        mpatches.Patch(color=PALETTE["oblique"], label="Oblique branches (mixed)"),
        mpatches.Patch(color=PALETTE["apical"],  label="Apical branches (context-gated)"),
        mpatches.Patch(color="#FFD700",          label="NMDA plateau active (P)"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        fontsize=8, framealpha=0.9, edgecolor="#cccccc",
    )

    ax.text(
        -4.8, 5.5,
        "y = f(Σ_b σ_b(w_b^T x_b + u_b^T c))",
        fontsize=10, fontstyle="italic", color="#333333",
    )

    return _save(fig, "dendritic_tree_structure.png")


# ---------------------------------------------------------------------------
# Figure 2: NMDA plateau voltage traces
# ---------------------------------------------------------------------------


def plot_nmda_plateau_traces() -> Path:
    """
    Simulate and plot NMDA plateau voltage traces for different numbers
    of co-active synapses.

    Biological reference:
      - Plateau duration 50–200 ms (Losonczy & Magee 2006)
      - Threshold: ~10–50 co-active synapses within 20–50 um
      - All-or-none character above threshold
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    fig.suptitle(
        "NMDA Plateau Potential Traces — Branch Sub-unit Nonlinearities\n"
        "(Losonczy & Magee 2006; Branco & Hausser 2011)",
        fontsize=11, fontweight="bold",
    )

    # Time axis: 0–300 ms
    T = 300
    t = np.linspace(0, T, 1000)
    dt_ms = T / 1000

    threshold = 0.4  # normalised branch activation threshold

    # Sub-threshold and supra-threshold activation levels
    activations = {
        "Sub-threshold (8 synapses, a=0.1)":   (0.10, "grey",    "--"),
        "Near-threshold (12 synapses, a=0.35)": (0.35, PALETTE["oblique"], "-."),
        "Threshold (15 synapses, a=0.42)":      (0.42, PALETTE["ewc"],    "-"),
        "Supra-threshold (25 syn, a=0.70)":     (0.70, PALETTE["dendrite"], "-"),
        "Strong drive (50 syn, a=0.95)":        (0.95, PALETTE["apical"],  "-"),
    }

    # Panel 1: Sigmoid plateau nonlinearity
    ax = axes[0]
    ax.set_title("Sigmoid (smooth threshold)", fontsize=9)
    for label, (a, color, ls) in activations.items():
        a_arr = np.full(len(t), a)
        # Convolve with plateau kernel to model sustained depolarisation
        output = sigmoid_plateau(a_arr, threshold=threshold, slope=12.0)
        # Add temporal dynamics: step onset at t=50ms, decay after t=220ms
        ramp = np.zeros(len(t))
        onset = int(50 / dt_ms)
        offset = int(220 / dt_ms)
        ramp[onset:offset] = output[0]
        # Smooth edges
        ramp = np.convolve(ramp, np.ones(20) / 20, mode="same")
        ax.plot(t, ramp, color=color, linestyle=ls, linewidth=1.8, label=label)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6, label="Threshold")
    ax.axvspan(50, 220, alpha=0.08, color="gold", label="Plateau window (170 ms)")
    ax.set_ylabel("Branch output σ(a)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(-0.05, 1.15)

    # Panel 2: Hard threshold (binary plateau indicator)
    ax = axes[1]
    ax.set_title("Hard threshold (all-or-none plateau)", fontsize=9)
    for label, (a, color, ls) in activations.items():
        a_arr = np.full(len(t), a)
        output = hard_threshold_plateau(a_arr, threshold=threshold)
        ramp = np.zeros(len(t))
        onset = int(50 / dt_ms)
        offset = int(220 / dt_ms)
        ramp[onset:offset] = output[0]
        # Add small noise to sub-threshold for visibility
        if a < threshold:
            ramp += np.random.default_rng(42).normal(0, 0.01, len(t))
        ax.plot(t, ramp, color=color, linestyle=ls, linewidth=1.8)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvspan(50, 220, alpha=0.08, color="gold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylim(-0.15, 1.25)

    # Panel 3: ReLU plateau (Poirazi & Mel 2003)
    ax = axes[2]
    ax.set_title("ReLU-plateau (Poirazi & Mel 2003)", fontsize=9)
    for label, (a, color, ls) in activations.items():
        a_arr = np.full(len(t), a)
        output = relu_plateau(a_arr, threshold=threshold)
        ramp = np.zeros(len(t))
        onset = int(50 / dt_ms)
        offset = int(220 / dt_ms)
        ramp[onset:offset] = output[0]
        ramp = np.convolve(ramp, np.ones(10) / 10, mode="same")
        ax.plot(t, ramp, color=color, linestyle=ls, linewidth=1.8)
    ax.axhline(threshold, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvspan(50, 220, alpha=0.08, color="gold")
    ax.set_xlabel("Time (ms)")

    # Add synapse count annotation
    for ax in axes:
        ax.text(260, 1.05, "Duration:\n~170 ms", fontsize=7, color="#555555",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

    plt.tight_layout()
    return _save(fig, "nmda_plateau_traces.png")


# ---------------------------------------------------------------------------
# Figure 3: Branch-task allocation heatmap
# ---------------------------------------------------------------------------


def plot_branch_task_allocation(
    branch_utilisation: np.ndarray,
    task_names: list[str],
) -> Path:
    """
    Heatmap showing which branches (y-axis) are active for each task (x-axis).

    branch_utilisation: shape (n_tasks, n_neurons, n_branches)
    Averaged across neurons to get (n_tasks, n_branches).
    """
    n_tasks, n_neurons, n_branches = branch_utilisation.shape
    # Average across neurons
    avg_util = branch_utilisation.mean(axis=1)  # (n_tasks, n_branches)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Branch-Specific Task Allocation\n"
        "(Cichon & Gan 2015 — different tasks recruit different branches)",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: Raw heatmap
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "dendrite_cmap", ["#f7fbff", "#2E86AB", "#0a3d5c"]
    )
    im = ax.imshow(
        avg_util.T,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=avg_util.max() + 1e-6,
        interpolation="nearest",
    )
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_yticks(range(n_branches))
    ax.set_yticklabels([f"b{i}" for i in range(n_branches)], fontsize=8)
    ax.set_xlabel("Task", fontsize=10)
    ax.set_ylabel("Branch index", fontsize=10)
    ax.set_title("Mean branch activation per task", fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean activation")

    # Panel 2: Binary allocation (threshold at 0.1)
    ax = axes[1]
    threshold = avg_util.max() * 0.15
    binary = (avg_util.T > threshold).astype(np.float64)
    im2 = ax.imshow(
        binary,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("bin", ["#f0f0f0", PALETTE["dendrite"]]),
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_yticks(range(n_branches))
    ax.set_yticklabels([f"b{i}" for i in range(n_branches)], fontsize=8)
    ax.set_xlabel("Task", fontsize=10)
    ax.set_title("Active branch allocation (binarised)", fontsize=9)
    plt.colorbar(im2, ax=ax, label="Active (1) / Silent (0)")

    # Compute and annotate per-task active fraction
    for t in range(n_tasks):
        active_frac = float(binary[:, t].mean())
        axes[1].text(
            t, n_branches + 0.2, f"{active_frac:.0%}",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
        )
    axes[1].text(
        n_tasks / 2 - 0.5, n_branches + 0.7, "Active fraction",
        ha="center", fontsize=8, color="#555555",
    )

    plt.tight_layout()
    return _save(fig, "branch_task_allocation.png")


# ---------------------------------------------------------------------------
# Figure 4: Catastrophic forgetting comparison
# ---------------------------------------------------------------------------


def plot_catastrophic_forgetting_comparison(
    dendrite_matrix: np.ndarray,
    ewc_matrix: np.ndarray,
    mlp_matrix: np.ndarray,
    task_names: list[str],
) -> Path:
    """
    Compare accuracy decay curves for three models as new tasks are learned.

    Shows:
      - Panel A: Accuracy on Task A as tasks B, C, D, E are added
      - Panel B: Heatmap of full accuracy matrix for each model
      - Panel C: Average forgetting bar chart
    """
    n_tasks = dendrite_matrix.shape[0]
    task_x = list(range(1, n_tasks + 1))

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        "Catastrophic Forgetting Comparison\n"
        "Active-Dendrite vs EWC vs Baseline MLP",
        fontsize=12, fontweight="bold",
    )

    # --- Panel A: Task retention curves for each task ---
    ax_retain = fig.add_subplot(gs[0, :2])
    colors = [PALETTE["dendrite"], PALETTE["ewc"], PALETTE["mlp"]]
    labels = ["ActiveDendriteNetwork", "EWC-MLP", "Baseline-MLP"]
    matrices = [dendrite_matrix, ewc_matrix, mlp_matrix]

    for mat, color, label in zip(matrices, colors, labels):
        # Plot accuracy of FIRST task as more tasks are learned
        retention = mat[:, 0]
        ax_retain.plot(
            task_x, retention,
            marker="o", color=color, linewidth=2.2, markersize=6, label=label,
        )

    ax_retain.axhline(dendrite_matrix[0, 0], color=PALETTE["dendrite"],
                      linestyle="--", linewidth=1, alpha=0.4, label="Task-A initial accuracy")
    ax_retain.set_xlabel("Task being trained", fontsize=10)
    ax_retain.set_ylabel(f"Accuracy on {task_names[0]}", fontsize=10)
    ax_retain.set_title(f"Retention of {task_names[0]} as new tasks are learned", fontsize=10)
    ax_retain.set_xticks(task_x)
    ax_retain.set_xticklabels([f"After\n{t}" for t in task_names], fontsize=8)
    ax_retain.legend(fontsize=8, loc="lower left")
    ax_retain.set_ylim(0, 1.05)

    # --- Panel B: Average forgetting bar chart ---
    ax_fgt = fig.add_subplot(gs[0, 2])

    def compute_avg_forgetting(mat: np.ndarray) -> list[float]:
        n = mat.shape[0]
        return [mat[n - 1, t] - mat[t, t] for t in range(n - 1)]

    fgt_d = compute_avg_forgetting(dendrite_matrix)
    fgt_e = compute_avg_forgetting(ewc_matrix)
    fgt_m = compute_avg_forgetting(mlp_matrix)

    avg_d = float(np.mean(fgt_d))
    avg_e = float(np.mean(fgt_e))
    avg_m = float(np.mean(fgt_m))

    bar_x = np.arange(3)
    bar_h = [avg_d, avg_e, avg_m]
    bar_colors = [PALETTE["dendrite"], PALETTE["ewc"], PALETTE["mlp"]]
    bars = ax_fgt.bar(bar_x, bar_h, color=bar_colors, width=0.6, alpha=0.85)
    ax_fgt.axhline(0, color="black", linewidth=0.8)
    ax_fgt.set_xticks(bar_x)
    ax_fgt.set_xticklabels(["Dendrite", "EWC", "MLP"], fontsize=9)
    ax_fgt.set_ylabel("Average BWT (forgetting)", fontsize=9)
    ax_fgt.set_title("Backward transfer\n(negative = forgetting)", fontsize=9)
    for bar, val in zip(bars, bar_h):
        ax_fgt.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.01 if val < 0 else val + 0.005,
            f"{val:+.3f}",
            ha="center", va="top" if val < 0 else "bottom",
            fontsize=8.5, fontweight="bold",
        )

    # --- Panels C, D, E: Accuracy matrices as heatmaps ---
    cmap_acc = LinearSegmentedColormap.from_list("acc", ["#ffe5e5", "#ffffff", "#e5f0ff", "#2E86AB"])
    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.imshow(mat, cmap=cmap_acc, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels([t[0] for t in task_names], fontsize=8)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([f"t={i+1}" for i in range(n_tasks)], fontsize=8)
        ax.set_xlabel("Eval task", fontsize=8)
        ax.set_ylabel("Train step", fontsize=8)
        ax.set_title(label, fontsize=8.5, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_tasks):
                val = mat[i, j]
                if j <= i:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if val > 0.4 else "white")

    return _save(fig, "catastrophic_forgetting_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: Parameter efficiency
# ---------------------------------------------------------------------------


def plot_parameter_efficiency(
    dendrite_params: int,
    mlp_params: int,
    dendrite_acc: float,
    mlp_acc: float,
) -> Path:
    """
    Bar chart comparing:
      - Total parameter count
      - Accuracy per 1000 parameters
      - Effective capacity (branches vs neurons)
    for dendritic net vs point-neuron MLP.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Parameter Efficiency: Active-Dendrite vs Point-Neuron MLP",
        fontsize=11, fontweight="bold",
    )

    models = ["ActiveDendriteNet", "Baseline-MLP"]
    colors = [PALETTE["dendrite"], PALETTE["mlp"]]

    # Panel 1: Parameter count
    ax = axes[0]
    params = [dendrite_params, mlp_params]
    bars = ax.bar(models, params, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel("Total trainable parameters", fontsize=9)
    ax.set_title("Parameter count", fontsize=9)
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Accuracy per 1000 params
    ax = axes[1]
    eff = [dendrite_acc / (dendrite_params / 1000), mlp_acc / (mlp_params / 1000)]
    bars = ax.bar(models, eff, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel("Accuracy per 1000 parameters", fontsize=9)
    ax.set_title("Parameter efficiency", fontsize=9)
    for bar, val in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 3: Capacity comparison (conceptual, based on Poirazi & Mel 2003)
    ax = axes[2]
    # Poirazi & Mel: dendritic neuron ~= 100x single point neuron in capacity
    capacity_ratio = 100  # multiplicative factor from Poirazi & Mel
    capacities = [capacity_ratio, 1]
    bars = ax.bar(models, capacities, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel("Relative capacity (point neuron = 1)", fontsize=9)
    ax.set_title("Memory capacity\n(Poirazi & Mel 2003)", fontsize=9)
    for bar, val in zip(bars, capacities):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"{val}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    return _save(fig, "parameter_efficiency.png")


# ---------------------------------------------------------------------------
# Figure 6: Context-gating visualisation
# ---------------------------------------------------------------------------


def plot_context_gating(
    net: ActiveDendriteNetwork,
    task_ids: list[int],
    task_names: list[str],
    x_sample: np.ndarray,
) -> Path:
    """
    Visualise how different context vectors gate branch activations.

    Shows:
      - Context vector patterns (SDR binary vectors)
      - Per-branch activation shift caused by context
      - Effective threshold modulation per branch
    """
    n_tasks = len(task_ids)
    n_ctx = net.config.n_context
    n_branches = net.config.n_branches

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Context-Gating: Apical Input Selects Task-Specific Branch Computations\n"
        "(Iyer et al. 2022 — sparse context vectors route through dendritic arbour)",
        fontsize=11, fontweight="bold",
    )

    # --- Panel A: Context vectors (SDR patterns) ---
    ax = axes[0, 0]
    ctx_matrix = np.zeros((n_tasks, n_ctx))
    for t_idx, t_id in enumerate(task_ids):
        ctx = net.get_task_context(t_id)
        ctx_matrix[t_idx] = ctx

    im = ax.imshow(
        ctx_matrix,
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list("sdr", ["#f0f0f0", PALETTE["apical"]]),
        interpolation="nearest",
    )
    ax.set_xticks(range(0, n_ctx, max(1, n_ctx // 8)))
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_xlabel("Context dimension", fontsize=9)
    ax.set_title("Task context vectors (SDR — ~5% active)", fontsize=9)
    plt.colorbar(im, ax=ax, label="Active (1) / Silent (0)")

    # Sparsity annotation
    for t_idx, ctx_row in enumerate(ctx_matrix):
        sparsity = float(ctx_row.mean())
        ax.text(n_ctx - 1, t_idx, f"{sparsity:.0%}", ha="right", va="center",
                fontsize=7, color="white", fontweight="bold")

    # --- Panel B: Branch activation shift per task ---
    ax = axes[0, 1]
    # For the first neuron, show how context shifts the u_b^T c term
    neuron_0 = net.neurons[0]
    branch_ctx_drives = np.zeros((n_tasks, n_branches))
    for t_idx, t_id in enumerate(task_ids):
        ctx = net.get_task_context(t_id)
        for b_idx, branch in enumerate(neuron_0.branches):
            ctx_drive = float(branch.u @ ctx)
            branch_ctx_drives[t_idx, b_idx] = ctx_drive

    im2 = ax.imshow(
        branch_ctx_drives,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-abs(branch_ctx_drives).max(),
        vmax=abs(branch_ctx_drives).max(),
        interpolation="nearest",
    )
    ax.set_xticks(range(n_branches))
    ax.set_xticklabels([f"b{i}" for i in range(n_branches)], fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_xlabel("Branch", fontsize=9)
    ax.set_title("Context drive  u_b^T c  per branch\n(positive = branch facilitated)", fontsize=9)
    plt.colorbar(im2, ax=ax, label="u_b^T c")

    # --- Panel C: Context overlap matrix ---
    ax = axes[1, 0]
    overlap = ctx_matrix @ ctx_matrix.T  # dot products
    norms = np.linalg.norm(ctx_matrix, axis=1, keepdims=True)
    cosine_sim = overlap / (norms @ norms.T + 1e-9)

    im3 = ax.imshow(
        cosine_sim,
        aspect="equal",
        cmap="Blues",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names, fontsize=9)
    ax.set_title("Context vector cosine similarity\n(low off-diagonal = good task separation)", fontsize=9)
    plt.colorbar(im3, ax=ax, label="Cosine similarity")
    for i in range(n_tasks):
        for j in range(n_tasks):
            ax.text(j, i, f"{cosine_sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if cosine_sim[i,j] < 0.7 else "white")

    # --- Panel D: Effective threshold modulation ---
    ax = axes[1, 1]
    base_thresh = net.config.nmda_threshold
    effective_thresholds = np.zeros((n_tasks, n_branches))
    for t_idx, t_id in enumerate(task_ids):
        ctx = net.get_task_context(t_id)
        for b_idx, branch in enumerate(neuron_0.branches):
            ctx_drive = float(branch.u @ ctx)
            # Effective threshold = base - ctx_drive (context lowers threshold)
            effective_thresholds[t_idx, b_idx] = base_thresh - ctx_drive * 0.1

    ax.set_title("Effective NMDA threshold per branch\n(context modulation)", fontsize=9)
    for t_idx, t_name in enumerate(task_names):
        ax.plot(
            range(n_branches),
            effective_thresholds[t_idx],
            marker="o", linewidth=1.8, markersize=5,
            label=t_name,
            alpha=0.85,
        )
    ax.axhline(base_thresh, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Base threshold ({base_thresh:.2f})")
    ax.set_xticks(range(n_branches))
    ax.set_xticklabels([f"b{i}" for i in range(n_branches)], fontsize=8)
    ax.set_xlabel("Branch", fontsize=9)
    ax.set_ylabel("Effective NMDA threshold", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")

    plt.tight_layout()
    return _save(fig, "context_gating_visualization.png")


# ---------------------------------------------------------------------------
# Main: run all plots
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all six visualisation figures."""
    print("\n" + "=" * 60)
    print("  Active-Dendrite Network — Generating Visualisations")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ---- Config ----
    N_INPUT = 128
    N_OUTPUT = 5
    N_TASKS = 5
    N_TRAIN = 300
    N_VAL = 100
    N_TEST = 100
    N_EPOCHS = 10

    cfg = NetworkConfig(
        n_neurons=32,
        n_input=N_INPUT,
        n_output=N_OUTPUT,
        n_context=32,
        n_branches=8,
        synapses_per_branch=N_INPUT // 8,
        input_allocation="disjoint",
        learning_rate=0.02,
        weight_decay=1e-4,
        ewc_lambda=400.0,
        nmda_threshold=0.15,
        plateau_duration=20,
        n_epochs=N_EPOCHS,
        batch_size=32,
        random_seed=42,
    )

    task_specs = [
        TaskSpec(
            task_id=t,
            name=f"Task-{chr(65+t)}",
            n_classes=N_OUTPUT,
            n_train=N_TRAIN,
            n_val=N_VAL,
            n_test=N_TEST,
            n_input=N_INPUT,
            noise_std=0.12 + t * 0.02,
            random_seed=t * 999,
        )
        for t in range(N_TASKS)
    ]
    task_names = [s.name for s in task_specs]
    task_ids = [s.task_id for s in task_specs]

    # ---- Generate data ----
    print("\nGenerating tasks...")
    tasks = [generate_task(spec) for spec in task_specs]

    # ---- Run experiments for accuracy data ----
    print("\nTraining models (this may take a minute)...")
    print("  Running ActiveDendriteNetwork...")
    dendrite_res = run_dendrite_experiment(tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS)
    print("  Running EWC baseline...")
    ewc_res = run_ewc_experiment(tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS)
    print("  Running MLP baseline...")
    mlp_res = run_mlp_experiment(tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS)

    # Build a trained network for context gating visualisation
    net = ActiveDendriteNetwork(cfg)
    for t_idx, (task_data, spec) in enumerate(zip(tasks, task_specs)):
        net.register_task(spec.task_id)
        net.train_task(
            task_id=spec.task_id,
            x_train=task_data["x_train"],
            y_train=task_data["y_train"],
            n_epochs=N_EPOCHS,
        )

    x_sample = tasks[0]["x_test"][0]

    # ---- Branch utilisation ----
    branch_analysis = analyse_branch_utilisation(dendrite_res, threshold=0.1)

    # ---- Compute parameter counts ----
    dendrite_params = net._count_parameters()
    n_hidden_mlp = cfg.n_neurons * cfg.n_branches // 2
    mlp_params = N_INPUT * n_hidden_mlp + n_hidden_mlp + n_hidden_mlp * N_OUTPUT + N_OUTPUT

    # ---- Generate figures ----
    print("\nGenerating figures...")

    print("\nFigure 1: Dendritic tree structure")
    plot_dendritic_tree(n_branches=8)

    print("Figure 2: NMDA plateau traces")
    plot_nmda_plateau_traces()

    if dendrite_res.branch_utilisation is not None:
        print("Figure 3: Branch-task allocation heatmap")
        plot_branch_task_allocation(dendrite_res.branch_utilisation, task_names)

    print("Figure 4: Catastrophic forgetting comparison")
    plot_catastrophic_forgetting_comparison(
        dendrite_res.accuracy_matrix,
        ewc_res.accuracy_matrix,
        mlp_res.accuracy_matrix,
        task_names,
    )

    print("Figure 5: Parameter efficiency")
    dendrite_final_acc = dendrite_res.average_accuracy()
    mlp_final_acc = mlp_res.average_accuracy()
    plot_parameter_efficiency(
        dendrite_params=dendrite_params,
        mlp_params=mlp_params,
        dendrite_acc=dendrite_final_acc,
        mlp_acc=mlp_final_acc,
    )

    print("Figure 6: Context-gating visualisation")
    plot_context_gating(net, task_ids, task_names, x_sample)

    print(f"\nAll figures written to: {PLOT_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
