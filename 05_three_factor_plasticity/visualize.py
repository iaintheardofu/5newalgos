#!/usr/bin/env python3
"""
Visualizations for the Three-Factor Neuromodulated Plasticity System
====================================================================

Generates eight publication-quality PNG figures illustrating each subsystem:

  1. eligibility_trace_dynamics.png
     -- Eligibility trace e(t) decay and STDP-like accumulation over time
        for three synapse types (potentiating, depressing, neutral)

  2. neuromodulator_signals.png
     -- DA / ACh / NE time-series during a 600-step training run,
        annotated with task boundaries and correct/incorrect predictions

  3. sleep_replay_recovery.png
     -- Per-task accuracy before and after sleep-replay:
        three-factor + sleep vs three-factor alone vs naive SGD

  4. neurogenesis_unit_turnover.png
     -- Unit contribution scores before and after neurogenesis;
        histogram of replaced vs retained units

  5. continual_learning_no_replay.png
     -- 8-task sequential accuracy matrix (heatmap) for three methods:
        three-factor+sleep+neuro, three-factor only, naive SGD

  6. bwt_fwt_comparison.png
     -- Backward Transfer (BWT) and Forward Transfer (FWT) bar charts
        comparing all three methods

  7. on_device_learning_trajectory.png
     -- Rolling accuracy over 8 tasks for three-factor+sleep+neuro vs EWC-like
        (simulated) vs replay buffer (upper bound); shows lifelong improvement

  8. eligibility_weight_evolution.png
     -- Weight matrix mean |W| and trace mean |e| over training steps,
        annotated with sleep phases and neurogenesis events

All figures saved to the same directory as this script.

Usage
-----
    python3 visualize.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# Add parent algorithms directory + repo root to sys.path
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # workforce root

from three_factor_system import (
    EligibilityTraceBuffer,
    NeuromodulatorSignals,
    ThreeFactorNetwork,
    NeurogenesisRegularizer,
    SyntheticTaskGenerator,
    run_forgetting_benchmark,
)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

COLORS = {
    "DA": "#E63946",        # dopamine: crimson
    "ACh": "#2A9D8F",       # acetylcholine: teal
    "NE": "#F4A261",        # norepinephrine: amber
    "three_factor": "#457B9D",   # three-factor: steel blue
    "naive": "#E76F51",     # naive SGD: coral
    "ewc": "#6A4C93",       # EWC: purple
    "replay": "#2D6A4F",    # replay buffer: dark green
    "sleep": "#A8DADC",     # sleep event: light blue
    "neuro": "#F1FAEE",     # neurogenesis: off-white
    "correct": "#52B788",   # correct prediction: green
    "incorrect": "#E63946", # incorrect prediction: red
}

SAVE_DIR = HERE


def _save(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    path = SAVE_DIR / name
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path.name}")
    return path


# ---------------------------------------------------------------------------
# Figure 1: Eligibility trace dynamics
# ---------------------------------------------------------------------------

def plot_eligibility_trace_dynamics() -> Path:
    """
    Simulate three synapse types over 300 time steps:
      A: potentiating synapse (pre then post, close in time)
      B: depressing synapse  (post before pre)
      C: neutral synapse     (random uncorrelated spikes)

    Show how the trace decays between spikes and accumulates at coincidences.
    Also show the resulting weight trajectory when M(t) = 1.0 for all t.
    """
    T = 300
    tau_e = 20.0
    decay = 1.0 - 1.0 / tau_e
    eta = 0.008
    M = 1.0  # constant positive modulation

    rng = np.random.default_rng(0)

    def simulate_trace(pre_times: list, post_times: list) -> tuple:
        e = np.zeros(T)
        w = np.zeros(T)
        w[0] = 0.1
        current_e = 0.0
        for t in range(1, T):
            # Decay
            current_e *= decay
            # Spike coincidence: if both fired at t
            if t in pre_times and t in post_times:
                current_e += 1.0  # LTP: pre before post (or same)
            elif t in post_times and any(p > t - 5 and p < t for p in pre_times):
                current_e += 0.6  # Weak LTP: post slightly after pre
            elif t in pre_times and any(p > t and p < t + 5 for p in post_times):
                current_e -= 0.3  # LTD: pre after post
            current_e = float(np.clip(current_e, -1.0, 1.0))
            e[t] = current_e
            # Weight update
            dw = eta * current_e * M
            w[t] = float(np.clip(w[t - 1] + dw, 0.0, 1.0))
        return e, w

    # Synapse A: potentiating (pre fires 5ms before post, repeatedly)
    pre_A = list(range(20, T, 35))
    post_A = [p + 5 for p in pre_A]
    e_A, w_A = simulate_trace(pre_A, post_A)

    # Synapse B: depressing (post fires before pre)
    post_B = list(range(20, T, 35))
    pre_B = [p + 8 for p in post_B if p + 8 < T]
    e_B, w_B = simulate_trace(pre_B, post_B)

    # Synapse C: neutral (uncorrelated random spikes)
    pre_C = sorted(rng.integers(0, T, 20).tolist())
    post_C = sorted(rng.integers(0, T, 20).tolist())
    e_C, w_C = simulate_trace(pre_C, post_C)

    t_axis = np.arange(T)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        "Eligibility Trace Dynamics (Three-Factor Plasticity)\n"
        r"$\Delta w_{ij}(t) = \eta \cdot e_{ij}(t) \cdot M(t)$"
        "   |   Fremaux & Gerstner (2016)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    synapse_configs = [
        ("Synapse A — Potentiating\n(pre 5ms before post, LTP)", e_A, w_A,
         pre_A, post_A, "#457B9D", "#1D3557"),
        ("Synapse B — Depressing\n(post before pre, LTD)", e_B, w_B,
         pre_B, post_B, "#E63946", "#9D0208"),
        ("Synapse C — Neutral\n(uncorrelated spikes)", e_C, w_C,
         pre_C, post_C, "#2A9D8F", "#264653"),
    ]

    for row, (title, e, w, pre_t, post_t, color, dark) in enumerate(synapse_configs):
        ax_e = axes[row, 0]
        ax_w = axes[row, 1]

        # Eligibility trace
        ax_e.fill_between(t_axis, 0, e, alpha=0.25, color=color)
        ax_e.plot(t_axis, e, color=color, lw=1.5, label="e(t)")
        ax_e.axhline(0, color="gray", lw=0.5, ls="--")

        # Mark spike times
        spike_y = np.max(np.abs(e)) * 0.8 if np.max(np.abs(e)) > 0 else 0.4
        for pt in [p for p in pre_t if p < T]:
            ax_e.axvline(pt, color=dark, alpha=0.3, lw=0.8, ls=":")
        for pt in [p for p in post_t if p < T]:
            ax_e.axvline(pt, color=color, alpha=0.3, lw=0.8, ls="-.")

        ax_e.set_title(title, fontsize=10)
        ax_e.set_ylabel("Trace e(t)", fontsize=9)
        ax_e.set_xlim(0, T)
        ax_e.tick_params(labelsize=8)

        # Legend patch
        from matplotlib.lines import Line2D
        ax_e.legend(
            handles=[
                Line2D([0], [0], color=dark, ls=":", lw=1, label="pre-spike"),
                Line2D([0], [0], color=color, ls="-.", lw=1, label="post-spike"),
                Line2D([0], [0], color=color, lw=1.5, label="e(t)"),
            ],
            fontsize=7, loc="upper right",
        )

        # Weight trajectory
        ax_w.plot(t_axis, w, color=color, lw=1.8)
        ax_w.fill_between(t_axis, w.min(), w, alpha=0.15, color=color)
        ax_w.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6, label="w=0.5")
        ax_w.set_ylabel("Weight w(t)", fontsize=9)
        ax_w.set_xlim(0, T)
        ax_w.set_ylim(0.0, 1.0)
        ax_w.tick_params(labelsize=8)

        # Annotation: tau_e
        ax_e.text(
            0.01, 0.93,
            rf"$\tau_e$={tau_e:.0f}",
            transform=ax_e.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    axes[-1, 0].set_xlabel("Time step t", fontsize=9)
    axes[-1, 1].set_xlabel("Time step t", fontsize=9)
    fig.tight_layout()
    return _save(fig, "eligibility_trace_dynamics.png")


# ---------------------------------------------------------------------------
# Figure 2: Neuromodulator signals during learning
# ---------------------------------------------------------------------------

def plot_neuromodulator_signals() -> Path:
    """
    Train a small network for 3 tasks and record DA, ACh, NE over time.
    Annotate task boundaries, correct/incorrect predictions.
    """
    n_steps_per_task = 200
    n_tasks = 3
    n_features = 50
    n_classes = 4
    rng = np.random.default_rng(1)

    net = ThreeFactorNetwork(
        n_input=n_features, n_hidden=64, n_classes=n_classes,
        eta_hidden=0.008, eta_output=0.015, tau_e=20.0,
        sleep_period=10**9, neuro_period=10**9, seed=1,
    )

    DA_hist, ACh_hist, NE_hist, correct_hist = [], [], [], []
    task_boundaries = []

    for task_id in range(n_tasks):
        X = rng.normal(0, 1, (n_steps_per_task, n_features)).astype(np.float32)
        y = (task_id * np.ones(n_steps_per_task, dtype=int) % n_classes)
        y += rng.integers(0, 2, n_steps_per_task)
        y = y % n_classes
        task_boundaries.append(len(DA_hist))
        for i in range(n_steps_per_task):
            info = net.train_step(X[i], int(y[i]), task_id=task_id)
            DA_hist.append(info["DA"])
            ACh_hist.append(info["ACh"])
            NE_hist.append(info["NE"])
            correct_hist.append(info["correct"])

    T = len(DA_hist)
    t_axis = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Neuromodulator Signals During Sequential Task Learning\n"
        "DA = Reward Prediction Error | ACh = Attention/Gain | NE = Surprise/Arousal",
        fontsize=12, fontweight="bold",
    )

    # Task boundary shading
    task_colors = ["#EDF2FB", "#E2EAFC", "#D7E3FC"]
    for ax in axes:
        for tid, tb in enumerate(task_boundaries):
            end = task_boundaries[tid + 1] if tid + 1 < len(task_boundaries) else T
            ax.axvspan(tb, end, alpha=0.2, color=task_colors[tid % len(task_colors)], zorder=0)
            if ax is axes[0]:
                ax.text(
                    (tb + end) / 2, 2.6,
                    f"Task {tid + 1}",
                    ha="center", va="bottom", fontsize=9, color="#495057",
                )
        for tb in task_boundaries[1:]:
            ax.axvline(tb, color="#ADB5BD", lw=1.0, ls="--", zorder=1)

    # Smooth the signals with a running average for readability
    def smooth(x: list, w: int = 15) -> np.ndarray:
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    # DA
    axes[0].plot(t_axis, DA_hist, color=COLORS["DA"], alpha=0.25, lw=0.7)
    axes[0].plot(t_axis, smooth(DA_hist, 15), color=COLORS["DA"], lw=2.0, label="DA (dopamine)")
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[0].set_ylabel("DA", fontsize=10, color=COLORS["DA"])
    axes[0].set_ylim(0.0, 2.8)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].annotate("Reward\nprediction error", xy=(30, 1.8), fontsize=8, color="#6C757D")

    # ACh
    axes[1].plot(t_axis, ACh_hist, color=COLORS["ACh"], alpha=0.25, lw=0.7)
    axes[1].plot(t_axis, smooth(ACh_hist, 15), color=COLORS["ACh"], lw=2.0, label="ACh (acetylcholine)")
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_ylabel("ACh", fontsize=10, color=COLORS["ACh"])
    axes[1].set_ylim(0.0, 3.5)
    axes[1].legend(fontsize=9, loc="upper right")

    # NE
    axes[2].plot(t_axis, NE_hist, color=COLORS["NE"], alpha=0.25, lw=0.7)
    axes[2].plot(t_axis, smooth(NE_hist, 15), color=COLORS["NE"], lw=2.0, label="NE (norepinephrine)")
    axes[2].axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    axes[2].set_ylabel("NE", fontsize=10, color=COLORS["NE"])
    axes[2].set_ylim(0.0, 2.5)
    axes[2].legend(fontsize=9, loc="upper right")

    # Correct/incorrect raster
    for i, c in enumerate(correct_hist):
        axes[3].axvline(
            i,
            color=COLORS["correct"] if c else COLORS["incorrect"],
            alpha=0.5, lw=0.6,
        )

    # Rolling accuracy line
    window = 30
    rolling_acc = np.convolve(
        [float(c) for c in correct_hist],
        np.ones(window) / window,
        mode="same",
    )
    axes[3].plot(t_axis, rolling_acc, color="#1D3557", lw=2.0, label=f"Rolling acc (w={window})")
    axes[3].set_ylabel("Accuracy", fontsize=10)
    axes[3].set_ylim(0.0, 1.1)
    axes[3].legend(fontsize=9, loc="upper right")
    axes[3].set_xlabel("Training step", fontsize=10)

    fig.tight_layout()
    return _save(fig, "neuromodulator_signals.png")


# ---------------------------------------------------------------------------
# Figure 3: Sleep-replay recovery curves
# ---------------------------------------------------------------------------

def plot_sleep_replay_recovery() -> Path:
    """
    Train on tasks 1-5, then test accuracy on task 1 (the oldest) at
    regular checkpoints.  Compare: three-factor+sleep vs no-sleep.
    """
    n_tasks = 5
    n_samples = 250
    n_features = 80
    n_hidden = 96
    n_classes = 4
    epochs = 2

    gen = SyntheticTaskGenerator(
        n_tasks=n_tasks, n_samples=n_samples, n_features=n_features,
        n_classes=n_classes, seed=3,
    )

    configs = [
        ("Three-factor + Sleep\n(Tadros 2022)", COLORS["three_factor"], 150, 150),
        ("Three-factor, No Sleep", COLORS["naive"], 10**9, 150),
        ("Naive SGD (no 3F, no sleep)", COLORS["ewc"], 10**9, 10**9),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Sleep-Replay Consolidation: Accuracy Recovery on Task 1\n"
        "Tadros et al. (Nature Communications 2022)",
        fontsize=12, fontweight="bold",
    )

    checkpoints: dict[str, list] = {}
    task_labels: list[str] = []

    for label, color, sleep_p, neuro_p in configs:
        net = ThreeFactorNetwork(
            n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
            eta_hidden=0.006, eta_output=0.012, tau_e=20.0,
            sleep_period=sleep_p, neuro_period=neuro_p, seed=2,
        )

        X0, y0 = gen.get_task(0)  # task 1 held out for evaluation
        acc_over_time = [net.evaluate(X0, y0)]  # before any training

        for task_id in range(n_tasks):
            Xtr, ytr = gen.get_task(task_id)
            net.train_on_task(Xtr, ytr, task_id=task_id, epochs=epochs)
            acc = net.evaluate(X0, y0)
            acc_over_time.append(acc)

        checkpoints[label] = acc_over_time

        x_ticks = list(range(n_tasks + 1))
        task_labels = ["Before\ntraining"] + [f"After\nTask {i+1}" for i in range(n_tasks)]

        axes[0].plot(
            x_ticks, acc_over_time, "o-",
            color=color, lw=2.2, ms=7, label=label,
        )

    axes[0].set_title("Task 1 Accuracy Over Sequential Training", fontsize=11)
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(task_labels, fontsize=8)
    axes[0].set_ylabel("Accuracy on Task 1", fontsize=10)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].axhline(1.0 / n_classes, color="gray", ls="--", lw=0.8, label=f"Chance ({1/n_classes:.2f})")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: bar chart of final accuracy
    labels_short = ["3F+Sleep\n(Tadros)", "3F only\n(no sleep)", "Naive SGD"]
    final_accs = [checkpoints[c[0]][-1] for c in configs]
    bar_colors = [c[1] for c in configs]

    bars = axes[1].bar(labels_short, final_accs, color=bar_colors, width=0.5, alpha=0.85)
    axes[1].axhline(1.0 / n_classes, color="gray", ls="--", lw=0.8, label=f"Chance")
    for bar, acc in zip(bars, final_accs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[1].set_title("Final Task 1 Accuracy\n(After Training All 5 Tasks)", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].set_ylim(0.0, 1.1)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "sleep_replay_recovery.png")


# ---------------------------------------------------------------------------
# Figure 4: Neurogenesis unit turnover
# ---------------------------------------------------------------------------

def plot_neurogenesis_unit_turnover() -> Path:
    """
    Visualize neurogenesis: unit contribution scores before and after,
    histogram of replaced vs retained units.
    """
    rng = np.random.default_rng(5)
    n_features, n_hidden, n_classes = 60, 80, 5

    net = ThreeFactorNetwork(
        n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
        eta_hidden=0.007, eta_output=0.012, tau_e=15.0,
        sleep_period=10**9, neuro_period=10**9, seed=5,
    )

    # Train briefly so some units develop non-trivial contributions
    X = rng.normal(0, 1, (400, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, 400)
    net.train_on_task(X, y, task_id=0, epochs=2)

    # Capture contribution scores before neurogenesis
    scores_before = net.hidden.unit_contributions()

    # Apply one neurogenesis event
    neuro = NeurogenesisRegularizer(period=1, fraction=0.05, sigma_init=0.05, rng=rng)
    event = neuro._apply(net.hidden, step=1)
    replaced_idx = set(event["target_units"])

    scores_after = net.hidden.unit_contributions()

    unit_ids = np.arange(n_hidden)
    replaced_mask = np.array([i in replaced_idx for i in range(n_hidden)])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Neurogenesis-as-Regularization: Unit Contribution and Turnover\n"
        "Sandia National Laboratories (2017) — Preferential re-init of low-contribution units",
        fontsize=12, fontweight="bold",
    )

    # Panel A: contribution scores before — sorted
    sorted_idx = np.argsort(scores_before)
    colors_sorted = [COLORS["incorrect"] if replaced_mask[i] else COLORS["three_factor"]
                     for i in sorted_idx]
    axes[0, 0].bar(range(n_hidden), scores_before[sorted_idx], color=colors_sorted, width=1.0)
    axes[0, 0].set_title("Unit Contribution Scores Before Neurogenesis\n(sorted ascending)", fontsize=10)
    axes[0, 0].set_xlabel("Unit rank (low → high contribution)", fontsize=9)
    axes[0, 0].set_ylabel("Score = mean|act| × L1|W_out|", fontsize=9)
    axes[0, 0].axvline(len(replaced_idx) - 0.5, color="#E63946", lw=1.5, ls="--",
                       label=f"Cutoff: {event['fraction']:.0%} replaced")
    axes[0, 0].legend(fontsize=8)

    # Panel B: before vs after scatter
    axes[0, 1].scatter(
        scores_before[~replaced_mask], scores_after[~replaced_mask],
        c=COLORS["three_factor"], alpha=0.6, s=25, label="Retained units",
    )
    axes[0, 1].scatter(
        scores_before[replaced_mask], scores_after[replaced_mask],
        c=COLORS["incorrect"], alpha=0.8, s=50, marker="x", lw=2,
        label=f"Replaced units (n={len(replaced_idx)})",
    )
    max_score = max(scores_before.max(), scores_after.max())
    axes[0, 1].plot([0, max_score], [0, max_score], "k--", lw=0.8, alpha=0.4, label="y=x")
    axes[0, 1].set_title("Contribution Scores: Before vs After", fontsize=10)
    axes[0, 1].set_xlabel("Score before neurogenesis", fontsize=9)
    axes[0, 1].set_ylabel("Score after neurogenesis", fontsize=9)
    axes[0, 1].legend(fontsize=8)

    # Panel C: histogram of scores — replaced vs retained
    retained_scores = scores_before[~replaced_mask]
    replaced_scores = scores_before[replaced_mask]
    bins = np.linspace(0, scores_before.max() * 1.1, 25)
    axes[1, 0].hist(retained_scores, bins=bins, color=COLORS["three_factor"],
                    alpha=0.7, label="Retained", edgecolor="white")
    axes[1, 0].hist(replaced_scores, bins=bins, color=COLORS["incorrect"],
                    alpha=0.8, label="Replaced", edgecolor="white")
    axes[1, 0].set_title("Score Distribution: Replaced vs Retained", fontsize=10)
    axes[1, 0].set_xlabel("Contribution score", fontsize=9)
    axes[1, 0].set_ylabel("Count", fontsize=9)
    axes[1, 0].legend(fontsize=9)

    # Panel D: weight norm before/after for each unit
    W_norm_before = np.sum(np.abs(net.hidden.W), axis=1)  # after neuro, some units reset
    # We need original norms — reconstruct from stored snapshot conceptually
    # Instead show the weight norms split by replaced/retained
    axes[1, 1].scatter(
        unit_ids[~replaced_mask], W_norm_before[~replaced_mask],
        c=COLORS["three_factor"], alpha=0.6, s=20, label="Retained",
    )
    axes[1, 1].scatter(
        unit_ids[replaced_mask], W_norm_before[replaced_mask],
        c=COLORS["incorrect"], alpha=0.8, s=50, marker="x", lw=2,
        label=f"After re-init (n={len(replaced_idx)})",
    )
    axes[1, 1].set_title("Outgoing Weight L1 Norm After Neurogenesis", fontsize=10)
    axes[1, 1].set_xlabel("Unit index", fontsize=9)
    axes[1, 1].set_ylabel(r"$\Sigma_j |w_{ij}|$", fontsize=9)
    axes[1, 1].legend(fontsize=8)

    # Summary text
    fraction_pct = event["fraction"] * 100
    axes[0, 0].text(
        0.99, 0.95,
        f"Replaced: {event['n_replaced']} units ({fraction_pct:.1f}%)\n"
        f"Min replaced score: {event['min_score_replaced']:.4f}",
        transform=axes[0, 0].transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    fig.tight_layout()
    return _save(fig, "neurogenesis_unit_turnover.png")


# ---------------------------------------------------------------------------
# Figure 5: Continual learning accuracy matrix
# ---------------------------------------------------------------------------

def plot_continual_learning_matrix() -> Path:
    """
    Run the forgetting benchmark for three configurations and show
    accuracy matrices as heatmaps.
    """
    print("  Running forgetting benchmark (this may take ~30s)...")

    results = run_forgetting_benchmark(
        n_tasks=6,
        n_samples=200,
        n_features=80,
        n_hidden=96,
        n_classes=4,
        epochs_per_task=2,
        seed=7,
    )

    # Run a "three-factor only" (no sleep, no neurogenesis) for comparison
    gen = SyntheticTaskGenerator(
        n_tasks=6, n_samples=200, n_features=80, n_classes=4, seed=7,
    )
    net_3f_only = ThreeFactorNetwork(
        n_input=80, n_hidden=96, n_classes=4,
        eta_hidden=0.005, eta_output=0.01, tau_e=20.0,
        sleep_period=10**9, neuro_period=10**9, seed=7,
    )
    acc_3f_only = np.zeros((6, 6))
    for task_id in range(6):
        X, y = gen.get_task(task_id)
        net_3f_only.train_on_task(X, y, task_id=task_id, epochs=2)
        for eval_id in range(task_id + 1):
            Xe, ye = gen.get_task(eval_id)
            acc_3f_only[task_id, eval_id] = net_3f_only.evaluate(Xe, ye)

    cmap = LinearSegmentedColormap.from_list(
        "acc", ["#E63946", "#FFDD57", "#52B788"], N=256
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Continual Learning Accuracy Matrix: Accuracy[eval_task] after Training Task k\n"
        "(No rehearsal buffer — data sovereignty compliant)",
        fontsize=12, fontweight="bold",
    )

    matrices = [
        (results["acc_3f"], "Three-Factor\n+ Sleep + Neurogenesis"),
        (acc_3f_only, "Three-Factor Only\n(no sleep, no neurogenesis)"),
        (results["acc_naive"], "Naive SGD\n(catastrophic forgetting baseline)"),
    ]

    for ax, (mat, title) in zip(axes, matrices):
        n = mat.shape[0]
        # Mask upper triangle (future tasks not yet evaluated)
        mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
        display = np.where(mask, np.nan, mat)

        im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Evaluation Task", fontsize=9)
        ax.set_ylabel("After Training Task k", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"T{i+1}" for i in range(n)], fontsize=8)
        ax.set_yticklabels([f"T{i+1}" for i in range(n)], fontsize=8)

        # Cell annotations
        for i in range(n):
            for j in range(n):
                if not mask[i, j]:
                    color = "white" if display[i, j] < 0.4 else "black"
                    ax.text(j, i, f"{display[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return _save(fig, "continual_learning_no_replay.png")


# ---------------------------------------------------------------------------
# Figure 6: BWT and FWT comparison
# ---------------------------------------------------------------------------

def plot_bwt_fwt_comparison() -> Path:
    """
    Bar charts of BWT (backward transfer) and FWT (forward transfer)
    for three-factor+sleep+neuro vs three-factor-only vs naive SGD.
    """
    print("  Running BWT/FWT benchmark...")

    results = run_forgetting_benchmark(
        n_tasks=7, n_samples=200, n_features=80,
        n_hidden=96, n_classes=4, epochs_per_task=2, seed=11,
    )

    gen = SyntheticTaskGenerator(n_tasks=7, n_samples=200, n_features=80, n_classes=4, seed=11)
    net_3f_only = ThreeFactorNetwork(
        n_input=80, n_hidden=96, n_classes=4, eta_hidden=0.005, eta_output=0.01,
        tau_e=20.0, sleep_period=10**9, neuro_period=10**9, seed=11,
    )
    acc_3f_only = np.zeros((7, 7))
    for t in range(7):
        X, y = gen.get_task(t)
        net_3f_only.train_on_task(X, y, task_id=t, epochs=2)
        for e in range(t + 1):
            Xe, ye = gen.get_task(e)
            acc_3f_only[t, e] = net_3f_only.evaluate(Xe, ye)

    n_tasks, n_classes = 7, 4

    def bwt(m):
        vals = [m[t2, t] - m[t, t] for t in range(n_tasks - 1) for t2 in range(t + 1, n_tasks)]
        return np.mean(vals) if vals else 0.0

    def fwt(m):
        chance = 1.0 / n_classes
        vals = [m[t - 1, t] - chance for t in range(1, n_tasks)]
        return np.mean(vals) if vals else 0.0

    methods = ["3F+Sleep+\nNeurogenesis", "3F Only\n(no sleep)", "Naive SGD"]
    colors = [COLORS["three_factor"], COLORS["ACh"], COLORS["naive"]]

    bwt_vals = [bwt(results["acc_3f"]), bwt(acc_3f_only), bwt(results["acc_naive"])]
    fwt_vals = [fwt(results["acc_3f"]), fwt(acc_3f_only), fwt(results["acc_naive"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle(
        "Backward Transfer (BWT) and Forward Transfer (FWT)\n"
        "BWT ≈ 0 = no forgetting | FWT > 0 = learning helps future tasks",
        fontsize=12, fontweight="bold",
    )

    x = np.arange(len(methods))
    width = 0.5

    # BWT
    bars_b = axes[0].bar(x, bwt_vals, width, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
    axes[0].axhline(0, color="black", lw=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=10)
    axes[0].set_title("Backward Transfer (BWT)\nHigher = more forgetting", fontsize=11)
    axes[0].set_ylabel("Mean accuracy change on old tasks", fontsize=9)
    for bar, val in zip(bars_b, bwt_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.005 if val >= 0 else -0.015),
            f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].annotate(
        "Closer to 0 = better\n(less catastrophic forgetting)",
        xy=(0.5, 0.05), xycoords="axes fraction", ha="center", fontsize=8, color="#6C757D",
    )

    # FWT
    bars_f = axes[1].bar(x, fwt_vals, width, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=10)
    axes[1].set_title("Forward Transfer (FWT)\nHigher = more beneficial transfer", fontsize=11)
    axes[1].set_ylabel("Mean accuracy gain on new tasks", fontsize=9)
    for bar, val in zip(bars_f, fwt_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.003 if val >= 0 else -0.012),
            f"{val:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "bwt_fwt_comparison.png")


# ---------------------------------------------------------------------------
# Figure 7: On-device lifelong learning trajectory
# ---------------------------------------------------------------------------

def plot_on_device_learning_trajectory() -> Path:
    """
    Rolling accuracy across 8 sequential tasks for three methods:
      A. Three-factor + sleep + neurogenesis (our method)
      B. EWC-like (simulated: slower forgetting via L2 regularization toward
         original weights — approximates EWC penalty without full Fisher)
      C. Replay buffer oracle (upper bound)

    All methods: no external data at inference; data sovereignty compliant.
    """
    n_tasks = 8
    n_samples = 200
    n_features = 80
    n_hidden = 96
    n_classes = 4
    epochs = 2
    seed = 13

    gen = SyntheticTaskGenerator(
        n_tasks=n_tasks, n_samples=n_samples, n_features=n_features,
        n_classes=n_classes, seed=seed,
    )

    def rolling_acc(task_accs: list[list[float]], window: int = 2) -> np.ndarray:
        """For each step k, compute average accuracy over current + last `window` tasks."""
        result = []
        for k in range(n_tasks):
            start = max(0, k - window)
            result.append(np.mean(task_accs[start:k + 1]))
        return np.array(result)

    # Method A: Three-factor + sleep + neurogenesis
    net_A = ThreeFactorNetwork(
        n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
        eta_hidden=0.005, eta_output=0.01, tau_e=20.0,
        sleep_period=300, neuro_period=300, neuro_fraction=0.05, seed=seed,
    )
    acc_A_current: list[float] = []
    for t in range(n_tasks):
        X, y = gen.get_task(t)
        net_A.train_on_task(X, y, task_id=t, epochs=epochs)
        acc_A_current.append(net_A.evaluate(X, y))  # accuracy on current task

    # Method B: EWC-like (three-factor + L2 anchor to previous task weights)
    net_B = ThreeFactorNetwork(
        n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
        eta_hidden=0.005, eta_output=0.01, tau_e=20.0,
        sleep_period=10**9, neuro_period=10**9, seed=seed,
    )
    acc_B_current: list[float] = []
    W_anchor = net_B.hidden.snapshot_weights()
    ewc_lambda = 0.3
    for t in range(n_tasks):
        X, y = gen.get_task(t)
        # EWC-like: add L2 penalty toward anchor weights during training
        for xi, yi in zip(X, y):
            info = net_B.train_step(xi, int(yi), task_id=t)
            # Pull weights toward anchor (EWC regularization approximation)
            drift = net_B.hidden.W - W_anchor
            net_B.hidden.W -= ewc_lambda * net_B.hidden.eta * drift
        W_anchor = net_B.hidden.snapshot_weights()
        acc_B_current.append(net_B.evaluate(X, y))

    # Method C: Replay buffer (retrain on all previous tasks — upper bound)
    net_C = ThreeFactorNetwork(
        n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
        eta_hidden=0.005, eta_output=0.01, tau_e=20.0,
        sleep_period=10**9, neuro_period=10**9, seed=seed,
    )
    replay_X: list[np.ndarray] = []
    replay_y: list[np.ndarray] = []
    acc_C_current: list[float] = []
    for t in range(n_tasks):
        X, y = gen.get_task(t)
        replay_X.append(X)
        replay_y.append(y)
        # Train on all accumulated data
        all_X = np.concatenate(replay_X, axis=0)
        all_y = np.concatenate(replay_y, axis=0)
        net_C.train_on_task(all_X, all_y, task_id=t, epochs=epochs)
        acc_C_current.append(net_C.evaluate(X, y))

    task_axis = np.arange(1, n_tasks + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "On-Device Lifelong Learning Trajectory (No External Data)\n"
        "Three-Factor + Sleep + Neurogenesis vs EWC-like vs Replay Buffer (oracle upper bound)",
        fontsize=12, fontweight="bold",
    )

    # Left: per-task accuracy on current task
    axes[0].plot(task_axis, acc_A_current, "o-", color=COLORS["three_factor"],
                 lw=2.2, ms=8, label="3F + Sleep + Neurogenesis")
    axes[0].plot(task_axis, acc_B_current, "s--", color=COLORS["ewc"],
                 lw=2.0, ms=7, label="EWC-like (L2 anchor)")
    axes[0].plot(task_axis, acc_C_current, "^:", color=COLORS["replay"],
                 lw=2.0, ms=7, label="Replay buffer (oracle)")
    axes[0].axhline(1.0 / n_classes, color="gray", ls="--", lw=0.8, label="Chance")
    axes[0].set_xlabel("Task index (sequential order)", fontsize=10)
    axes[0].set_ylabel("Accuracy on current task", fontsize=10)
    axes[0].set_title("Per-Task Accuracy (Current Task)", fontsize=11)
    axes[0].set_xticks(task_axis)
    axes[0].set_xticklabels([f"T{t}" for t in task_axis])
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.3)

    # Right: cumulative average accuracy (all tasks seen so far)
    def cumulative_avg(accs: list[float]) -> np.ndarray:
        return np.array([np.mean(accs[:k + 1]) for k in range(len(accs))])

    axes[1].plot(task_axis, cumulative_avg(acc_A_current), "o-",
                 color=COLORS["three_factor"], lw=2.2, ms=8, label="3F + Sleep + Neurogenesis")
    axes[1].plot(task_axis, cumulative_avg(acc_B_current), "s--",
                 color=COLORS["ewc"], lw=2.0, ms=7, label="EWC-like")
    axes[1].plot(task_axis, cumulative_avg(acc_C_current), "^:",
                 color=COLORS["replay"], lw=2.0, ms=7, label="Replay buffer (oracle)")
    axes[1].axhline(1.0 / n_classes, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Number of tasks trained", fontsize=10)
    axes[1].set_ylabel("Cumulative average accuracy", fontsize=10)
    axes[1].set_title("Cumulative Average Accuracy\n(all tasks seen so far)", fontsize=11)
    axes[1].set_xticks(task_axis)
    axes[1].set_xticklabels([f"T{t}" for t in task_axis])
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "on_device_learning_trajectory.png")


# ---------------------------------------------------------------------------
# Figure 8: Weight and trace evolution
# ---------------------------------------------------------------------------

def plot_eligibility_weight_evolution() -> Path:
    """
    Show mean |W| and mean |e| over training steps, annotated with
    sleep phases and neurogenesis events.
    """
    n_features, n_hidden, n_classes = 60, 80, 5
    n_tasks = 5
    n_samples = 300
    seed = 17

    gen = SyntheticTaskGenerator(
        n_tasks=n_tasks, n_samples=n_samples, n_features=n_features,
        n_classes=n_classes, seed=seed,
    )

    net = ThreeFactorNetwork(
        n_input=n_features, n_hidden=n_hidden, n_classes=n_classes,
        eta_hidden=0.006, eta_output=0.012, tau_e=20.0,
        sleep_period=150, neuro_period=200, neuro_fraction=0.05, seed=seed,
    )

    # Collect per-step metrics
    W_means, trace_means, step_losses, step_accs = [], [], [], []
    task_start_steps: list[int] = []

    for task_id in range(n_tasks):
        X, y = gen.get_task(task_id)
        task_start_steps.append(net.step)
        indices = np.arange(len(X))
        np.random.default_rng(seed + task_id).shuffle(indices)
        for i in indices:
            info = net.train_step(X[i], int(y[i]), task_id=task_id)
            if net.step % 5 == 0:
                W_means.append(float(np.mean(np.abs(net.hidden.W))))
                trace_means.append(net.hidden.E.mean_abs)
                step_losses.append(info["loss"])
                step_accs.append(float(info["correct"]))

    sample_steps = np.arange(len(W_means)) * 5

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "Weight Matrix and Eligibility Trace Evolution During Lifelong Learning\n"
        "Annotated with Sleep Phases (blue) and Neurogenesis Events (orange)",
        fontsize=12, fontweight="bold",
    )

    # Shade task regions
    task_colors_bg = ["#F8F9FA", "#E9ECEF"]
    for i, ts in enumerate(task_start_steps):
        end = task_start_steps[i + 1] if i + 1 < len(task_start_steps) else sample_steps[-1] + 5
        for ax in axes:
            ax.axvspan(ts, end, alpha=0.25, color=task_colors_bg[i % 2], zorder=0)
        axes[0].text(
            (ts + end) / 2, axes[0].get_ylim()[1] if axes[0].get_ylim()[1] != 0 else 0.01,
            f"Task {i+1}", ha="center", va="bottom", fontsize=8.5, color="#6C757D",
        )

    # Annotate task boundaries
    for ax in axes:
        for ts in task_start_steps[1:]:
            ax.axvline(ts, color="#ADB5BD", lw=1.0, ls="--", zorder=1)

    # Annotate sleep events
    for se in net.sleep_events:
        step = se["step"]
        for ax in axes:
            ax.axvspan(step - 20, step + 20, color=COLORS["sleep"], alpha=0.4, zorder=2)

    # Annotate neurogenesis events
    for ne in net.neuro_events:
        step = ne["step"]
        for ax in axes:
            ax.axvline(step, color="#F4A261", lw=1.8, ls="-.", alpha=0.9, zorder=3)

    # Panel 1: mean |W|
    axes[0].plot(sample_steps, W_means, color=COLORS["three_factor"], lw=1.8, label="mean|W|")
    axes[0].set_ylabel("Mean |W| (hidden layer)", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.25)

    # Panel 2: mean |e|
    axes[1].plot(sample_steps, trace_means, color=COLORS["DA"], lw=1.5, label="mean|e| (eligibility trace)")
    axes[1].set_ylabel("Mean |e| (eligibility trace)", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(alpha=0.25)

    # Panel 3: rolling accuracy
    window = 40
    rolling = np.convolve(step_accs, np.ones(window) / window, mode="same")
    axes[2].plot(sample_steps, rolling, color=COLORS["replay"], lw=2.0, label=f"Rolling accuracy (w={window})")
    axes[2].axhline(1.0 / n_classes, color="gray", ls="--", lw=0.8, label="Chance")
    axes[2].set_ylabel("Accuracy (rolling)", fontsize=10)
    axes[2].set_xlabel("Training step", fontsize=10)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_ylim(0.0, 1.1)
    axes[2].grid(alpha=0.25)

    # Legend patches for annotations
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["sleep"], alpha=0.6, label="Sleep-replay phase"),
        Line2D([0], [0], color=COLORS["NE"], lw=2.0, ls="-.", label="Neurogenesis event"),
        Line2D([0], [0], color="#ADB5BD", lw=1.0, ls="--", label="Task boundary"),
    ]
    axes[0].legend(handles=legend_elements + [Line2D([0], [0], color=COLORS["three_factor"],
                   lw=1.8, label="mean|W|")], fontsize=8, loc="upper right")

    fig.tight_layout()
    return _save(fig, "eligibility_weight_evolution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating Three-Factor Plasticity visualizations...\n")

    plots = [
        ("Figure 1: Eligibility trace dynamics", plot_eligibility_trace_dynamics),
        ("Figure 2: Neuromodulator signals", plot_neuromodulator_signals),
        ("Figure 3: Sleep-replay recovery", plot_sleep_replay_recovery),
        ("Figure 4: Neurogenesis unit turnover", plot_neurogenesis_unit_turnover),
        ("Figure 5: Continual learning matrix", plot_continual_learning_matrix),
        ("Figure 6: BWT/FWT comparison", plot_bwt_fwt_comparison),
        ("Figure 7: On-device learning trajectory", plot_on_device_learning_trajectory),
        ("Figure 8: Weight and trace evolution", plot_eligibility_weight_evolution),
    ]

    paths = []
    for title, fn in plots:
        print(f"{title}")
        path = fn()
        paths.append(path)

    print(f"\nAll {len(paths)} figures saved to:\n  {SAVE_DIR}")


if __name__ == "__main__":
    main()
