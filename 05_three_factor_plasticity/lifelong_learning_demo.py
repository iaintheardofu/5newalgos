#!/usr/bin/env python3
"""
Lifelong Learning Demo — Three-Factor Plasticity System
========================================================

Demonstrates sequential task learning over 10+ tasks with:
  - No rehearsal buffer (data sovereignty compliant — task data never retained)
  - Sleep-replay triggered at configurable intervals
  - Neurogenesis preventing capacity saturation
  - Full forward and backward transfer measurements

This demo is designed to be self-contained and reproducible.

Usage
-----
    python3 lifelong_learning_demo.py

Output
------
  - Console table: per-task accuracy, BWT, FWT
  - runtime/three_factor_demo_results.json (if workforce dir is available)
  - runtime/three_factor_demo_log.jsonl (event trace)

Architecture note
-----------------
The demo intentionally uses NO task-specific data after training — each
task's dataset is generated, used for training, then discarded.  Sleep-replay
works entirely from the network's internal weight consolidation; it does not
store or replay actual training samples.

References
----------
Fremaux & Gerstner (2016). Frontiers in Neural Circuits.
Tadros et al. (2022). Nature Communications.
Sandia NL (2017). Neurogenesis Deep Learning. arXiv:1710.06759.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # workforce root

from three_factor_system import (
    ThreeFactorNetwork,
    SyntheticTaskGenerator,
    NeuromodulatorSignals,
    RewardPredictionError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_CONFIG = {
    # Task sequence
    "n_tasks": 12,
    "n_samples_per_task": 400,
    "n_features": 120,
    "n_hidden": 192,
    "n_classes": 6,
    "epochs_per_task": 3,
    "task_overlap": 0.25,  # fraction of feature space shared between tasks
    "seed": 42,

    # Three-factor hyperparameters
    "eta_hidden": 0.005,
    "eta_output": 0.010,
    "tau_e": 25.0,           # eligibility trace time constant

    # Sleep-replay
    "sleep_period": 400,     # run consolidation every N training steps

    # Neurogenesis
    "neuro_period": 400,     # run neurogenesis check every N steps
    "neuro_fraction": 0.05,  # replace 5% of hidden units per event

    # Reporting
    "eval_on_all_previous": True,   # evaluate on all previous tasks after each new task
    "verbose": True,
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _format_table(rows: list[list], headers: list[str]) -> str:
    """Format a list of rows as a fixed-width text table."""
    all_rows = [headers] + rows
    col_widths = [max(len(str(r[c])) for r in all_rows) + 2 for c in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    lines = [sep]
    for i, row in enumerate(all_rows):
        line = "|" + "|".join(f" {str(v):<{w-1}}" for v, w in zip(row, col_widths)) + "|"
        lines.append(line)
        if i == 0:
            lines.append(sep.replace("-", "="))
    lines.append(sep)
    return "\n".join(lines)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(str(tmp), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-task training and evaluation
# ---------------------------------------------------------------------------

class LifelongLearningSession:
    """
    Manages a sequential multi-task learning session using the three-factor
    network.  Tracks all metrics needed for BWT, FWT, and continual learning
    evaluation.

    No training data is retained between tasks — data sovereignty compliant.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config["seed"])

        # Task generator
        self.gen = SyntheticTaskGenerator(
            n_tasks=config["n_tasks"],
            n_samples=config["n_samples_per_task"],
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            overlap=config["task_overlap"],
            seed=config["seed"],
        )

        # Main learner
        self.net = ThreeFactorNetwork(
            n_input=config["n_features"],
            n_hidden=config["n_hidden"],
            n_classes=config["n_classes"],
            eta_hidden=config["eta_hidden"],
            eta_output=config["eta_output"],
            tau_e=config["tau_e"],
            sleep_period=config["sleep_period"],
            neuro_period=config["neuro_period"],
            neuro_fraction=config["neuro_fraction"],
            seed=config["seed"],
        )

        # Accuracy matrix: acc[trained_up_to_task_t, eval_task_k]
        n = config["n_tasks"]
        self.acc_matrix = np.full((n, n), np.nan)

        # Training history
        self.task_losses: dict[int, list[float]] = {}
        self.task_train_acc: dict[int, float] = {}
        self.task_train_time: dict[int, float] = {}

        # Event log
        self.event_log: list[dict] = []

        # Neuromodulator TD estimator
        self.rpe = RewardPredictionError(
            n_states=config["n_tasks"] + 1,
            alpha=0.1,
            gamma=0.9,
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def train_task(self, task_id: int) -> dict:
        """
        Train on task `task_id` and immediately discard the training data.

        Returns
        -------
        info : dict with loss history, final accuracy, sleep/neuro events
        """
        X, y = self.gen.get_task(task_id)
        n_samples = len(X)

        t0 = time.perf_counter()
        n_sleep_before = len(self.net.sleep_events)
        n_neuro_before = len(self.net.neuro_events)

        losses = self.net.train_on_task(
            X, y, task_id=task_id, epochs=self.cfg["epochs_per_task"], shuffle=True,
        )

        elapsed = time.perf_counter() - t0

        # Evaluate on training data (in-distribution accuracy)
        train_acc = self.net.evaluate(X, y)

        # Immediately discard training data — data sovereignty
        del X, y

        # TD reward prediction error for DA signal
        reward = float(train_acc)
        next_state = min(task_id + 1, self.cfg["n_tasks"])
        da = self.rpe.step(state=task_id, reward=reward, next_state=next_state)

        n_sleep_after = len(self.net.sleep_events)
        n_neuro_after = len(self.net.neuro_events)

        info = {
            "task_id": task_id,
            "n_samples": n_samples,
            "epochs": self.cfg["epochs_per_task"],
            "mean_loss": float(np.mean(losses)),
            "final_loss": float(losses[-1]) if losses else 0.0,
            "train_accuracy": round(train_acc, 4),
            "elapsed_s": round(elapsed, 3),
            "steps_this_task": len(losses),
            "sleep_events_triggered": n_sleep_after - n_sleep_before,
            "neuro_events_triggered": n_neuro_after - n_neuro_before,
            "da_signal": round(da, 4),
            "total_steps": self.net.step,
        }

        self.task_losses[task_id] = losses
        self.task_train_acc[task_id] = train_acc
        self.task_train_time[task_id] = elapsed

        self.event_log.append({"event": "task_trained", **info})
        return info

    def evaluate_all_seen(self, up_to_task: int) -> dict[int, float]:
        """
        Evaluate on all tasks 0..up_to_task.

        Data is freshly generated from the task generator (same seed),
        so this does NOT require retaining training data.
        """
        accs = {}
        for eval_task in range(up_to_task + 1):
            X_eval, y_eval = self.gen.get_task(eval_task)
            acc = self.net.evaluate(X_eval, y_eval)
            accs[eval_task] = round(acc, 4)
            del X_eval, y_eval
        return accs

    def run(self) -> dict:
        """
        Execute the full sequential training session.

        Returns
        -------
        results : comprehensive metrics dict
        """
        n_tasks = self.cfg["n_tasks"]
        verbose = self.cfg["verbose"]

        if verbose:
            print("\n" + "=" * 70)
            print("  Three-Factor Plasticity: Lifelong Learning Demo")
            print("  No rehearsal buffer | Data sovereignty compliant")
            print("=" * 70)
            print(f"  Tasks: {n_tasks} | Features: {self.cfg['n_features']} | "
                  f"Hidden: {self.cfg['n_hidden']} | Classes: {self.cfg['n_classes']}")
            print(f"  tau_e: {self.cfg['tau_e']} | Sleep every {self.cfg['sleep_period']} steps | "
                  f"Neurogenesis every {self.cfg['neuro_period']} steps")
            print("=" * 70 + "\n")

        t_session_start = time.perf_counter()

        for task_id in range(n_tasks):
            if verbose:
                print(f"--- Task {task_id + 1}/{n_tasks} ---")

            train_info = self.train_task(task_id)

            if self.cfg["eval_on_all_previous"]:
                accs = self.evaluate_all_seen(task_id)
                for eval_k, acc in accs.items():
                    self.acc_matrix[task_id, eval_k] = acc

                self.event_log.append({
                    "event": "evaluation",
                    "after_task": task_id,
                    "accuracies": accs,
                })

                if verbose:
                    acc_str = "  ".join(
                        f"T{k+1}:{v:.3f}" for k, v in accs.items()
                    )
                    print(f"  Train acc: {train_info['train_accuracy']:.3f}  "
                          f"Loss: {train_info['mean_loss']:.4f}  "
                          f"DA: {train_info['da_signal']:.3f}  "
                          f"Time: {train_info['elapsed_s']:.2f}s")
                    print(f"  Eval on all:  {acc_str}")
                    if train_info["sleep_events_triggered"] > 0:
                        print(f"  Sleep-replay: {train_info['sleep_events_triggered']} event(s)")
                    if train_info["neuro_events_triggered"] > 0:
                        print(f"  Neurogenesis: {train_info['neuro_events_triggered']} event(s)")
                    print()

        session_elapsed = time.perf_counter() - t_session_start

        # Compute BWT and FWT
        bwt = self._backward_transfer()
        fwt = self._forward_transfer()
        final_avg_acc = self._final_avg_accuracy()
        intransigence = self._intransigence()

        results = {
            "config": self.cfg,
            "acc_matrix": self.acc_matrix.tolist(),
            "bwt": round(bwt, 4),
            "fwt": round(fwt, 4),
            "final_avg_accuracy": round(final_avg_acc, 4),
            "intransigence": round(intransigence, 4),
            "total_steps": self.net.step,
            "total_sleep_events": len(self.net.sleep_events),
            "total_neuro_events": len(self.net.neuro_events),
            "session_elapsed_s": round(session_elapsed, 2),
            "task_train_accs": {str(k): round(v, 4) for k, v in self.task_train_acc.items()},
            "task_train_times": {str(k): round(v, 3) for k, v in self.task_train_time.items()},
            "network_stats": self.net.get_stats(),
        }

        if verbose:
            self._print_summary(results)

        return results

    # ------------------------------------------------------------------
    # Transfer metrics (Lopez-Paz & Ranzato, 2017 definitions)
    # ------------------------------------------------------------------

    def _backward_transfer(self) -> float:
        """
        BWT = (1/(T-1)) * sum_{t=2}^{T} sum_{k=1}^{t-1} (R[t,k] - R[k,k])

        Measures average forgetting on previous tasks.
        BWT < 0 = forgetting; BWT ≈ 0 = no forgetting; BWT > 0 = positive transfer
        """
        n = self.cfg["n_tasks"]
        total, count = 0.0, 0
        for t in range(1, n):
            for k in range(t):
                if not np.isnan(self.acc_matrix[t, k]) and not np.isnan(self.acc_matrix[k, k]):
                    total += self.acc_matrix[t, k] - self.acc_matrix[k, k]
                    count += 1
        return total / count if count > 0 else 0.0

    def _forward_transfer(self) -> float:
        """
        FWT = (1/(T-1)) * sum_{t=2}^{T} (R[t-1, t] - random_baseline)

        Measures how training on previous tasks helps on new tasks.
        FWT > 0 = beneficial forward transfer.
        """
        n = self.cfg["n_tasks"]
        chance = 1.0 / self.cfg["n_classes"]
        total, count = 0.0, 0
        for t in range(1, n):
            if not np.isnan(self.acc_matrix[t - 1, t]):
                total += self.acc_matrix[t - 1, t] - chance
                count += 1
        return total / count if count > 0 else 0.0

    def _final_avg_accuracy(self) -> float:
        """Average accuracy on all tasks evaluated after training task T."""
        n = self.cfg["n_tasks"]
        last_row = self.acc_matrix[n - 1, :]
        valid = last_row[~np.isnan(last_row)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def _intransigence(self) -> float:
        """
        Intransigence = (1/T) * sum_{t} (R_star[t] - R[t, t])

        where R_star[t] is the training accuracy on task t (in-distribution).
        Measures how hard it is to learn each new task (should be near 0
        if the network has enough capacity).
        """
        n = self.cfg["n_tasks"]
        total, count = 0.0, 0
        for t in range(n):
            if not np.isnan(self.acc_matrix[t, t]):
                total += self.task_train_acc.get(t, 0.0) - self.acc_matrix[t, t]
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_summary(self, results: dict) -> None:
        n = self.cfg["n_tasks"]

        print("\n" + "=" * 70)
        print("  LIFELONG LEARNING RESULTS")
        print("=" * 70)

        # Accuracy matrix table
        print("\nAccuracy Matrix  (row = trained up to task k, col = eval task)")
        print("(diagonal = in-task accuracy; below diagonal = retention)\n")

        headers = ["Trained↓\\Eval→"] + [f"T{t+1}" for t in range(n)]
        rows = []
        for t in range(n):
            row_vals = [f"T{t+1}"]
            for k in range(n):
                v = self.acc_matrix[t, k]
                if np.isnan(v):
                    row_vals.append("—")
                elif t == k:
                    row_vals.append(f"[{v:.3f}]")  # diagonal
                else:
                    row_vals.append(f"{v:.3f}")
            rows.append(row_vals)
        print(_format_table(rows, headers))

        # Transfer metrics
        print(f"\nTransfer Metrics:")
        print(f"  Backward Transfer (BWT):  {results['bwt']:+.4f}  "
              f"(< 0 = forgetting, 0 = stable, > 0 = improvement)")
        print(f"  Forward Transfer (FWT):   {results['fwt']:+.4f}  "
              f"(> 0 = learning helps future tasks)")
        print(f"  Final Average Accuracy:   {results['final_avg_accuracy']:.4f}")
        print(f"  Intransigence:            {results['intransigence']:.4f}  "
              f"(near 0 = sufficient capacity)")
        print(f"\nSystem Events:")
        print(f"  Total training steps:  {results['total_steps']}")
        print(f"  Sleep-replay events:   {results['total_sleep_events']}")
        print(f"  Neurogenesis events:   {results['total_neuro_events']}")
        print(f"  Session duration:      {results['session_elapsed_s']:.2f}s")
        print(f"  Steps/second:          {results['total_steps'] / max(results['session_elapsed_s'], 1e-6):.0f}")

        stats = results["network_stats"]
        print(f"\nFinal Network State:")
        print(f"  mean|W_hidden|:   {stats['mean_abs_W_hidden']:.6f}")
        print(f"  mean|e| (trace):  {stats['mean_trace']:.6f}")
        print(f"  DA:  {stats['DA']:.4f}  ACh: {stats['ACh']:.4f}  NE: {stats['NE']:.4f}")
        print("=" * 70 + "\n")

    def save_results(self, results: dict, output_dir: Optional[Path] = None) -> dict[str, Path]:
        """Save results JSON and event log JSONL."""
        if output_dir is None:
            # Try workforce runtime dir, fall back to HERE
            candidate = HERE.parent.parent / "runtime"
            output_dir = candidate if candidate.exists() else HERE

        results_path = output_dir / "three_factor_demo_results.json"
        log_path = output_dir / "three_factor_demo_log.jsonl"

        _write_json(results_path, results)
        for event in self.event_log:
            _append_jsonl(log_path, event)

        return {"results": results_path, "log": log_path}


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(config: dict) -> dict:
    """
    Compare four configurations on the same task sequence:
      1. Full system: three-factor + sleep + neurogenesis
      2. No sleep (three-factor + neurogenesis only)
      3. No neurogenesis (three-factor + sleep only)
      4. Baseline: three-factor only (no sleep, no neurogenesis)

    Returns final average accuracy and BWT for each configuration.
    """
    ablations = [
        ("Full (3F+Sleep+Neuro)", config["sleep_period"], config["neuro_period"]),
        ("No Sleep (3F+Neuro)", 10**9, config["neuro_period"]),
        ("No Neurogenesis (3F+Sleep)", config["sleep_period"], 10**9),
        ("Baseline (3F only)", 10**9, 10**9),
    ]

    results_by_config = {}

    for label, sleep_p, neuro_p in ablations:
        cfg = {**config, "sleep_period": sleep_p, "neuro_period": neuro_p, "verbose": False}
        session = LifelongLearningSession(cfg)
        r = session.run()
        results_by_config[label] = {
            "final_avg_accuracy": r["final_avg_accuracy"],
            "bwt": r["bwt"],
            "fwt": r["fwt"],
            "intransigence": r["intransigence"],
            "total_sleep_events": r["total_sleep_events"],
            "total_neuro_events": r["total_neuro_events"],
        }

    return results_by_config


def print_ablation_results(ablation_results: dict) -> None:
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Component Contributions")
    print("=" * 70)
    headers = ["Configuration", "Avg Acc", "BWT", "FWT", "Intransigence"]
    rows = []
    for label, r in ablation_results.items():
        rows.append([
            label,
            f"{r['final_avg_accuracy']:.4f}",
            f"{r['bwt']:+.4f}",
            f"{r['fwt']:+.4f}",
            f"{r['intransigence']:.4f}",
        ])
    print(_format_table(rows, headers))
    print()


# ---------------------------------------------------------------------------
# Neuromodulator dynamics report
# ---------------------------------------------------------------------------

def print_neuromodulator_report(session: LifelongLearningSession) -> None:
    """Print a summary of neuromodulator dynamics across the session."""
    if not session.net.modulator_history:
        print("  (No modulator history recorded)")
        return

    history = session.net.modulator_history
    DA_vals = [h["DA"] for h in history]
    ACh_vals = [h["ACh"] for h in history]
    NE_vals = [h["NE"] for h in history]
    M_vals = [h["M"] for h in history]

    print("\nNeuromodulator Summary:")
    print(f"  {'Signal':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 46)
    for name, vals in [("DA", DA_vals), ("ACh", ACh_vals), ("NE", NE_vals), ("M(t)", M_vals)]:
        arr = np.array(vals)
        print(f"  {name:<12} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")

    rpe_history = session.rpe.history
    if rpe_history:
        rpes = [r["rpe"] for r in rpe_history]
        print(f"\nTD Reward Prediction Errors (n={len(rpes)}):")
        arr = np.array(rpes)
        print(f"  Mean RPE: {arr.mean():+.4f}  |  Std: {arr.std():.4f}  |  "
              f"Positive: {(arr > 0).mean():.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nThree-Factor Neuromodulated Plasticity — Lifelong Learning Demo")
    print("Fremaux & Gerstner (2016) | Tadros et al. (2022) | Sandia (2017)")
    print("-" * 66)

    # Full session
    session = LifelongLearningSession(DEMO_CONFIG)
    results = session.run()

    # Neuromodulator report
    print_neuromodulator_report(session)

    # Ablation study (smaller config for speed)
    ablation_config = {
        **DEMO_CONFIG,
        "n_tasks": 6,
        "n_samples_per_task": 200,
        "n_hidden": 96,
        "n_features": 80,
        "n_classes": 4,
        "epochs_per_task": 2,
        "verbose": False,
    }
    print("\nRunning ablation study (6 tasks, 4 configurations)...")
    ablation_results = run_ablation_study(ablation_config)
    print_ablation_results(ablation_results)

    # Save results if possible
    try:
        saved_paths = session.save_results(results)
        print(f"Results saved:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")
    except Exception as e:
        print(f"  (Could not save results to runtime/: {e})")

    # Final summary line
    print(f"\nFinal average accuracy: {results['final_avg_accuracy']:.4f}")
    print(f"BWT: {results['bwt']:+.4f}  |  FWT: {results['fwt']:+.4f}")
    print(f"Sleep events: {results['total_sleep_events']}  |  "
          f"Neurogenesis events: {results['total_neuro_events']}")
    print(f"Total steps: {results['total_steps']}  |  "
          f"Time: {results['session_elapsed_s']:.2f}s\n")


if __name__ == "__main__":
    main()
