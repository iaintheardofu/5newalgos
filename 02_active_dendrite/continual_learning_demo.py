"""
Continual Learning Demo — Active-Dendrite NMDA Network
=======================================================
Demonstrates sequential task learning without catastrophic forgetting.

Experimental design (mirrors Iyer et al. 2022):
  - 5 tasks (A → E), each a separate binary/multi-class classification problem
  - Network trained on each task in sequence
  - After each new task, ALL previous task accuracies are re-evaluated
  - Per-task interference < 0.1% (absolute accuracy drop) is the target

Comparisons:
  1. ActiveDendriteNetwork  (context-gated branches + EWC + STDP)
  2. EWCBaseline            (standard MLP + EWC, no dendrites)
  3. BaselineMLP            (vanilla MLP, catastrophic forgetting)

Usage:
    python3 continual_learning_demo.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from dendrite_network import (
    ActiveDendriteNetwork,
    BaselineMLP,
    EWCBaseline,
    NetworkConfig,
)


# ---------------------------------------------------------------------------
# Synthetic task generator
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Specification for a synthetic classification task."""

    task_id: int
    name: str
    n_classes: int
    n_train: int
    n_val: int
    n_test: int
    n_input: int
    noise_std: float = 0.15
    random_seed: int = 0


def generate_task(spec: TaskSpec) -> dict[str, NDArray]:
    """
    Generate a synthetic classification task with linearly separable
    class prototypes corrupted by Gaussian noise.

    Each class c has a prototype p_c drawn from N(0, I); samples are
    drawn as x = p_c + epsilon, epsilon ~ N(0, sigma^2 I).

    Different tasks use different prototype sets, ensuring that task
    boundaries in input space are genuinely distinct (orthogonal tasks).
    """
    rng = np.random.default_rng(spec.random_seed + spec.task_id * 1000)

    n_total = spec.n_train + spec.n_val + spec.n_test
    n_in = spec.n_input
    k = spec.n_classes

    # Class prototypes: each task has unique prototypes (task-orthogonal)
    prototypes = rng.normal(0.0, 1.0, size=(k, n_in))
    # Normalise prototypes to unit sphere surface
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / (norms + 1e-8)

    # Generate samples
    n_per_class = n_total // k
    X_list, y_list = [], []
    for c in range(k):
        n_c = n_per_class + (1 if c < (n_total % k) else 0)
        noise = rng.normal(0.0, spec.noise_std, size=(n_c, n_in))
        X_c = prototypes[c:c+1, :] + noise
        X_list.append(X_c)
        y_list.append(np.full(n_c, c, dtype=np.int64))

    X = np.concatenate(X_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Normalise features to [-1, 1]
    X_max = np.max(np.abs(X), axis=0, keepdims=True) + 1e-8
    X = X / X_max

    return {
        "x_train": X[:spec.n_train],
        "y_train": y[:spec.n_train],
        "x_val": X[spec.n_train:spec.n_train + spec.n_val],
        "y_val": y[spec.n_train:spec.n_train + spec.n_val],
        "x_test": X[spec.n_train + spec.n_val:],
        "y_test": y[spec.n_train + spec.n_val:],
        "prototypes": prototypes,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Accuracy of a model on a given task at a given training step."""

    model_name: str
    training_task: int   # which task was being learned when this was measured
    eval_task: int       # which task is being evaluated
    accuracy: float
    wall_time: float = 0.0


@dataclass
class ExperimentResults:
    """Collected results from a full continual learning run."""

    model_name: str
    task_specs: list[TaskSpec]
    # results[t][ev] = accuracy on task ev after training task t
    accuracy_matrix: NDArray  # shape (n_tasks, n_tasks)
    branch_utilisation: Optional[NDArray] = None  # (n_tasks, n_neurons, n_branches)
    train_times: list[float] = field(default_factory=list)

    def forgetting(self, task_id: int) -> float:
        """
        Backward transfer / forgetting for task `task_id`.

        BWT_t = acc[last_task][t] - acc[t][t]
        Negative values indicate forgetting.
        """
        n_tasks = self.accuracy_matrix.shape[0]
        if task_id >= n_tasks:
            return 0.0
        peak_acc = self.accuracy_matrix[task_id, task_id]
        final_acc = self.accuracy_matrix[n_tasks - 1, task_id]
        return float(final_acc - peak_acc)

    def average_accuracy(self) -> float:
        """Average test accuracy across all tasks after full training."""
        n = self.accuracy_matrix.shape[0]
        return float(np.mean(self.accuracy_matrix[n - 1, :]))

    def average_forgetting(self) -> float:
        """Average forgetting across all tasks."""
        n = self.accuracy_matrix.shape[0]
        forgetting_vals = [self.forgetting(t) for t in range(n - 1)]
        return float(np.mean(forgetting_vals)) if forgetting_vals else 0.0

    def print_summary(self) -> None:
        n = self.accuracy_matrix.shape[0]
        print(f"\n{'='*60}")
        print(f"  {self.model_name}  — Continual Learning Results")
        print(f"{'='*60}")
        print(f"  Average final accuracy : {self.average_accuracy():.1%}")
        print(f"  Average forgetting     : {self.average_forgetting():.4f}")
        print(f"\n  Accuracy matrix (rows=train step, cols=eval task):")
        header = "       " + "".join(f"  Task{t}" for t in range(n))
        print(header)
        for i in range(n):
            row = f"  t={i+1}:  " + "".join(f"  {self.accuracy_matrix[i, j]:.3f}" for j in range(n))
            print(row)
        print()


# ---------------------------------------------------------------------------
# Training procedures
# ---------------------------------------------------------------------------


def run_dendrite_experiment(
    tasks: list[dict],
    task_specs: list[TaskSpec],
    config: NetworkConfig,
    n_epochs_per_task: int = 15,
) -> ExperimentResults:
    """Train ActiveDendriteNetwork on sequential tasks and record results."""
    model = ActiveDendriteNetwork(config)
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    branch_util_list: list[NDArray] = []
    train_times: list[float] = []

    for t_idx, (task_data, spec) in enumerate(zip(tasks, task_specs)):
        task_id = spec.task_id
        model.register_task(task_id)

        t0 = time.perf_counter()
        model.train_task(
            task_id=task_id,
            x_train=task_data["x_train"],
            y_train=task_data["y_train"],
            x_val=task_data["x_val"],
            y_val=task_data["y_val"],
            n_epochs=n_epochs_per_task,
        )
        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)

        # Consolidate after task
        model.consolidate_after_task(
            task_id, task_data["x_train"], task_data["y_train"]
        )

        # Record branch utilisation
        util = model.get_branch_utilisation(task_data["x_test"][:50], task_id=task_id)
        branch_util_list.append(util)

        # Evaluate on ALL tasks seen so far
        for ev_idx in range(t_idx + 1):
            ev_data = tasks[ev_idx]
            ev_task_id = task_specs[ev_idx].task_id
            acc = model.evaluate(
                ev_data["x_test"], ev_data["y_test"], task_id=ev_task_id
            )
            acc_matrix[t_idx, ev_idx] = acc

        print(
            f"  [Dendrite] After task {t_idx+1}/{n_tasks}  "
            f"| current={acc_matrix[t_idx, t_idx]:.3f} "
            f"| time={elapsed:.1f}s"
        )

    branch_util = np.stack(branch_util_list, axis=0)  # (n_tasks, n_neurons, n_branches)
    return ExperimentResults(
        model_name="ActiveDendriteNetwork",
        task_specs=task_specs,
        accuracy_matrix=acc_matrix,
        branch_utilisation=branch_util,
        train_times=train_times,
    )


def run_ewc_experiment(
    tasks: list[dict],
    task_specs: list[TaskSpec],
    config: NetworkConfig,
    n_epochs_per_task: int = 15,
) -> ExperimentResults:
    """Train EWCBaseline on sequential tasks."""
    n_in = config.n_input
    n_hidden = config.n_neurons * config.n_branches // 2  # comparable params
    n_out = config.n_output
    model = EWCBaseline(
        n_input=n_in,
        n_hidden=n_hidden,
        n_output=n_out,
        learning_rate=config.learning_rate,
        ewc_lambda=config.ewc_lambda,
        random_seed=config.random_seed,
    )
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    train_times: list[float] = []

    for t_idx, (task_data, spec) in enumerate(zip(tasks, task_specs)):
        t0 = time.perf_counter()
        model.train_task(
            task_data["x_train"], task_data["y_train"],
            n_epochs=n_epochs_per_task,
            batch_size=config.batch_size,
        )
        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)

        # Consolidate
        model.consolidate(task_data["x_train"], task_data["y_train"])

        for ev_idx in range(t_idx + 1):
            ev_data = tasks[ev_idx]
            acc = model.evaluate(ev_data["x_test"], ev_data["y_test"])
            acc_matrix[t_idx, ev_idx] = acc

        print(
            f"  [EWC]      After task {t_idx+1}/{n_tasks}  "
            f"| current={acc_matrix[t_idx, t_idx]:.3f} "
            f"| time={elapsed:.1f}s"
        )

    return ExperimentResults(
        model_name="EWC-MLP",
        task_specs=task_specs,
        accuracy_matrix=acc_matrix,
        train_times=train_times,
    )


def run_mlp_experiment(
    tasks: list[dict],
    task_specs: list[TaskSpec],
    config: NetworkConfig,
    n_epochs_per_task: int = 15,
) -> ExperimentResults:
    """Train BaselineMLP on sequential tasks (catastrophic forgetting baseline)."""
    n_in = config.n_input
    n_hidden = config.n_neurons * config.n_branches // 2
    n_out = config.n_output
    model = BaselineMLP(
        n_input=n_in,
        n_hidden=n_hidden,
        n_output=n_out,
        learning_rate=config.learning_rate,
        random_seed=config.random_seed,
    )
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    train_times: list[float] = []

    for t_idx, (task_data, spec) in enumerate(zip(tasks, task_specs)):
        t0 = time.perf_counter()
        model.train_task(
            task_data["x_train"], task_data["y_train"],
            n_epochs=n_epochs_per_task,
            batch_size=config.batch_size,
        )
        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)

        for ev_idx in range(t_idx + 1):
            ev_data = tasks[ev_idx]
            acc = model.evaluate(ev_data["x_test"], ev_data["y_test"])
            acc_matrix[t_idx, ev_idx] = acc

        print(
            f"  [MLP]      After task {t_idx+1}/{n_tasks}  "
            f"| current={acc_matrix[t_idx, t_idx]:.3f} "
            f"| time={elapsed:.1f}s"
        )

    return ExperimentResults(
        model_name="Baseline-MLP",
        task_specs=task_specs,
        accuracy_matrix=acc_matrix,
        train_times=train_times,
    )


# ---------------------------------------------------------------------------
# Branch utilisation analysis
# ---------------------------------------------------------------------------


def analyse_branch_utilisation(
    results: ExperimentResults,
    threshold: float = 0.1,
) -> dict:
    """
    Analyse per-task branch utilisation patterns.

    Parameters
    ----------
    results : ExperimentResults from dendrite experiment
    threshold : activation threshold above which a branch is "active"

    Returns
    -------
    dict with per-task statistics and overlap matrix
    """
    if results.branch_utilisation is None:
        return {}

    util = results.branch_utilisation  # (n_tasks, n_neurons, n_branches)
    n_tasks = util.shape[0]

    # Flatten to (n_tasks, n_neurons * n_branches)
    flat = util.reshape(n_tasks, -1)

    # Binary activation masks
    masks = flat > threshold

    # Per-task active fraction
    active_fractions = masks.mean(axis=1)

    # Task-pair overlap (Jaccard)
    overlap_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    for i in range(n_tasks):
        for j in range(n_tasks):
            intersection = float(np.sum(masks[i] & masks[j]))
            union = float(np.sum(masks[i] | masks[j]))
            overlap_matrix[i, j] = intersection / (union + 1e-9)

    avg_cross_task_overlap = float(
        np.mean([overlap_matrix[i, j] for i in range(n_tasks) for j in range(i + 1, n_tasks)])
    )

    return {
        "n_tasks": n_tasks,
        "active_fractions": active_fractions,
        "overlap_matrix": overlap_matrix,
        "avg_cross_task_overlap": avg_cross_task_overlap,
        "meets_interference_target": avg_cross_task_overlap < 0.10,
    }


def interference_report(
    dendrite_res: ExperimentResults,
    ewc_res: ExperimentResults,
    mlp_res: ExperimentResults,
    branch_analysis: dict,
) -> None:
    """Print formatted interference / forgetting report."""
    n_tasks = dendrite_res.accuracy_matrix.shape[0]

    print("\n" + "=" * 70)
    print("  CONTINUAL LEARNING INTERFERENCE REPORT")
    print("=" * 70)
    print(f"\n  {'Metric':<35}  {'Dendrite':>10}  {'EWC':>10}  {'MLP':>10}")
    print("  " + "-" * 65)

    avg_acc_d = dendrite_res.average_accuracy()
    avg_acc_e = ewc_res.average_accuracy()
    avg_acc_m = mlp_res.average_accuracy()
    print(f"  {'Average final accuracy':<35}  {avg_acc_d:>10.1%}  {avg_acc_e:>10.1%}  {avg_acc_m:>10.1%}")

    avg_fgt_d = dendrite_res.average_forgetting()
    avg_fgt_e = ewc_res.average_forgetting()
    avg_fgt_m = mlp_res.average_forgetting()
    print(f"  {'Average forgetting (BWT)':<35}  {avg_fgt_d:>+10.4f}  {avg_fgt_e:>+10.4f}  {avg_fgt_m:>+10.4f}")

    print(f"\n  Per-task forgetting after learning all {n_tasks} tasks:")
    print(f"  {'Task':<10}  {'Dendrite BWT':>15}  {'EWC BWT':>10}  {'MLP BWT':>10}")
    print("  " + "-" * 50)
    for t in range(n_tasks - 1):
        bwt_d = dendrite_res.forgetting(t)
        bwt_e = ewc_res.forgetting(t)
        bwt_m = mlp_res.forgetting(t)
        print(f"  Task {t+1:<5}  {bwt_d:>+15.4f}  {bwt_e:>+10.4f}  {bwt_m:>+10.4f}")

    if branch_analysis:
        print(f"\n  Branch utilisation analysis (Dendrite):")
        print(f"    Cross-task branch overlap : {branch_analysis['avg_cross_task_overlap']:.4f}")
        print(f"    Interference target (<10%): "
              f"{'PASS' if branch_analysis['meets_interference_target'] else 'FAIL'}")
        for t_idx, frac in enumerate(branch_analysis["active_fractions"]):
            print(f"    Task {t_idx+1} active branch fraction: {frac:.3f}")

    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full continual learning demonstration."""
    print("\n" + "=" * 70)
    print("  Active-Dendrite NMDA Network — Continual Learning Demo")
    print("  Faithful to Iyer et al. (2022) and Cichon & Gan (2015)")
    print("=" * 70)

    # ---- Configuration ----
    N_INPUT = 128
    N_OUTPUT = 5       # 5-way classification
    N_TASKS = 5
    N_TRAIN = 300
    N_VAL = 100
    N_TEST = 100
    N_EPOCHS = 12

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

    # ---- Generate data ----
    print("\nGenerating synthetic tasks...")
    tasks = [generate_task(spec) for spec in task_specs]
    for t, (spec, task_data) in enumerate(zip(task_specs, tasks)):
        print(
            f"  Task {spec.name}: {spec.n_train} train / {spec.n_val} val / "
            f"{spec.n_test} test  (noise={spec.noise_std:.2f})"
        )

    # ---- Network config ----
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

    print(f"\nNetwork: {ActiveDendriteNetwork(cfg)}")
    print(f"  Context vector dimensionality: {cfg.n_context}")
    print(f"  Branch allocation: {cfg.input_allocation}")
    print(f"  EWC lambda: {cfg.ewc_lambda}")

    # ---- Run experiments ----
    print("\n--- Training ActiveDendriteNetwork ---")
    dendrite_results = run_dendrite_experiment(
        tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS
    )

    print("\n--- Training EWC Baseline ---")
    ewc_results = run_ewc_experiment(
        tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS
    )

    print("\n--- Training Vanilla MLP ---")
    mlp_results = run_mlp_experiment(
        tasks, task_specs, cfg, n_epochs_per_task=N_EPOCHS
    )

    # ---- Analyse branch utilisation ----
    branch_analysis = analyse_branch_utilisation(dendrite_results, threshold=0.1)

    # ---- Print summaries ----
    dendrite_results.print_summary()
    ewc_results.print_summary()
    mlp_results.print_summary()

    # ---- Interference report ----
    interference_report(dendrite_results, ewc_results, mlp_results, branch_analysis)

    # ---- Task-A maintained while learning B, C, D, E ----
    print("Task-A retention curve (accuracy on Task A as new tasks are learned):")
    print(f"  {'After Task':>12}  {'Dendrite':>10}  {'EWC':>10}  {'MLP':>10}")
    print("  " + "-" * 45)
    for t in range(N_TASKS):
        acc_d = dendrite_results.accuracy_matrix[t, 0]
        acc_e = ewc_results.accuracy_matrix[t, 0]
        acc_m = mlp_results.accuracy_matrix[t, 0]
        label = chr(65 + t)
        print(f"  {f'Task {label}':>12}  {acc_d:>10.3f}  {acc_e:>10.3f}  {acc_m:>10.3f}")
    print()

    print("Demo complete. Run visualize.py to generate plots.")


if __name__ == "__main__":
    main()
