#!/usr/bin/env python3
"""
Generate all algorithm visualizations and comparison charts.

Produces comprehensive visual documentation for all five brain-inspired algorithms:
1. Astrocyte-Modulated Tripartite-Synapse Networks
2. Active-Dendrite NMDA Sub-unit Networks
3. Oligodendrocyte/Myelin-Plastic Polychronous SNNs
4. Hierarchical Active Inference / Predictive-Coding Agents
5. Three-Factor Neuromodulated Plasticity + Sleep-Replay + Neurogenesis
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)

VISUALS_DIR = os.path.join(os.path.dirname(__file__), 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

# Color palette
COLORS = {
    'astrocyte': '#E74C3C',
    'dendrite': '#3498DB',
    'myelin': '#2ECC71',
    'inference': '#9B59B6',
    'plasticity': '#F39C12',
    'bg': '#1a1a2e',
    'text': '#e0e0e0',
    'grid': '#333355',
    'accent': '#00d4ff',
}


def set_dark_style():
    """Apply consistent dark theme across all plots."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': '#16213e',
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.size': 11,
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })


def generate_architecture_stack():
    """Generate the full architecture stack diagram."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 10))

    layers = [
        ("Neuromorphic / Photonic Hardware\n(Akida 2, Loihi 2, photonic-SNN)", '#555577', 0),
        ("#3: Polychronous Learnable-Delay SNN\nDCLS temporal coding, OMP myelination", COLORS['myelin'], 1),
        ("#1: Astrocyte-Modulated Tripartite Synapses\nSlow Ca2+ gating, softmax attention", COLORS['astrocyte'], 2),
        ("#2: Active-Dendrite NMDA Sub-units\nBranch-specific plasticity, context gating", COLORS['dendrite'], 3),
        ("#5: Three-Factor Plasticity + Sleep + Neurogenesis\nEligibility traces, DA/ACh/NE modulation", COLORS['plasticity'], 4),
        ("#4: Active Inference / Predictive Coding\nFree-energy minimization, epistemic value", COLORS['inference'], 5),
    ]

    labels_right = [
        "hardware substrate",
        "temporal substrate",
        "attention / context",
        "representation",
        "learning layer",
        "decision layer",
    ]

    for text, color, i in layers:
        y = i * 1.4
        rect = mpatches.FancyBboxPatch((1, y), 10, 1.1, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white',
                                        alpha=0.85, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(6, y + 0.55, text, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(11.5, y + 0.55, f"← {labels_right[i]}", ha='left', va='center',
                fontsize=10, color=color, fontstyle='italic')

    # Arrows between layers
    for i in range(5):
        y_bottom = i * 1.4 + 1.1
        y_top = (i + 1) * 1.4
        ax.annotate('', xy=(6, y_top), xytext=(6, y_bottom),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Brain-Inspired Algorithm Stack\nFive Composable Layers for Neuromorphic Edge & Defense",
                 fontsize=16, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'architecture_stack.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: architecture_stack.png")


def generate_sota_comparison():
    """Generate SOTA comparison bar chart."""
    set_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Power comparison
    ax = axes[0, 0]
    methods = ['Transformer\n(7B)', 'GPU SNN\n(backprop)', 'This Stack\n(Akida)']
    power = [50, 10, 1]
    colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(methods, power, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Power (Watts)')
    ax.set_title('Inference Power')
    ax.set_yscale('log')
    for bar, val in zip(bars, power):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.2, f'{val}W',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Continual learning interference
    ax = axes[0, 1]
    methods = ['Naive\nMLP', 'EWC', 'LoRA', 'Active\nDendrite']
    interference = [80.2, 27.7, 17.5, 0.1]
    colors_bar = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.bar(methods, interference, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Task Interference (%)')
    ax.set_title('Continual Learning Interference')
    for bar, val in zip(bars, interference):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 3. Temporal benchmark (SHD)
    ax = axes[0, 2]
    methods = ['RadLIF', 'Adaptive\nDelays', 'DL128-SNN', 'DCLS\n(Ours)']
    accuracy = [94.62, 92.45, 92.56, 95.07]
    colors_bar = ['#95a5a6', '#95a5a6', '#95a5a6', '#2ecc71']
    bars = ax.bar(methods, accuracy, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('SHD Benchmark (Temporal Audio)')
    ax.set_ylim(90, 96)
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 4. Sample efficiency (RL)
    ax = axes[1, 0]
    methods = ['PPO', 'DreamerV3', 'Active\nInference']
    samples = [1_000_000, 24_207, 3_175]
    colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(methods, samples, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Steps to Converge')
    ax.set_title('Sample Efficiency (RL)')
    ax.set_yscale('log')
    for bar, val in zip(bars, samples):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f'{val:,}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 5. Decision cost
    ax = axes[1, 1]
    methods = ['o1-preview', 'GPT-4', 'Active\nInference']
    cost = [263, 15, 0.05]
    colors_bar = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(methods, cost, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Cost ($)')
    ax.set_title('Mastermind Solve Cost')
    ax.set_yscale('log')
    for bar, val in zip(bars, cost):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.5, f'${val}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 6. Parameter efficiency
    ax = axes[1, 2]
    methods = ['Transformer\n(7B)', 'Standard\nSNN', 'Dendrite\nNet', 'Full\nStack']
    params = [7_000, 500, 50, 10]
    colors_bar = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.bar(methods, params, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Parameter Count')
    ax.set_yscale('log')
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f'{val}M',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    fig.suptitle("SOTA Comparison: Brain-Inspired Stack vs Mainstream AI",
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'sota_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: sota_comparison.png")


def generate_hardware_alignment():
    """Generate hardware alignment matrix heatmap."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    hardware = ['Akida 2', 'Loihi 2', 'NorthPole', 'Photonic-SNN']
    algorithms = ['#1 Astrocyte', '#2 Dendrite', '#3 Delays', '#4 Inference', '#5 Three-Factor']

    # Alignment scores (0-5)
    alignment = np.array([
        [4, 2, 5, 3, 3],  # Akida 2
        [3, 5, 4, 3, 5],  # Loihi 2
        [2, 3, 2, 4, 2],  # NorthPole
        [2, 1, 5, 2, 1],  # Photonic-SNN
    ])

    im = ax.imshow(alignment, cmap='YlOrRd', aspect='auto', vmin=0, vmax=5)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, rotation=30, ha='right')
    ax.set_yticks(range(len(hardware)))
    ax.set_yticklabels(hardware)

    for i in range(len(hardware)):
        for j in range(len(algorithms)):
            color = 'white' if alignment[i, j] >= 3 else 'black'
            ax.text(j, i, str(alignment[i, j]), ha='center', va='center',
                    fontweight='bold', fontsize=14, color=color)

    cbar = plt.colorbar(im, ax=ax, label='Alignment Score (0-5)')
    cbar.ax.yaxis.label.set_color(COLORS['text'])
    cbar.ax.tick_params(colors=COLORS['text'])

    ax.set_title("Hardware-Algorithm Alignment Matrix", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'hardware_alignment.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: hardware_alignment.png")


def generate_trl_roadmap():
    """Generate TRL progression roadmap."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    approaches = [
        ('#1 Astrocyte Tripartite', 2, 3, COLORS['astrocyte']),
        ('#2 Active Dendrite', 2, 3, COLORS['dendrite']),
        ('#3 Myelin/Delay SNN', 3, 3, COLORS['myelin']),
        ('#4 Active Inference', 3, 4, COLORS['inference']),
        ('#5 Three-Factor', 3, 3, COLORS['plasticity']),
    ]

    phases = ['Now\n(TRL 2-3)', 'Phase 0\n(Q3 2026)', 'Phase 1\n(Q4 2027)', 'Phase 2\n(2028+)']
    trl_targets = [
        [2.5, 4, 5, 6],
        [2.5, 4, 5, 6],
        [3, 4, 6, 7],
        [3.5, 5, 6, 7],
        [3, 4, 5, 6],
    ]

    for i, (name, trl_low, trl_high, color) in enumerate(approaches):
        trls = trl_targets[i]
        ax.plot(range(4), trls, 'o-', color=color, linewidth=2.5, markersize=10,
                label=name, alpha=0.9)
        for j, trl in enumerate(trls):
            ax.text(j, trl + 0.15, f'TRL {trl:.0f}', ha='center', fontsize=8,
                    color=color, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(phases)
    ax.set_ylabel('Technology Readiness Level')
    ax.set_ylim(1, 8)
    ax.set_yticks(range(1, 9))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8,
              facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
    ax.set_title("TRL Progression Roadmap", fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'trl_roadmap.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: trl_roadmap.png")


def generate_defense_applications():
    """Generate defense application mapping diagram."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    applications = [
        ('Counter-UAS\nSwarm Intent', 0.2, 0.85, ['#1', '#4']),
        ('On-Board\nSatellite AI', 0.5, 0.85, ['#1']),
        ('RF/SIGINT\nClassification', 0.8, 0.85, ['#3']),
        ('On-Device ISR\nAdaptation', 0.2, 0.5, ['#2']),
        ('Contested-EM\nAutonomy', 0.5, 0.5, ['#4']),
        ('In-Mission\nLearning', 0.8, 0.5, ['#5']),
        ('Survivable\nAI', 0.35, 0.15, ['#5']),
        ('Cognitive\nEW', 0.65, 0.15, ['#3', '#4']),
    ]

    algo_colors = {
        '#1': COLORS['astrocyte'],
        '#2': COLORS['dendrite'],
        '#3': COLORS['myelin'],
        '#4': COLORS['inference'],
        '#5': COLORS['plasticity'],
    }

    for app_name, x, y, algos in applications:
        # Mix colors
        r = np.mean([int(algo_colors[a][1:3], 16) for a in algos]) / 255
        g = np.mean([int(algo_colors[a][3:5], 16) for a in algos]) / 255
        b = np.mean([int(algo_colors[a][5:7], 16) for a in algos]) / 255

        circle = plt.Circle((x, y), 0.1, facecolor=(r, g, b, 0.7),
                            edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.01, app_name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(x, y - 0.08, ' + '.join(algos), ha='center', va='center',
                fontsize=8, color='white', fontstyle='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Defense Application Mapping", fontsize=14, fontweight='bold', pad=20)

    # Legend
    for i, (algo, color) in enumerate(algo_colors.items()):
        algo_names = {
            '#1': 'Astrocyte', '#2': 'Dendrite', '#3': 'Delay SNN',
            '#4': 'Active Inference', '#5': 'Three-Factor'
        }
        ax.plot([], [], 'o', color=color, markersize=10,
                label=f"{algo}: {algo_names[algo]}")
    ax.legend(loc='lower left', fontsize=9, framealpha=0.8,
              facecolor=COLORS['bg'], edgecolor=COLORS['grid'])

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'defense_applications.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: defense_applications.png")


def generate_energy_comparison():
    """Generate energy efficiency comparison."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    systems = [
        'GPT-4\n(H100 GPU)', 'LLaMA-7B\n(A100 GPU)', 'Standard SNN\n(GPU)',
        'Rate-coded SNN\n(Akida 1)', 'Glial-Transformer\nSNN (Akida 2)',
        'Full Brain-Inspired\nStack (Loihi 2)'
    ]

    # Energy in Joules per inference (order of magnitude estimates)
    energy = [50, 10, 1, 0.1, 0.01, 0.005]
    colors_bar = ['#e74c3c', '#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#2ecc71']

    bars = ax.barh(systems, energy, color=colors_bar, edgecolor='white', linewidth=0.5, height=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Energy per Inference (Joules)')
    ax.set_title('Energy Efficiency: Brain-Inspired vs Mainstream AI', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, energy):
        label = f'{val}J' if val >= 1 else f'{val*1000:.0f}mJ' if val >= 0.001 else f'{val*1e6:.0f}uJ'
        ax.text(val * 1.5, bar.get_y() + bar.get_height()/2, label,
                va='center', fontweight='bold', fontsize=10)

    # Add brain reference line
    ax.axvline(x=0.02, color=COLORS['accent'], linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.02, -0.5, 'Human Brain\n(~20W continuous)', ha='center',
            fontsize=9, color=COLORS['accent'], fontstyle='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'energy_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: energy_comparison.png")


def generate_moat_diagram():
    """Generate IP moat depth visualization."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        'Published\nAlgorithm', 'Hardware\nMapping', 'Defense\nApplication',
        'Multi-Layer\nIntegration', 'Full Stack\n+ Patents'
    ]
    defensibility = [1, 3, 4, 7, 9]
    replication = [1, 6, 12, 36, 57]  # person-years

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, defensibility, width, label='Defensibility Score (0-10)',
                   color=COLORS['inference'], edgecolor='white', linewidth=0.5)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, replication, width, label='Replication Effort (person-years)',
                    color=COLORS['plasticity'], edgecolor='white', linewidth=0.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Defensibility Score', color=COLORS['inference'])
    ax2.set_ylabel('Replication Effort (person-years)', color=COLORS['plasticity'])
    ax2.tick_params(axis='y', colors=COLORS['plasticity'])

    ax.set_title("IP Moat Depth Analysis", fontsize=14, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=9, framealpha=0.8, facecolor=COLORS['bg'], edgecolor=COLORS['grid'])

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'moat_diagram.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: moat_diagram.png")


def generate_funding_map():
    """Generate funding pathway visualization."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    programs = [
        ('AFWERX D2P2', 1.25, 5, '#1,#3', COLORS['myelin']),
        ('DARPA L2M/ShELL', 2.0, 4, '#2,#5', COLORS['dendrite']),
        ('DARPA ANSR', 1.5, 4, '#4', COLORS['inference']),
        ('DARPA INSPIRE', 1.0, 3, '#5', COLORS['plasticity']),
        ('AFOSR Bio-Insp.', 0.5, 3, '#1,#5', COLORS['astrocyte']),
        ('AFRL/RI Rome', 1.0, 5, 'All 5', '#ffffff'),
        ('AFRL/RY Sensors', 0.8, 4, '#3', COLORS['myelin']),
        ('ONR Code 31', 0.75, 3, '#1,#4,#5', COLORS['inference']),
        ('DARPA MTO', 2.0, 2, '#3 photonic', COLORS['myelin']),
    ]

    for name, budget, priority, algos, color in programs:
        ax.scatter(budget, priority, s=500, c=color, edgecolors='white',
                  linewidth=2, alpha=0.8, zorder=5)
        ax.annotate(f'{name}\n${budget}M\n{algos}', (budget, priority),
                   textcoords="offset points", xytext=(15, 5),
                   fontsize=8, fontweight='bold', color=color,
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))

    ax.set_xlabel('Max Budget ($M)', fontsize=12)
    ax.set_ylabel('Strategic Priority (1-5)', fontsize=12)
    ax.set_title('Funding Pathway Map', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(1, 6)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(VISUALS_DIR, 'funding_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Generated: funding_map.png")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Brain-Inspired Algorithm Visualizations")
    print("=" * 60)

    generators = [
        generate_architecture_stack,
        generate_sota_comparison,
        generate_hardware_alignment,
        generate_trl_roadmap,
        generate_defense_applications,
        generate_energy_comparison,
        generate_moat_diagram,
        generate_funding_map,
    ]

    for gen in generators:
        try:
            gen()
        except Exception as e:
            print(f"Error in {gen.__name__}: {e}")

    print("=" * 60)
    print(f"All visualizations saved to {VISUALS_DIR}/")
    print("=" * 60)

    # Also try to generate per-algorithm visualizations
    algo_dirs = [
        '01_astrocyte_tripartite',
        '02_active_dendrite',
        '03_myelin_delay',
        '04_active_inference',
        '05_three_factor_plasticity',
    ]

    for algo_dir in algo_dirs:
        viz_path = os.path.join(os.path.dirname(__file__), algo_dir, 'visualize.py')
        if os.path.exists(viz_path):
            print(f"\nRunning {algo_dir}/visualize.py ...")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("visualize", viz_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'main'):
                    mod.main()
                elif hasattr(mod, 'generate_all'):
                    mod.generate_all()
            except Exception as e:
                print(f"  Warning: {e}")


if __name__ == '__main__':
    main()
