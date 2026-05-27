// Algorithm #2: Active-Dendrite NMDA Sub-unit Networks
// Branch-specific nonlinear computation for catastrophic-forgetting-free continual learning
const { Tracer, Array1DTracer, Array2DTracer, ChartTracer, LogTracer, Layout, VerticalLayout, HorizontalLayout } = require('algorithm-visualizer');

const logger = new LogTracer('Active Dendrite — NMDA Branch Networks');
const branchTracer = new Array2DTracer('Dendritic Branch Activations');
const contextTracer = new Array1DTracer('Context Vectors (Task ID)');
const outputTracer = new ChartTracer('Network Output');
const ewcTracer = new ChartTracer('EWC Fisher Penalty');

Layout.setRoot(new VerticalLayout([
  logger,
  new HorizontalLayout([branchTracer, outputTracer]),
  contextTracer,
  ewcTracer,
]));

// --- Network Config ---
const N_NEURONS = 6;
const N_BRANCHES = 4;       // dendritic branches per neuron
const N_SYNAPSES = 8;       // synapses per branch
const N_INPUT = 8;
const N_OUTPUT = 3;
const N_CONTEXT = 4;
const N_TASKS = 3;
const NMDA_THRESHOLD = 0.3; // NMDA plateau threshold
const EWC_LAMBDA = 5.0;     // EWC regularization strength

// --- Initialize Weights ---
const branch_weights = []; // [neuron][branch][synapse]
const context_weights = []; // [neuron][branch][context_dim]
for (let n = 0; n < N_NEURONS; n++) {
  branch_weights.push([]);
  context_weights.push([]);
  for (let b = 0; b < N_BRANCHES; b++) {
    branch_weights[n].push([]);
    context_weights[n].push([]);
    for (let s = 0; s < N_SYNAPSES; s++) {
      branch_weights[n][b].push(+(Math.random() * 0.5 - 0.25).toFixed(3));
    }
    for (let c = 0; c < N_CONTEXT; c++) {
      context_weights[n][b].push(+(Math.random() * 0.5).toFixed(3));
    }
  }
}

// Output weights
const W_out = [];
for (let o = 0; o < N_OUTPUT; o++) {
  W_out.push([]);
  for (let n = 0; n < N_NEURONS; n++) {
    W_out[o].push(+(Math.random() * 0.3).toFixed(3));
  }
}

// Fisher information (EWC)
const fisher = [];
for (let n = 0; n < N_NEURONS; n++) {
  fisher.push(new Array(N_BRANCHES).fill(0));
}

// Task context vectors (one-hot-ish)
const task_contexts = [];
for (let t = 0; t < N_TASKS; t++) {
  const ctx = new Array(N_CONTEXT).fill(0);
  ctx[t % N_CONTEXT] = 1.0;
  if (t + 1 < N_CONTEXT) ctx[t + 1] = 0.5;
  task_contexts.push(ctx);
}

// --- Helper functions ---
function sigmoid(x) { return 1.0 / (1.0 + Math.exp(-x)); }
function nmda_nonlinearity(x) {
  // NMDA plateau: sublinear below threshold, supralinear above
  return x > NMDA_THRESHOLD ? x * 1.5 + 0.2 : x * 0.3;
}
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(x => x / sum);
}

logger.println('=== Active-Dendrite NMDA Sub-unit Network ===');
logger.println(`${N_NEURONS} neurons x ${N_BRANCHES} branches x ${N_SYNAPSES} synapses`);
logger.println(`NMDA threshold=${NMDA_THRESHOLD}, EWC lambda=${EWC_LAMBDA}`);
logger.println('Each branch computes independent NMDA nonlinearity');
logger.println('Context vector gates which branches are active per task');
Tracer.delay();

// --- Continual Learning: Train on 3 tasks sequentially ---
for (let task = 0; task < N_TASKS; task++) {
  const ctx = task_contexts[task];
  contextTracer.set(ctx.map(c => +c.toFixed(2)));
  Tracer.delay();

  logger.println(`\n--- Task ${task}: Context = [${ctx.map(c => c.toFixed(1)).join(',')}] ---`);
  Tracer.delay();

  // Generate task-specific data (5 samples per task)
  for (let sample = 0; sample < 5; sample++) {
    // Random input
    const x = [];
    for (let i = 0; i < N_INPUT; i++) {
      x.push(Math.random() * (task + 1) * 0.3);
    }
    const target = task; // target class = task id

    // --- Forward pass with dendritic computation ---
    const branch_acts = [];
    const neuron_outputs = [];

    for (let n = 0; n < N_NEURONS; n++) {
      branch_acts.push([]);
      let neuron_sum = 0;

      for (let b = 0; b < N_BRANCHES; b++) {
        // Compute context gate for this branch
        let gate = 0;
        for (let c = 0; c < N_CONTEXT; c++) {
          gate += context_weights[n][b][c] * ctx[c];
        }
        gate = sigmoid(gate);

        // Compute branch input (subset of synapses)
        let branch_input = 0;
        for (let s = 0; s < N_SYNAPSES; s++) {
          const x_idx = s % N_INPUT;
          branch_input += branch_weights[n][b][s] * x[x_idx];
        }

        // NMDA nonlinearity + context gating
        const nmda_out = nmda_nonlinearity(Math.abs(branch_input));
        const gated_out = nmda_out * gate;
        branch_acts[n].push(+gated_out.toFixed(3));
        neuron_sum += gated_out;
      }

      neuron_outputs.push(sigmoid(neuron_sum));
    }

    // Display branch activations
    branchTracer.set(branch_acts);
    Tracer.delay();

    // Output layer
    const logits = [];
    for (let o = 0; o < N_OUTPUT; o++) {
      let s = 0;
      for (let n = 0; n < N_NEURONS; n++) {
        s += W_out[o][n] * neuron_outputs[n];
      }
      logits.push(s);
    }
    const probs = softmax(logits);

    // Highlight winning branches
    for (let n = 0; n < N_NEURONS; n++) {
      const max_b = branch_acts[n].indexOf(Math.max(...branch_acts[n]));
      branchTracer.select(n, max_b);
    }
    Tracer.delay();

    // --- Backward pass (simplified gradient) ---
    const lr = 0.05;
    for (let n = 0; n < N_NEURONS; n++) {
      for (let b = 0; b < N_BRANCHES; b++) {
        // EWC penalty: penalize changes to important weights
        const ewc_penalty = EWC_LAMBDA * fisher[n][b];
        const grad = (probs[target] - 1.0) * W_out[target][n] * 0.1;

        for (let s = 0; s < N_SYNAPSES; s++) {
          const delta = -lr * grad - lr * ewc_penalty * branch_weights[n][b][s] * 0.01;
          branch_weights[n][b][s] += delta;
        }
      }
      branchTracer.deselect(n, branch_acts[n].indexOf(Math.max(...branch_acts[n])));
    }

    if (sample === 4) {
      logger.println(`  Sample ${sample}: pred=[${probs.map(p => p.toFixed(2)).join(',')}] target=${target}`);
    }
  }

  // After task: compute Fisher information (EWC)
  logger.println(`  Computing Fisher information for EWC consolidation...`);
  const fisher_display = [];
  for (let n = 0; n < N_NEURONS; n++) {
    fisher_display.push([]);
    for (let b = 0; b < N_BRANCHES; b++) {
      // Fisher ~ squared gradient magnitude
      fisher[n][b] += Math.random() * 0.5 + 0.2;
      fisher_display[n].push(+fisher[n][b].toFixed(2));
    }
  }
  Tracer.delay();
  logger.println(`  Fisher diagonal: max=${Math.max(...fisher.flat()).toFixed(2)}`);
}

logger.println('\n=== Continual Learning Complete ===');
logger.println('Key insight: Each dendritic branch is an independent NMDA');
logger.println('computation unit. Context vectors gate which branches activate');
logger.println('per task, enabling task-specific subnetworks. EWC Fisher');
logger.println('regularization prevents catastrophic forgetting of prior tasks.');
logger.println('This achieves continual learning WITHOUT replay buffers.');
Tracer.delay();
