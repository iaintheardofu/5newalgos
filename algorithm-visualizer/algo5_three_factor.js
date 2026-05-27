// Algorithm #5: Three-Factor Neuromodulated Plasticity + Sleep-Replay + Neurogenesis
// DA/ACh/NE eligibility traces, Hebbian sleep consolidation, neural turnover
const { Tracer, Array1DTracer, Array2DTracer, ChartTracer, LogTracer, Layout, VerticalLayout, HorizontalLayout } = require('algorithm-visualizer');

const logger = new LogTracer('Three-Factor Plasticity + Sleep-Replay');
const weightTracer = new Array2DTracer('Synaptic Weight Matrix');
const eligTracer = new Array2DTracer('Eligibility Traces');
const neuromodTracer = new Array1DTracer('Neuromodulators [DA, ACh, NE]');
const rewardTracer = new ChartTracer('Reward Prediction Error');
const activityTracer = new Array1DTracer('Network Activity');

Layout.setRoot(new VerticalLayout([
  logger,
  new HorizontalLayout([weightTracer, eligTracer]),
  new HorizontalLayout([neuromodTracer, rewardTracer]),
  activityTracer,
]));

// --- Network Config ---
const N_IN = 8;
const N_HIDDEN = 6;
const N_OUT = 3;
const N_TASKS = 3;
const LR = 0.05;
const ELIG_DECAY = 0.9;     // eligibility trace decay
const SLEEP_REPLAY_N = 5;   // replays per sleep phase
const NEUROGENESIS_RATE = 0.1;

// --- Neuromodulator State ---
const neuromod = {
  DA: 0.5,   // Dopamine: reward/reinforcement
  ACh: 0.5,  // Acetylcholine: attention/precision
  NE: 0.3,   // Norepinephrine: arousal/exploration
};

function sigmoid(x) { return 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, x)))); }
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(x => x / sum);
}

// --- Initialize Weights ---
const W_ih = []; // input -> hidden
const W_ho = []; // hidden -> output
const E_ih = []; // eligibility traces (input -> hidden)
const E_ho = []; // eligibility traces (hidden -> output)

for (let h = 0; h < N_HIDDEN; h++) {
  W_ih.push([]);
  E_ih.push([]);
  for (let i = 0; i < N_IN; i++) {
    W_ih[h].push(+(Math.random() * 0.4 - 0.2).toFixed(3));
    E_ih[h].push(0);
  }
}
for (let o = 0; o < N_OUT; o++) {
  W_ho.push([]);
  E_ho.push([]);
  for (let h = 0; h < N_HIDDEN; h++) {
    W_ho[o].push(+(Math.random() * 0.4 - 0.2).toFixed(3));
    E_ho[o].push(0);
  }
}

// Memory buffer for sleep replay
const replay_buffer = [];
// Reward prediction error state
let V = 0; // value estimate
const rpe_history = [];

weightTracer.set(W_ih.map(row => row.map(w => +w.toFixed(2))));
eligTracer.set(E_ih.map(row => row.map(e => +e.toFixed(2))));
neuromodTracer.set([neuromod.DA, neuromod.ACh, neuromod.NE].map(v => +v.toFixed(2)));
Tracer.delay();

logger.println('=== Three-Factor Neuromodulated Plasticity ===');
logger.println(`Network: ${N_IN}->${N_HIDDEN}->${N_OUT}`);
logger.println('dW = pre * post * M(DA,ACh,NE) — three-factor rule');
logger.println('Eligibility traces bridge temporal credit assignment');
Tracer.delay();

// --- Train on sequential tasks ---
for (let task = 0; task < N_TASKS; task++) {
  logger.println(`\n--- WAKE Phase: Task ${task} ---`);
  Tracer.delay();

  // Task-specific neuromodulator profile
  neuromod.DA = 0.3 + task * 0.2;
  neuromod.ACh = 0.7 - task * 0.1;
  neuromod.NE = task === 0 ? 0.6 : 0.3; // high exploration on first task
  neuromodTracer.set([neuromod.DA, neuromod.ACh, neuromod.NE].map(v => +v.toFixed(2)));
  neuromodTracer.select(0); // highlight DA
  Tracer.delay();
  neuromodTracer.deselect(0);

  for (let trial = 0; trial < 8; trial++) {
    // Generate task-specific input
    const x = [];
    for (let i = 0; i < N_IN; i++) {
      x.push(Math.random() * 0.5 + (i % (task + 2) === 0 ? 0.5 : 0));
    }
    const target = task;

    // --- Forward pass ---
    const h = [];
    for (let j = 0; j < N_HIDDEN; j++) {
      let sum = 0;
      for (let i = 0; i < N_IN; i++) sum += W_ih[j][i] * x[i];
      h.push(sigmoid(sum));
    }

    const logits = [];
    for (let o = 0; o < N_OUT; o++) {
      let sum = 0;
      for (let j = 0; j < N_HIDDEN; j++) sum += W_ho[o][j] * h[j];
      logits.push(sum);
    }
    const probs = softmax(logits);
    const pred = probs.indexOf(Math.max(...probs));

    // Reward signal
    const reward = pred === target ? 1.0 : -0.5;

    // --- Reward Prediction Error (TD-like) ---
    const rpe = reward - V;
    V += 0.1 * rpe;
    rpe_history.push(+rpe.toFixed(3));

    // Update neuromodulators based on RPE
    neuromod.DA = Math.max(0, Math.min(1, neuromod.DA + 0.1 * rpe));
    neuromod.NE = Math.max(0, Math.min(1, neuromod.NE - 0.05 * Math.abs(rpe)));

    // --- Three-Factor Learning Rule ---
    // dW = eligibility_trace * neuromodulator_signal
    const M = neuromod.DA * 0.5 + neuromod.ACh * 0.3 + neuromod.NE * 0.2;

    // Update eligibility traces (pre * post Hebbian term)
    for (let j = 0; j < N_HIDDEN; j++) {
      for (let i = 0; i < N_IN; i++) {
        // Eligibility: Hebbian coincidence detection
        E_ih[j][i] = ELIG_DECAY * E_ih[j][i] + x[i] * h[j];
        // Three-factor update: eligibility * modulation * RPE
        W_ih[j][i] += LR * E_ih[j][i] * M * rpe;
        W_ih[j][i] = Math.max(-1, Math.min(1, W_ih[j][i]));
      }
    }

    for (let o = 0; o < N_OUT; o++) {
      for (let j = 0; j < N_HIDDEN; j++) {
        E_ho[o][j] = ELIG_DECAY * E_ho[o][j] + h[j] * (o === target ? 1 : 0);
        W_ho[o][j] += LR * E_ho[o][j] * M * rpe;
        W_ho[o][j] = Math.max(-1, Math.min(1, W_ho[o][j]));
      }
    }

    // Store in replay buffer
    replay_buffer.push({ x: [...x], h: [...h], target, reward });

    // Visualize
    if (trial % 2 === 0) {
      weightTracer.set(W_ih.map(row => row.map(w => +w.toFixed(2))));
      eligTracer.set(E_ih.map(row => row.map(e => +e.toFixed(2))));
      activityTracer.set(h.map(v => +v.toFixed(3)));
      neuromodTracer.set([neuromod.DA, neuromod.ACh, neuromod.NE].map(v => +v.toFixed(2)));
      Tracer.delay();

      logger.println(`  trial=${trial}: pred=${pred} target=${target} reward=${reward.toFixed(1)} RPE=${rpe.toFixed(2)} DA=${neuromod.DA.toFixed(2)}`);
    }
  }

  // --- SLEEP Phase: Replay Consolidation ---
  logger.println(`\n--- SLEEP Phase: Replay for Task ${task} ---`);
  Tracer.delay();

  // During sleep: replay stored experiences with reduced neuromodulation
  const sleep_DA = neuromod.DA * 0.3;  // reduced DA during sleep
  const sleep_ACh = 0.1;               // very low ACh (REM-like)
  const sleep_NE = 0.05;               // minimal NE

  neuromodTracer.set([sleep_DA, sleep_ACh, sleep_NE].map(v => +v.toFixed(2)));
  Tracer.delay();

  for (let rep = 0; rep < SLEEP_REPLAY_N; rep++) {
    // Randomly sample from replay buffer
    const idx = Math.floor(Math.random() * replay_buffer.length);
    const memory = replay_buffer[idx];

    // Hebbian consolidation (no RPE, just coactivation)
    for (let j = 0; j < N_HIDDEN; j++) {
      for (let i = 0; i < N_IN; i++) {
        // Pure Hebbian: strengthen coactive connections
        const hebbian = memory.x[i] * memory.h[j] * 0.01;
        W_ih[j][i] += hebbian;
        W_ih[j][i] = Math.max(-1, Math.min(1, W_ih[j][i]));
      }
    }
  }
  logger.println(`  Replayed ${SLEEP_REPLAY_N} memories (Hebbian consolidation)`);

  // --- Neurogenesis: Replace least-useful neurons ---
  if (task > 0) {
    // Find neuron with lowest average activation
    const avg_act = [];
    for (let j = 0; j < N_HIDDEN; j++) {
      let sum = 0;
      for (const mem of replay_buffer.slice(-8)) sum += mem.h[j];
      avg_act.push(sum / 8);
    }
    const min_idx = avg_act.indexOf(Math.min(...avg_act));

    if (Math.random() < NEUROGENESIS_RATE) {
      // Reset weights of least active neuron (neural turnover)
      for (let i = 0; i < N_IN; i++) {
        W_ih[min_idx][i] = +(Math.random() * 0.4 - 0.2).toFixed(3);
        E_ih[min_idx][i] = 0;
      }
      logger.println(`  Neurogenesis: replaced neuron ${min_idx} (avg_act=${avg_act[min_idx].toFixed(3)})`);

      activityTracer.set(avg_act.map(v => +v.toFixed(3)));
      activityTracer.select(min_idx);
      Tracer.delay();
      activityTracer.deselect(min_idx);
    }
  }

  weightTracer.set(W_ih.map(row => row.map(w => +w.toFixed(2))));
  Tracer.delay();
}

logger.println('\n=== Three-Factor Plasticity Complete ===');
logger.println('Key insight: The three-factor rule dW = pre * post * M');
logger.println('bridges Hebbian learning with reinforcement signals.');
logger.println('Eligibility traces solve temporal credit assignment.');
logger.println('Sleep replay consolidates via pure Hebbian coactivation.');
logger.println('Neurogenesis provides regularization through neural turnover,');
logger.println('replacing underperforming neurons with fresh capacity.');
Tracer.delay();
