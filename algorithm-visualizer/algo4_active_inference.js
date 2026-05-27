// Algorithm #4: Hierarchical Active Inference / Predictive-Coding Agent
// Friston's free-energy principle — epistemic + pragmatic value
const { Tracer, Array1DTracer, Array2DTracer, ChartTracer, LogTracer, Layout, VerticalLayout, HorizontalLayout } = require('algorithm-visualizer');

const logger = new LogTracer('Hierarchical Active Inference Agent');
const beliefTracer = new Array1DTracer('Belief State (posterior)');
const errorTracer = new ChartTracer('Prediction Error');
const feTracer = new ChartTracer('Free Energy');
const actionTracer = new Array1DTracer('Action Selection');
const obsTracer = new Array2DTracer('Observation Likelihood (A matrix)');

Layout.setRoot(new VerticalLayout([
  logger,
  new HorizontalLayout([beliefTracer, actionTracer]),
  new HorizontalLayout([feTracer, errorTracer]),
  obsTracer,
]));

// --- Helper functions ---
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(x => x / sum);
}
function log_safe(x) { return Math.log(Math.max(x, 1e-16)); }
function entropy(p) { return -p.reduce((s, pi) => s + (pi > 1e-10 ? pi * log_safe(pi) : 0), 0); }
function kl_div(p, q) {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-10) kl += p[i] * (log_safe(p[i]) - log_safe(q[i]));
  }
  return Math.max(0, kl);
}
function dot(A, x) {
  // Matrix-vector multiply: A[i][j] * x[j]
  return A.map(row => row.reduce((s, a, j) => s + a * x[j], 0));
}

// --- Model Config ---
const N_OBS = 5;      // observation dimensions
const N_STATES = 4;   // hidden state dimensions
const N_ACTIONS = 3;  // available actions
const T_SIM = 25;     // simulation steps
const PRECISION = 2.0;// precision weighting

// --- Generative Model ---
// A matrix: observation likelihood P(o|s)
const A = [];
for (let o = 0; o < N_OBS; o++) {
  A.push([]);
  for (let s = 0; s < N_STATES; s++) {
    A[o].push(o === s ? 0.7 : (o === s + 1 ? 0.2 : 0.1 / (N_OBS - 2)));
  }
  // Normalize columns
  const col_sum = A[o].reduce((a, b) => a + b);
  for (let s = 0; s < N_STATES; s++) A[o][s] /= col_sum;
}

// B matrices: transition model P(s'|s, a)
const B = [];
for (let a = 0; a < N_ACTIONS; a++) {
  const Ba = [];
  for (let i = 0; i < N_STATES; i++) {
    Ba.push([]);
    for (let j = 0; j < N_STATES; j++) {
      if (a === 0) { // action 0: stay
        Ba[i].push(i === j ? 0.8 : 0.2 / (N_STATES - 1));
      } else if (a === 1) { // action 1: shift right
        Ba[i].push(i === (j + 1) % N_STATES ? 0.7 : 0.3 / (N_STATES - 1));
      } else { // action 2: shift left
        Ba[i].push(i === (j - 1 + N_STATES) % N_STATES ? 0.7 : 0.3 / (N_STATES - 1));
      }
    }
  }
  B.push(Ba);
}

// Preferred observations (prior preferences / goals)
const C = new Array(N_OBS).fill(0);
C[0] = 3.0;  // strongly prefer observation 0
C[1] = 1.0;  // mildly prefer observation 1
// Others: neutral/aversive

// State beliefs (posterior)
let mu = new Array(N_STATES).fill(1.0 / N_STATES); // uniform prior
const fe_history = [];

obsTracer.set(A.map(row => row.map(v => +v.toFixed(2))));
beliefTracer.set(mu.map(m => +m.toFixed(3)));
Tracer.delay();

logger.println('=== Hierarchical Active Inference Agent ===');
logger.println(`${N_STATES} hidden states, ${N_OBS} observations, ${N_ACTIONS} actions`);
logger.println('Minimizes variational free energy = complexity - accuracy');
logger.println('Balances epistemic (exploration) vs pragmatic (exploitation)');
Tracer.delay();

// --- Simulation: Agent interacts with environment ---
let true_state = 2; // actual hidden state

for (let t = 0; t < T_SIM; t++) {
  // Generate observation from true state
  const obs_probs = A.map(row => row[true_state]);
  // Sample observation
  let obs_idx = 0;
  let cumsum = 0;
  const r = Math.random();
  for (let o = 0; o < N_OBS; o++) {
    cumsum += obs_probs[o];
    if (r < cumsum) { obs_idx = o; break; }
  }
  const obs = new Array(N_OBS).fill(0);
  obs[obs_idx] = 1.0;

  // --- Belief Update (Predictive Coding) ---
  // Prediction: what we expect to see
  const predicted_obs = dot(A, mu);

  // Prediction error
  const epsilon = obs.map((o, i) => o - predicted_obs[i]);
  const pe_magnitude = Math.sqrt(epsilon.reduce((s, e) => s + e * e, 0));

  // Update beliefs: gradient descent on free energy
  // dmu/dt = A^T * precision * epsilon
  const AT = [];
  for (let s = 0; s < N_STATES; s++) {
    AT.push(A.map(row => row[s]));
  }
  const belief_update = AT.map(row =>
    PRECISION * row.reduce((s, a, i) => s + a * epsilon[i], 0)
  );

  for (let s = 0; s < N_STATES; s++) {
    mu[s] += 0.3 * belief_update[s];
  }
  // Normalize to valid distribution
  mu = softmax(mu.map(m => m * 5));

  // --- Free Energy Computation ---
  // F = complexity - accuracy
  const prior = new Array(N_STATES).fill(1.0 / N_STATES);
  const complexity = kl_div(mu, prior);
  const accuracy = mu.reduce((s, m, i) => {
    return s + m * A.map(row => row[i]).reduce((ss, a, o) => ss + obs[o] * log_safe(a), 0);
  }, 0);
  const free_energy = complexity - accuracy;
  fe_history.push(+free_energy.toFixed(3));

  // --- Action Selection (Expected Free Energy) ---
  const G = []; // expected free energy per action
  for (let a = 0; a < N_ACTIONS; a++) {
    // Predict next state under action a
    const next_mu = dot(B[a], mu);
    const next_mu_norm = softmax(next_mu.map(m => m * 5));

    // Predicted observation
    const pred_obs = dot(A, next_mu_norm);

    // Epistemic value: expected information gain
    const epistemic = entropy(next_mu_norm);

    // Pragmatic value: expected utility (alignment with preferences)
    const pragmatic = pred_obs.reduce((s, p, o) => s + p * C[o], 0);

    // Expected free energy = epistemic_cost - pragmatic_value
    G.push(epistemic - 0.5 * pragmatic);
  }

  // Select action (softmax policy)
  const action_probs = softmax(G.map(g => -PRECISION * g));
  let action = 0;
  let max_prob = 0;
  for (let a = 0; a < N_ACTIONS; a++) {
    if (action_probs[a] > max_prob) { max_prob = action_probs[a]; action = a; }
  }

  // Execute action: transition true state
  const trans_probs = B[action].map(row => row[true_state]);
  cumsum = 0;
  const r2 = Math.random();
  for (let s = 0; s < N_STATES; s++) {
    cumsum += trans_probs[s];
    if (r2 < cumsum) { true_state = s; break; }
  }

  // --- Visualization ---
  beliefTracer.set(mu.map(m => +m.toFixed(3)));
  actionTracer.set(action_probs.map(p => +p.toFixed(3)));

  beliefTracer.select(mu.indexOf(Math.max(...mu)));
  actionTracer.select(action);
  Tracer.delay();
  beliefTracer.deselect(mu.indexOf(Math.max(...mu)));
  actionTracer.deselect(action);

  if (t % 3 === 0) {
    const action_names = ['stay', 'right', 'left'];
    logger.println(`t=${t}: obs=${obs_idx} state=${true_state} belief=[${mu.map(m=>m.toFixed(2)).join(',')}] action=${action_names[action]} FE=${free_energy.toFixed(2)} PE=${pe_magnitude.toFixed(2)}`);
  }
}

logger.println('\n=== Active Inference Complete ===');
logger.println(`Final free energy: ${fe_history[fe_history.length-1]}`);
logger.println('Key insight: The agent minimizes variational free energy,');
logger.println('which naturally balances exploration (epistemic value =');
logger.println('reducing uncertainty) with exploitation (pragmatic value =');
logger.println('achieving preferred outcomes). No separate reward function');
logger.println('needed — preferences are encoded as prior beliefs.');
Tracer.delay();
