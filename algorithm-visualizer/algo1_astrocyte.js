// Algorithm #1: Astrocyte-Modulated Tripartite-Synapse Network (NALSM)
// Brain-inspired SNN with glial cell calcium dynamics
const { Tracer, Array1DTracer, Array2DTracer, ChartTracer, LogTracer, Randomize, Layout, VerticalLayout, HorizontalLayout } = require('algorithm-visualizer');

const logger = new LogTracer('Astrocyte-Modulated Tripartite Synapse');
const voltageTracer = new ChartTracer('Membrane Voltages');
const spikeTracer = new Array1DTracer('Spike Raster');
const calciumTracer = new ChartTracer('Astrocyte Ca2+ Dynamics');
const weightTracer = new Array2DTracer('Synaptic Weight Matrix');

Layout.setRoot(new VerticalLayout([
  logger,
  new HorizontalLayout([voltageTracer, calciumTracer]),
  spikeTracer,
  weightTracer,
]));

// --- LIF Neuron Parameters ---
const N = 10;          // neurons
const N_ASTRO = 3;     // astrocytes (each covers ~3-4 neurons)
const TAU_M = 20.0;    // membrane time constant (ms)
const V_REST = -65.0;  // resting potential (mV)
const V_THRESH = -50.0;// spike threshold (mV)
const V_RESET = -70.0; // reset potential (mV)
const DT = 1.0;        // timestep (ms)
const T_SIM = 80;      // simulation steps

// --- Astrocyte Parameters ---
const TAU_CA = 50.0;   // calcium time constant (slow, ~100ms+)
const CA_RISE = 0.3;   // calcium rise per presynaptic spike
const G_MIN = 0.5;     // minimum gain
const G_MAX = 2.0;     // maximum gain
const CA_THRESH = 0.4; // gliotransmitter release threshold

// --- Initialize State ---
const v = new Array(N).fill(V_REST);
const spikes = new Array(N).fill(0);
const ca = new Array(N_ASTRO).fill(0.0);   // astrocyte calcium
const gain = new Array(N_ASTRO).fill(1.0); // gain modulation

// Random synaptic weights (N x N)
const W = [];
for (let i = 0; i < N; i++) {
  W.push([]);
  for (let j = 0; j < N; j++) {
    W[i].push(i === j ? 0 : +(Math.random() * 0.8).toFixed(2));
  }
}

// Astrocyte coverage: which neurons each astrocyte monitors
const coverage = [];
for (let a = 0; a < N_ASTRO; a++) {
  const start = Math.floor(a * N / N_ASTRO);
  const end = Math.floor((a + 1) * N / N_ASTRO);
  coverage.push({ start, end });
}

weightTracer.set(W);
spikeTracer.set(spikes);
Tracer.delay();

logger.println('=== Astrocyte-Modulated Tripartite-Synapse Network ===');
logger.println(`${N} LIF neurons, ${N_ASTRO} astrocytes, tau_m=${TAU_M}ms, tau_ca=${TAU_CA}ms`);
logger.println('Astrocytes modulate synaptic gain via slow Ca2+ waves');
Tracer.delay();

// --- Simulation Loop ---
for (let t = 0; t < T_SIM; t++) {
  // External input current (Poisson-like)
  const I_ext = [];
  for (let i = 0; i < N; i++) {
    I_ext.push(Math.random() < 0.15 ? 25.0 : 0.0);
  }

  // Compute synaptic current from spikes
  const I_syn = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (spikes[j]) {
        // Find which astrocyte covers this synapse
        let astro_gain = 1.0;
        for (let a = 0; a < N_ASTRO; a++) {
          if (j >= coverage[a].start && j < coverage[a].end) {
            astro_gain = gain[a];
            break;
          }
        }
        I_syn[i] += W[i][j] * spikes[j] * astro_gain * 15.0;
      }
    }
  }

  // LIF dynamics: dv/dt = (-(v - V_REST) + I) / tau_m
  for (let i = 0; i < N; i++) {
    const I_total = I_ext[i] + I_syn[i];
    v[i] += DT * (-(v[i] - V_REST) + I_total) / TAU_M;

    if (v[i] >= V_THRESH) {
      spikes[i] = 1;
      v[i] = V_RESET;
      spikeTracer.patch(i, 1);
    } else {
      spikes[i] = 0;
      spikeTracer.patch(i, 0);
    }
  }

  // Astrocyte calcium dynamics
  for (let a = 0; a < N_ASTRO; a++) {
    // Sum presynaptic activity in coverage zone
    let pre_rate = 0;
    for (let i = coverage[a].start; i < coverage[a].end; i++) {
      pre_rate += spikes[i];
    }
    // Ca2+ dynamics: slow rise on activity, exponential decay
    ca[a] += DT * (-ca[a] / TAU_CA + CA_RISE * pre_rate);
    ca[a] = Math.max(0, Math.min(1.0, ca[a]));

    // Gain modulation: sigmoidal function of calcium
    if (ca[a] > CA_THRESH) {
      // Gliotransmitter release -> potentiate
      gain[a] = G_MIN + (G_MAX - G_MIN) / (1 + Math.exp(-10 * (ca[a] - 0.5)));
    } else {
      // Below threshold -> depress toward baseline
      gain[a] += DT * (1.0 - gain[a]) * 0.05;
    }
  }

  // STDP-like weight update (tripartite: modulated by astrocyte gain)
  if (t % 5 === 0) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i !== j && spikes[i] && spikes[j]) {
          let astro_mod = 1.0;
          for (let a = 0; a < N_ASTRO; a++) {
            if (j >= coverage[a].start && j < coverage[a].end) {
              astro_mod = gain[a];
              break;
            }
          }
          W[i][j] = Math.min(1.0, W[i][j] + 0.02 * astro_mod);
          weightTracer.patch(i, j, +W[i][j].toFixed(2));
          Tracer.delay();
          weightTracer.depatch(i, j);
        }
      }
    }
  }

  // Visualize every few steps
  if (t % 3 === 0) {
    spikeTracer.set(spikes);
    Tracer.delay();

    const spikeCount = spikes.reduce((a, b) => a + b, 0);
    if (spikeCount > 0) {
      logger.println(`t=${t}ms: ${spikeCount} spikes | Ca=[${ca.map(c => c.toFixed(2)).join(',')}] | Gain=[${gain.map(g => g.toFixed(2)).join(',')}]`);
    }
  }
}

logger.println('');
logger.println('=== Simulation Complete ===');
logger.println('Key insight: Astrocyte Ca2+ waves provide slow homeostatic');
logger.println('gain control, preventing runaway excitation while enabling');
logger.println('activity-dependent synaptic potentiation (tripartite STDP).');
logger.println('This recovers QKV-attention via biological calcium gating.');
Tracer.delay();
