// Algorithm #3: Oligodendrocyte/Myelin-Plastic Polychronous SNNs
// DCLS delays + OMP myelination + polychronous group detection
const { Tracer, Array1DTracer, Array2DTracer, ChartTracer, LogTracer, Layout, VerticalLayout, HorizontalLayout } = require('algorithm-visualizer');

const logger = new LogTracer('Myelin-Plastic Polychronous SNN');
const delayTracer = new Array2DTracer('Conduction Delay Matrix (ms)');
const spikeTracer = new Array1DTracer('Spike Raster');
const myelinTracer = new ChartTracer('Myelination Level');
const groupTracer = new Array2DTracer('Polychronous Groups');

Layout.setRoot(new VerticalLayout([
  logger,
  new HorizontalLayout([delayTracer, groupTracer]),
  spikeTracer,
  myelinTracer,
]));

// --- Network Config ---
const N = 12;           // neurons
const D_MAX = 10;       // max conduction delay (ms)
const DT = 1.0;
const T_SIM = 60;
const TAU_M = 20.0;
const V_REST = -65.0;
const V_THRESH = -50.0;
const V_RESET = -70.0;

// --- DCLS: Dilated Convolutions with Learnable Spacings ---
// Each synapse has a learnable delay (continuous, discretized for sim)
const delays = [];       // [post][pre] delay in ms
const weights = [];      // [post][pre] synaptic weight
for (let i = 0; i < N; i++) {
  delays.push([]);
  weights.push([]);
  for (let j = 0; j < N; j++) {
    delays[i].push(i === j ? 0 : Math.floor(Math.random() * D_MAX) + 1);
    weights[i].push(i === j ? 0 : +(Math.random() * 0.6 + 0.1).toFixed(2));
  }
}

// Myelination level per synapse (0=unmyelinated, 1=fully myelinated)
const myelin = [];
for (let i = 0; i < N; i++) {
  myelin.push(new Array(N).fill(0.3));
}

// Spike buffer for delay lines
const spike_history = [];
for (let t = 0; t < D_MAX + 1; t++) {
  spike_history.push(new Array(N).fill(0));
}

// State
const v = new Array(N).fill(V_REST);
const spikes = new Array(N).fill(0);
const spike_times = new Array(N).fill(-100);

delayTracer.set(delays);
spikeTracer.set(spikes);
Tracer.delay();

logger.println('=== Oligodendrocyte/Myelin-Plastic Polychronous SNN ===');
logger.println(`${N} neurons, D_max=${D_MAX}ms, DCLS learnable delays`);
logger.println('Oligodendrocytes adjust myelination to synchronize groups');
Tracer.delay();

// Detected polychronous groups
const poly_groups = [];

// --- Simulation ---
for (let t = 0; t < T_SIM; t++) {
  // Shift spike history
  spike_history.pop();
  spike_history.unshift(new Array(N).fill(0));

  // External input
  const I_ext = [];
  for (let i = 0; i < N; i++) {
    I_ext.push(Math.random() < 0.12 ? 30.0 : 0.0);
  }

  // Compute synaptic current with delays
  const I_syn = new Array(N).fill(0);
  for (let post = 0; post < N; post++) {
    for (let pre = 0; pre < N; pre++) {
      if (post === pre) continue;
      const d = delays[post][pre];
      if (d > 0 && d < spike_history.length && spike_history[d][pre]) {
        // Myelination speeds up conduction (reduces effective delay jitter)
        const myelin_factor = 1.0 + myelin[post][pre] * 0.5;
        I_syn[post] += weights[post][pre] * myelin_factor * 12.0;
      }
    }
  }

  // LIF dynamics
  for (let i = 0; i < N; i++) {
    v[i] += DT * (-(v[i] - V_REST) + I_ext[i] + I_syn[i]) / TAU_M;
    if (v[i] >= V_THRESH) {
      spikes[i] = 1;
      spike_history[0][i] = 1;
      spike_times[i] = t;
      v[i] = V_RESET;
    } else {
      spikes[i] = 0;
    }
  }

  // Oligodendrocyte-Mediated Plasticity (OMP)
  // Myelinate synapses with correlated pre/post activity
  if (t % 5 === 0) {
    for (let post = 0; post < N; post++) {
      for (let pre = 0; pre < N; pre++) {
        if (post === pre) continue;
        const dt_spike = Math.abs(spike_times[post] - spike_times[pre]);
        if (dt_spike < delays[post][pre] + 3 && dt_spike > 0) {
          // Correlated activity -> increase myelination
          myelin[post][pre] = Math.min(1.0, myelin[post][pre] + 0.05);
          // Adjust delay toward optimal synchronization
          const target_delay = Math.max(1, dt_spike);
          delays[post][pre] += Math.sign(target_delay - delays[post][pre]);
          delays[post][pre] = Math.max(1, Math.min(D_MAX, delays[post][pre]));

          delayTracer.patch(post, pre, delays[post][pre]);
          Tracer.delay();
          delayTracer.depatch(post, pre);
        } else {
          // Uncorrelated -> slow demyelination
          myelin[post][pre] = Math.max(0, myelin[post][pre] - 0.005);
        }
      }
    }
  }

  // Visualize
  if (t % 4 === 0) {
    spikeTracer.set(spikes);
    const spiking = [];
    for (let i = 0; i < N; i++) if (spikes[i]) spiking.push(i);
    if (spiking.length >= 2) {
      spikeTracer.select(...spiking.slice(0, 3));
      Tracer.delay();
      spikeTracer.deselect(...spiking.slice(0, 3));
    }
    Tracer.delay();

    if (spiking.length > 0) {
      logger.println(`t=${t}ms: spikes=[${spiking.join(',')}] | avg_myelin=${(myelin.flat().reduce((a,b)=>a+b)/myelin.flat().length).toFixed(3)}`);
    }
  }

  // Polychronous group detection (simplified)
  if (t % 15 === 0 && t > 10) {
    // Find neurons that fire in sequence with consistent delays
    const recent_spikers = [];
    for (let i = 0; i < N; i++) {
      if (t - spike_times[i] < 10) recent_spikers.push(i);
    }
    if (recent_spikers.length >= 3) {
      // Sort by spike time to find temporal ordering
      recent_spikers.sort((a, b) => spike_times[a] - spike_times[b]);
      const group = recent_spikers.slice(0, Math.min(4, recent_spikers.length));
      poly_groups.push(group);
      logger.println(`  Polychronous group detected: [${group.join('->')}]`);
    }
  }
}

// Display detected polychronous groups
const group_matrix = [];
for (let g = 0; g < Math.min(5, poly_groups.length); g++) {
  const row = new Array(N).fill(0);
  for (const n of poly_groups[g]) row[n] = g + 1;
  group_matrix.push(row);
}
if (group_matrix.length > 0) {
  groupTracer.set(group_matrix);
  Tracer.delay();
}

delayTracer.set(delays);
Tracer.delay();

logger.println('\n=== Simulation Complete ===');
logger.println(`Detected ${poly_groups.length} polychronous groups`);
logger.println('Key insight: Oligodendrocytes dynamically adjust myelin thickness,');
logger.println('which controls conduction velocity. DCLS provides learnable delay');
logger.println('spacings. Together they enable polychronous groups — neurons that');
logger.println('fire in precisely-timed sequences, the basis for temporal coding.');
Tracer.delay();
