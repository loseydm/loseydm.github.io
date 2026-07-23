/* ========================================================================
   HiPPO interactive demo
   Integrates the HiPPO ODEs (LegS / LegT / LagT) with a bilinear
   discretization and decodes the polynomial memory at every frame.
   Formulas follow Gu et al., "HiPPO: Recurrent Memory with Optimal
   Polynomial Projections" (NeurIPS 2020), in the normalized-basis
   convention; they were validated against direct projection integrals.
   ======================================================================== */
(() => {
'use strict';

/* ---------------- constants ---------------- */

const T = 10;                 // signal duration in seconds
const STEPS = 1000;           // integration steps
const DT = T / STEPS;
const YMAX = 1.6;             // y-axis half-range of the main chart
const LAGT_CUTOFF = 6;        // display support where weight >= e^-6

const CAT = ['#3987e5', '#008300', '#d55181', '#c98500',
             '#199e70', '#d95926', '#9085e9', '#e66767'];
const COL = {
  sig: '#8f8f8f', sigDim: 'rgba(143,143,143,0.28)',
  recon: '#10b981', now: '#34d399', err: '#e66767',
  grid: '#242424', axis: '#3a3a3a', text: '#898781',
  barOther: '#5c5c58',
  measureFill: 'rgba(16,185,129,0.10)',
  posMax: '#3987e5', negMax: '#e34948', mid: [43, 43, 41], // #2b2b29
};

const MEASURE_INFO = {
  legs: {
    desc: 'Scaled Legendre: every moment since t = 0 counts equally. ' +
          'The memory never forgets; it spreads its budget thinner as history grows. No parameters.',
    matrix: 'Lower-triangular LegS operator, applied with 1/t scaling, so updates slow as history grows. Blue > 0, red < 0.',
  },
  legt: {
    desc: 'Translated Legendre: uniform weight over the last θ seconds, ' +
          'with everything older discarded. This is exactly the Legendre Memory Unit.',
    matrix: 'Dense LegT operator (÷ θ). Note the alternating sign pattern above the diagonal. Blue > 0, red < 0.',
  },
  lagt: {
    desc: 'Laguerre: weight decays exponentially into the past at rate r. ' +
          'Sharp recency, smooth fading, constant dynamics.',
    matrix: 'The LagT operator (× r): A is lower-triangular ones. Blue > 0, red < 0.',
  },
};

const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------------- state ---------------- */

const S = {
  measure: 'legs',
  N: 16,
  theta: 3.0,          // LegT window length
  rate: 1.0,           // LagT decay rate
  signalName: 'waves',
  f: new Float64Array(STEPS + 1),
  drawn: null,         // user-drawn signal (lazy init)
  states: null,        // Float64Array (STEPS+1) * N
  k: 0,                // current step index; t = k * DT
  playing: !REDUCED_MOTION,
  speed: 1,
  drawMode: false,
  stroking: false,     // pointer currently drawing
  showError: false,
  hoverX: null,        // main-chart hover position (css px) or null
};

// integration workspace, rebuilt on measure/N/param change
const W = {
  Apos: null,          // positive-form matrix (paper convention), N*N
  Bpos: null,          // B_n = sqrt(2n+1) (or ones for LagT)
  M0: null, M1inv: null, // precomputed bilinear operators (LegT/LagT)
  scale: 1,            // 1/theta or rate for the constant-A measures
};

/* ---------------- signal presets ---------------- */

function gauss(t, mu, sig) { const d = (t - mu) / sig; return Math.exp(-0.5 * d * d); }

const PRESETS = {
  waves:  t => 0.85 * Math.sin(1.1 * t + 0.4) + 0.45 * Math.sin(2.6 * t + 2.0) + 0.22 * Math.sin(4.3 * t + 1.0),
  bumps:  t => 1.05 * gauss(t, 1.5, 0.30) - 0.85 * gauss(t, 3.9, 0.35) + 0.95 * gauss(t, 6.3, 0.28) - 0.70 * gauss(t, 8.4, 0.30),
  square: t => 0.9 * Math.sign(Math.sin(1.4 * t) + 1e-12),
  chirp:  t => 0.95 * Math.sin(0.45 * t * t),
  damped: t => 1.3 * Math.exp(-0.25 * t) * Math.sin(3.2 * t),
};

function loadSignal(name) {
  S.signalName = name;
  if (name === 'drawn') {
    if (!S.drawn) {
      S.drawn = new Float64Array(STEPS + 1);
      for (let i = 0; i <= STEPS; i++) S.drawn[i] = PRESETS.waves(i * DT);
    }
    S.f.set(S.drawn);
  } else {
    const fn = PRESETS[name];
    for (let i = 0; i <= STEPS; i++) S.f[i] = fn(i * DT);
  }
}

/* ---------------- HiPPO matrices ---------------- */

function buildMatrices() {
  const N = S.N;
  const A = new Float64Array(N * N);
  const B = new Float64Array(N);
  if (S.measure === 'legs') {
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < N; k++) {
        if (n > k)      A[n * N + k] = Math.sqrt((2 * n + 1) * (2 * k + 1));
        else if (n === k) A[n * N + k] = n + 1;
      }
      B[n] = Math.sqrt(2 * n + 1);
    }
  } else if (S.measure === 'legt') {
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < N; k++) {
        const r = Math.sqrt((2 * n + 1) * (2 * k + 1));
        A[n * N + k] = n >= k ? r : r * ((n - k) % 2 === 0 ? 1 : -1);
      }
      B[n] = Math.sqrt(2 * n + 1);
    }
  } else { // lagt
    for (let n = 0; n < N; n++) {
      for (let k = 0; k <= n; k++) A[n * N + k] = 1;
      B[n] = 1;
    }
  }
  W.Apos = A;
  W.Bpos = B;
  W.scale = S.measure === 'legt' ? 1 / S.theta : S.measure === 'lagt' ? S.rate : 1;

  if (S.measure !== 'legs') {
    // constant dynamics: c' = scale * (-A c + B f). Precompute bilinear operators
    //   M1 = I + (dt/2) scale A   (to invert),  M0 = I - (dt/2) scale A
    const h = (DT / 2) * W.scale;
    const M1 = new Float64Array(N * N);
    const M0 = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        M1[i * N + j] = (i === j ? 1 : 0) + h * A[i * N + j];
        M0[i * N + j] = (i === j ? 1 : 0) - h * A[i * N + j];
      }
    }
    W.M0 = M0;
    W.M1inv = invert(M1, N);
  } else {
    W.M0 = W.M1inv = null;
  }
}

/* Gauss-Jordan inverse with partial pivoting (N <= 64). */
function invert(M, N) {
  const a = Float64Array.from(M);
  const inv = new Float64Array(N * N);
  for (let i = 0; i < N; i++) inv[i * N + i] = 1;
  for (let col = 0; col < N; col++) {
    let p = col, best = Math.abs(a[col * N + col]);
    for (let r = col + 1; r < N; r++) {
      const v = Math.abs(a[r * N + col]);
      if (v > best) { best = v; p = r; }
    }
    if (p !== col) {
      for (let j = 0; j < N; j++) {
        let tmp = a[col * N + j]; a[col * N + j] = a[p * N + j]; a[p * N + j] = tmp;
        tmp = inv[col * N + j]; inv[col * N + j] = inv[p * N + j]; inv[p * N + j] = tmp;
      }
    }
    const piv = a[col * N + col];
    for (let j = 0; j < N; j++) { a[col * N + j] /= piv; inv[col * N + j] /= piv; }
    for (let r = 0; r < N; r++) {
      if (r === col) continue;
      const f = a[r * N + col];
      if (f === 0) continue;
      for (let j = 0; j < N; j++) {
        a[r * N + j] -= f * a[col * N + j];
        inv[r * N + j] -= f * inv[col * N + j];
      }
    }
  }
  return inv;
}

function matvec(M, x, out, N) {
  for (let i = 0; i < N; i++) {
    let s = 0;
    const row = i * N;
    for (let j = 0; j < N; j++) s += M[row + j] * x[j];
    out[i] = s;
  }
}

/* ---------------- trajectory ---------------- */

function computeTrajectory() {
  const N = S.N;
  const f = S.f;
  const A = W.Apos, B = W.Bpos;
  const states = new Float64Array((STEPS + 1) * N);
  const c = new Float64Array(N);
  const rhs = new Float64Array(N);
  const tmp = new Float64Array(N);

  if (S.measure === 'legs') {
    // time-varying: c' = (1/t)(-A c + B f); bilinear with the lower-triangular solve
    //   (I + a1 A) c_new = (I - a0 A) c + (dt/2)(B f0 / t0 + B f1 / t1)
    for (let k = 0; k < STEPS; k++) {
      const t1 = (k + 1) * DT;
      const t0 = k === 0 ? t1 : k * DT;    // first step: treat singular 1/t at t=0
      const a0 = DT / (2 * t0);
      const a1 = DT / (2 * t1);
      // rhs = (I - a0 A) c + input terms
      for (let i = 0; i < N; i++) {
        let s = c[i];
        const row = i * N;
        for (let j = 0; j <= i; j++) s -= a0 * A[row + j] * c[j];  // A lower-triangular
        rhs[i] = s + (DT / 2) * B[i] * (f[k] / t0 + f[k + 1] / t1);
      }
      // forward substitution: (I + a1 A) c_new = rhs
      for (let i = 0; i < N; i++) {
        let s = rhs[i];
        const row = i * N;
        for (let j = 0; j < i; j++) s -= a1 * A[row + j] * c[j];
        c[i] = s / (1 + a1 * A[row + i]);
      }
      states.set(c, (k + 1) * N);
    }
  } else {
    // constant dynamics: c_new = M1inv (M0 c + (dt/2) scale B (f0 + f1))
    const h = (DT / 2) * W.scale;
    for (let k = 0; k < STEPS; k++) {
      matvec(W.M0, c, tmp, N);
      const u = h * (f[k] + f[k + 1]);
      for (let i = 0; i < N; i++) tmp[i] += u * B[i];
      matvec(W.M1inv, tmp, rhs, N);
      c.set(rhs);
      states.set(c, (k + 1) * N);
    }
  }
  S.states = states;
}

function stateAt(k) {
  return S.states.subarray(k * S.N, (k + 1) * S.N);
}

/* ---------------- decoding (reconstruction) ---------------- */

/* Support of the measure at time t, clipped for display: [x0, t]. */
function support(t) {
  if (S.measure === 'legs') return [0, t];
  if (S.measure === 'legt') return [Math.max(0, t - S.theta), t];
  return [Math.max(0, t - LAGT_CUTOFF / S.rate), t];
}

/* Reconstruction f^(x) from state c at time t. x must lie in the support. */
function reconAt(c, t, x) {
  const N = S.N;
  if (S.measure === 'lagt') {
    const u = S.rate * (t - x);              // Laguerre argument
    let l0 = 1, l1 = 1 - u, s = c[0];
    if (N > 1) s += c[1] * l1;
    for (let n = 1; n < N - 1; n++) {
      const l2 = ((2 * n + 1 - u) * l1 - n * l0) / (n + 1);
      s += c[n + 1] * l2;
      l0 = l1; l1 = l2;
    }
    return s;
  }
  const z = S.measure === 'legs' ? 2 * x / t - 1 : 2 * (x - t) / S.theta + 1;
  let p0 = 1, p1 = z, s = c[0];
  if (N > 1) s += c[1] * Math.sqrt(3) * z;
  for (let n = 1; n < N - 1; n++) {
    const p2 = ((2 * n + 1) * z * p1 - n * p0) / (n + 1);
    s += c[n + 1] * Math.sqrt(2 * n + 3) * p2;
    p0 = p1; p1 = p2;
  }
  return s;
}

/* One basis contribution c_n * g_n(t, x). */
function basisTermAt(c, n, t, x) {
  const N = S.N;
  if (S.measure === 'lagt') {
    const u = S.rate * (t - x);
    let l0 = 1, l1 = 1 - u;
    if (n === 0) return c[0];
    if (n === 1) return c[1] * l1;
    for (let m = 1; m < n; m++) {
      const l2 = ((2 * m + 1 - u) * l1 - m * l0) / (m + 1);
      l0 = l1; l1 = l2;
    }
    return c[n] * l1;
  }
  const z = S.measure === 'legs' ? 2 * x / t - 1 : 2 * (x - t) / S.theta + 1;
  let p0 = 1, p1 = z;
  if (n === 0) return c[0];
  if (n === 1) return c[1] * Math.sqrt(3) * z;
  for (let m = 1; m < n; m++) {
    const p2 = ((2 * m + 1) * z * p1 - m * p0) / (m + 1);
    p0 = p1; p1 = p2;
  }
  return c[n] * Math.sqrt(2 * n + 1) * p1;
}

/* Signal value at arbitrary x by linear interpolation of samples. */
function signalAt(x) {
  const u = Math.min(Math.max(x / DT, 0), STEPS);
  const i = Math.floor(u);
  if (i >= STEPS) return S.f[STEPS];
  const fr = u - i;
  return S.f[i] * (1 - fr) + S.f[i + 1] * fr;
}

/* Measure weight (unnormalized shape in [0,1]) at x given time t. */
function weightAt(t, x) {
  if (S.measure === 'lagt') return Math.exp(-S.rate * (t - x));
  return 1;
}

/* Weighted RMS reconstruction error over the support. */
function reconError(c, t) {
  const [x0, x1] = support(t);
  if (x1 - x0 < 5 * DT) return null;
  const M = 240;
  let num = 0, den = 0;
  for (let i = 0; i <= M; i++) {
    const x = x0 + (x1 - x0) * i / M;
    const w = weightAt(t, x);
    const d = reconAt(c, t, x) - signalAt(x);
    num += w * d * d;
    den += w;
  }
  return Math.sqrt(num / den);
}

/* ---------------- DOM ---------------- */

const $ = id => document.getElementById(id);
const mainCanvas = $('mainCanvas'), stateCanvas = $('stateCanvas'),
      basisCanvas = $('basisCanvas'), matrixCanvas = $('matrixCanvas');
const tooltip = $('tooltip');
const demoPanel = document.querySelector('.demo-panel');

let dirty = true;

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, r.width, r.height];
}

/* ---------------- main chart ---------------- */

const MARGIN = { l: 36, r: 12, t: 10, b: 24 };

function mainGeom() {
  const r = mainCanvas.getBoundingClientRect();
  const pw = r.width - MARGIN.l - MARGIN.r;
  const ph = r.height - MARGIN.t - MARGIN.b;
  return {
    xOf: t => MARGIN.l + (t / T) * pw,
    tOf: px => (px - MARGIN.l) / pw * T,
    yOf: v => MARGIN.t + ph / 2 - (v / YMAX) * ph / 2,
    vOf: py => ((MARGIN.t + ph / 2) - py) / (ph / 2) * YMAX,
    pw, ph, W: r.width, H: r.height,
  };
}

function drawMain() {
  const [ctx, cw, ch] = fit(mainCanvas);
  const g = mainGeom();
  ctx.clearRect(0, 0, cw, ch);
  const t = S.k * DT;
  const c = stateAt(S.k);

  // measure band
  const [x0, x1] = support(t);
  if (t > 0) {
    if (S.measure === 'lagt') {
      const grad = ctx.createLinearGradient(g.xOf(x0), 0, g.xOf(x1), 0);
      grad.addColorStop(0, 'rgba(16,185,129,0)');
      grad.addColorStop(0.6, 'rgba(16,185,129,0.045)');
      grad.addColorStop(1, 'rgba(16,185,129,0.14)');
      ctx.fillStyle = grad;
    } else {
      ctx.fillStyle = COL.measureFill;
    }
    ctx.fillRect(g.xOf(x0), MARGIN.t, Math.max(0, g.xOf(x1) - g.xOf(x0)), g.ph);
  }

  // grid + axes
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 1;
  ctx.font = '11px Inter, sans-serif';
  ctx.fillStyle = COL.text;
  for (const v of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(MARGIN.l, g.yOf(v)); ctx.lineTo(cw - MARGIN.r, g.yOf(v));
    ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(v.toString(), MARGIN.l - 6, g.yOf(v));
  }
  for (let tv = 0; tv <= T; tv += 2) {
    const x = g.xOf(tv);
    ctx.beginPath();
    ctx.moveTo(x, MARGIN.t); ctx.lineTo(x, MARGIN.t + g.ph);
    ctx.strokeStyle = COL.grid; ctx.stroke();
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(tv + ' s', x, MARGIN.t + g.ph + 6);
  }
  ctx.strokeStyle = COL.axis;
  ctx.beginPath();
  ctx.moveTo(MARGIN.l, g.yOf(0)); ctx.lineTo(cw - MARGIN.r, g.yOf(0));
  ctx.stroke();
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText('0', MARGIN.l - 6, g.yOf(0));

  // signal: future dim, past solid
  const plotSignal = (from, to, style, width) => {
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    for (let i = from; i <= to; i++) {
      const x = g.xOf(i * DT), y = g.yOf(S.f[i]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  plotSignal(S.k, STEPS, COL.sigDim, 1.4);
  if (S.k > 0) plotSignal(0, S.k, COL.sig, 1.6);

  // reconstruction + error over the support
  if (t > 8 * DT && !S.stroking) {
    const M = Math.max(160, Math.round(g.pw));
    ctx.strokeStyle = COL.recon;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    const err = S.showError ? [] : null;
    for (let i = 0; i <= M; i++) {
      const x = x0 + (x1 - x0) * i / M;
      const v = reconAt(c, t, x);
      const y = g.yOf(Math.min(Math.max(v, -YMAX * 1.05), YMAX * 1.05));
      if (i === 0) ctx.moveTo(g.xOf(x), y); else ctx.lineTo(g.xOf(x), y);
      if (err) err.push([x, Math.abs(v - signalAt(x))]);
    }
    ctx.stroke();

    if (err) {
      ctx.strokeStyle = COL.err;
      ctx.lineWidth = 1.25;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      const bottom = MARGIN.t + g.ph;
      const scale = g.ph / (2 * YMAX);
      for (let i = 0; i < err.length; i++) {
        const [x, e] = err[i];
        const y = bottom - Math.min(e, YMAX) * scale;
        if (i === 0) ctx.moveTo(g.xOf(x), y); else ctx.lineTo(g.xOf(x), y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // now marker
  const xNow = g.xOf(t);
  ctx.strokeStyle = COL.now;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  ctx.moveTo(xNow, MARGIN.t); ctx.lineTo(xNow, MARGIN.t + g.ph);
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = COL.now;
  ctx.beginPath();
  ctx.arc(xNow, g.yOf(S.f[S.k]), 4, 0, 2 * Math.PI);
  ctx.fill();

  // hover crosshair
  if (S.hoverX !== null && !S.drawMode) {
    ctx.strokeStyle = 'rgba(229,229,229,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(S.hoverX, MARGIN.t); ctx.lineTo(S.hoverX, MARGIN.t + g.ph);
    ctx.stroke();
  }
}

/* ---------------- state bars ---------------- */

function drawState() {
  const [ctx, cw, ch] = fit(stateCanvas);
  ctx.clearRect(0, 0, cw, ch);
  const c = stateAt(S.k);
  const N = S.N;
  const m = { l: 10, r: 10, t: 12, b: 20 };
  const pw = cw - m.l - m.r, ph = ch - m.t - m.b;
  const mid = m.t + ph / 2;

  let vmax = 1;
  for (let i = 0; i < N; i++) vmax = Math.max(vmax, Math.abs(c[i]));
  vmax *= 1.12;

  // zero line
  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m.l, mid); ctx.lineTo(cw - m.r, mid); ctx.stroke();

  const slot = pw / N;
  const gap = slot >= 5 ? 2 : 1;
  const bw = Math.max(1, slot - gap);
  for (let i = 0; i < N; i++) {
    const h = (c[i] / vmax) * (ph / 2);
    ctx.fillStyle = i < 8 ? CAT[i] : COL.barOther;
    const x = m.l + i * slot + gap / 2;
    if (h >= 0) ctx.fillRect(x, mid - h, bw, Math.max(h, 0.5));
    else ctx.fillRect(x, mid, bw, Math.max(-h, 0.5));
  }

  ctx.fillStyle = COL.text;
  ctx.font = '10px Inter, sans-serif';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText('n = 0', m.l, ch - 14);
  ctx.textAlign = 'right';
  ctx.fillText('n = ' + (N - 1), cw - m.r, ch - 14);
  ctx.textAlign = 'left';
  ctx.fillText('+' + vmax.toFixed(1), m.l, 2);
}

/* ---------------- basis curves ---------------- */

function drawBasis() {
  const [ctx, cw, ch] = fit(basisCanvas);
  ctx.clearRect(0, 0, cw, ch);
  const t = S.k * DT;
  const c = stateAt(S.k);
  const m = { l: 10, r: 10, t: 18, b: 10 };
  const pw = cw - m.l - m.r, ph = ch - m.t - m.b;
  const mid = m.t + ph / 2;
  const nShow = Math.min(8, S.N);

  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m.l, mid); ctx.lineTo(cw - m.r, mid); ctx.stroke();

  if (t <= 8 * DT || S.stroking) return;

  const [x0, x1] = support(t);
  const M = 160;

  // scale from the max |contribution|
  let vmax = 0.4;
  const vals = [];
  for (let n = 0; n < nShow; n++) {
    const row = new Float64Array(M + 1);
    for (let i = 0; i <= M; i++) {
      const x = x0 + (x1 - x0) * i / M;
      row[i] = basisTermAt(c, n, t, x);
      vmax = Math.max(vmax, Math.abs(row[i]));
    }
    vals.push(row);
  }
  const yOf = v => mid - (v / vmax) * (ph / 2) * 0.92;

  // faint full reconstruction (sum of all N components, not just the 8 shown)
  ctx.strokeStyle = 'rgba(229,229,229,0.30)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i <= M; i++) {
    const xv = x0 + (x1 - x0) * i / M;
    const s = reconAt(c, t, xv);
    const x = m.l + pw * i / M;
    const y = Math.min(Math.max(yOf(s), 1), ch - 1);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  for (let n = 0; n < nShow; n++) {
    ctx.strokeStyle = CAT[n];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    for (let i = 0; i <= M; i++) {
      const x = m.l + pw * i / M;
      const y = yOf(vals[n][i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // index chips
  ctx.font = '600 10px Inter, sans-serif';
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  let lx = m.l;
  for (let n = 0; n < nShow; n++) {
    ctx.fillStyle = CAT[n];
    ctx.fillText(String(n), lx, 3);
    lx += ctx.measureText(String(n)).width + 8;
  }
}

/* ---------------- operator heatmap ---------------- */

function effectiveA(n, k) {
  // signed continuous operator as used in c' = A'c + B'f (LegS shown unscaled)
  const v = -W.Apos[n * S.N + k];
  return S.measure === 'legs' ? v : v * W.scale;
}

function effectiveB(n) {
  return S.measure === 'legs' ? W.Bpos[n] : W.Bpos[n] * W.scale;
}

function divergingColor(v, vmax) {
  const a = Math.pow(Math.min(Math.abs(v) / vmax, 1), 0.55);
  const [mr, mg, mb] = COL.mid;
  const to = v >= 0 ? [57, 135, 229] : [227, 73, 72];
  const r = Math.round(mr + (to[0] - mr) * a);
  const g = Math.round(mg + (to[1] - mg) * a);
  const b = Math.round(mb + (to[2] - mb) * a);
  return `rgb(${r},${g},${b})`;
}

let matrixLayout = null; // for hover hit-testing

function drawMatrix() {
  const [ctx, cw, ch] = fit(matrixCanvas);
  ctx.clearRect(0, 0, cw, ch);
  const N = S.N;
  const m = { l: 10, r: 10, t: 18, b: 16 };
  const gapBA = 10;

  let vmax = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) vmax = Math.max(vmax, Math.abs(effectiveA(i, j)));
  for (let i = 0; i < N; i++) vmax = Math.max(vmax, Math.abs(effectiveB(i)));

  const availW = cw - m.l - m.r, availH = ch - m.t - m.b;
  const cell = Math.min((availW - gapBA) / (N + 1), availH / N);
  const side = cell * N;
  const ox = m.l + (availW - (side + gapBA + cell)) / 2;
  const oy = m.t + (availH - side) / 2;

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      ctx.fillStyle = divergingColor(effectiveA(i, j), vmax);
      ctx.fillRect(ox + j * cell, oy + i * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
  const bx = ox + side + gapBA;
  for (let i = 0; i < N; i++) {
    ctx.fillStyle = divergingColor(effectiveB(i), vmax);
    ctx.fillRect(bx, oy + i * cell, Math.ceil(cell), Math.ceil(cell));
  }

  ctx.fillStyle = COL.text;
  ctx.font = '600 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('A', ox + side / 2, oy - 4);
  ctx.fillText('B', bx + cell / 2, oy - 4);

  matrixLayout = { ox, oy, cell, side, bx, N };
}

/* ---------------- readouts & captions ---------------- */

function fmtParamLabel() {
  if (S.measure === 'legt') {
    return 'Window length — θ = ' + S.theta.toFixed(1) + ' s';
  }
  const halfLife = Math.LN2 / S.rate;
  return 'Decay rate — r = ' + S.rate.toFixed(2) +
         '  (half-life ' + halfLife.toFixed(2) + ' s)';
}

function updateReadouts() {
  const t = S.k * DT;
  $('tVal').textContent = 't = ' + t.toFixed(2) + ' s';
  const e = S.states ? reconError(stateAt(S.k), t) : null;
  $('errVal').textContent = e === null ? 'error —' : 'error ' + e.toFixed(3);
  $('timeRange').value = S.k;
}

function updateCaptions() {
  $('measureDesc').textContent = MEASURE_INFO[S.measure].desc;
  $('matrixCaption').textContent = MEASURE_INFO[S.measure].matrix;
  $('nVal').textContent = 'N = ' + S.N;
  $('stateSub').textContent =
    'These ' + S.N + ' numbers are everything the model remembers.';
  $('basisNote').textContent =
    (S.N > 8 ? 'Showing components 0–7 of ' + S.N + '. ' : '') +
    'Each coefficient scales one polynomial; the sum of all ' + S.N +
    ' (faint white) is the green reconstruction.';
  const pg = $('paramGroup');
  if (S.measure === 'legs') {
    pg.style.display = 'none';
  } else {
    pg.style.display = '';
    $('paramLabel').textContent = fmtParamLabel();
    const pr = $('pRange');
    if (S.measure === 'legt') {
      pr.min = 0.5; pr.max = 6; pr.step = 0.1; pr.value = S.theta;
    } else {
      pr.min = 0.2; pr.max = 3; pr.step = 0.05; pr.value = S.rate;
    }
  }
}

/* ---------------- orchestration ---------------- */

function rebuild() {
  buildMatrices();
  computeTrajectory();
  drawMatrix();
  dirty = true;
}

function drawAll() {
  drawMain();
  drawState();
  drawBasis();
  updateReadouts();
}

function setPlaying(p) {
  S.playing = p && S.k < STEPS || p && S.k === STEPS && (S.k = 0, true);
  $('playBtn').innerHTML = S.playing ? '&#10074;&#10074;' : '&#9654;';
  $('playBtn').setAttribute('aria-label', S.playing ? 'Pause' : 'Play');
}

let lastTime = performance.now();
let acc = 0;

function frame(now) {
  if (S.playing) {
    acc += Math.min((now - lastTime) / 1000, 0.1) * S.speed;
    const adv = Math.floor(acc / DT);
    if (adv > 0) {
      acc -= adv * DT;
      S.k = Math.min(S.k + adv, STEPS);
      if (S.k === STEPS) setPlaying(false);
      dirty = true;
    }
  }
  lastTime = now;
  if (dirty) { drawAll(); dirty = false; }
  requestAnimationFrame(frame);
}

/* ---------------- tooltip ---------------- */

function showTooltip(clientX, clientY, html) {
  const pr = demoPanel.getBoundingClientRect();
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  let x = clientX - pr.left + 14;
  let y = clientY - pr.top + 14;
  const tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
  if (x + tw > pr.width - 8) x = clientX - pr.left - tw - 12;
  if (y + th > pr.height - 8) y = clientY - pr.top - th - 12;
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}

function hideTooltip() { tooltip.style.display = 'none'; }

/* ---------------- interaction: main canvas ---------------- */

function canvasPos(ev) {
  const r = mainCanvas.getBoundingClientRect();
  return [ev.clientX - r.left, ev.clientY - r.top];
}

function applyStrokePoint(px, py) {
  const g = mainGeom();
  const idx = Math.round(Math.min(Math.max(g.tOf(px) / DT, 0), STEPS));
  const val = Math.min(Math.max(g.vOf(py), -1.5), 1.5);
  if (S.lastDrawIdx === null || S.lastDrawIdx === undefined) {
    S.drawn[idx] = val;
  } else {
    const i0 = S.lastDrawIdx, v0 = S.drawn[i0];
    const lo = Math.min(i0, idx), hi = Math.max(i0, idx);
    for (let i = lo; i <= hi; i++) {
      const fr = hi === lo ? 1 : (i - lo) / (hi - lo);
      S.drawn[i] = i0 <= idx ? v0 + (val - v0) * fr : val + (v0 - val) * fr;
    }
  }
  S.lastDrawIdx = idx;
  S.f.set(S.drawn);
}

mainCanvas.addEventListener('pointerdown', ev => {
  ev.preventDefault();
  mainCanvas.setPointerCapture(ev.pointerId);
  const [px, py] = canvasPos(ev);
  if (S.drawMode) {
    if (!S.drawn) loadSignal('drawn');
    $('signalSel').value = 'drawn';
    S.signalName = 'drawn';
    S.stroking = true;
    S.playing = false;
    setPlaying(false);
    S.lastDrawIdx = null;
    applyStrokePoint(px, py);
    dirty = true;
  } else {
    S.scrubbing = true;
    setPlaying(false);
    const g = mainGeom();
    S.k = Math.round(Math.min(Math.max(g.tOf(px) / DT, 0), STEPS));
    dirty = true;
  }
});

mainCanvas.addEventListener('pointermove', ev => {
  const [px, py] = canvasPos(ev);
  if (S.stroking) {
    applyStrokePoint(px, py);
    dirty = true;
    return;
  }
  if (S.scrubbing) {
    const g = mainGeom();
    S.k = Math.round(Math.min(Math.max(g.tOf(px) / DT, 0), STEPS));
    dirty = true;
    return;
  }
  if (S.drawMode) return;
  // hover crosshair + tooltip
  const g = mainGeom();
  const th = g.tOf(px);
  if (th >= 0 && th <= T && py >= MARGIN.t && py <= MARGIN.t + g.ph) {
    S.hoverX = px;
    const t = S.k * DT;
    const [x0, x1] = support(t);
    let html = '<span class="tt-mut">t′ = </span>' + th.toFixed(2) + ' s' +
               '<br><span class="tt-mut">f = </span>' + signalAt(th).toFixed(3);
    if (S.states && t > 8 * DT && th >= x0 && th <= x1) {
      const v = reconAt(stateAt(S.k), t, th);
      html += '<br><span class="tt-mut">f̂ = </span>' + v.toFixed(3);
    }
    showTooltip(ev.clientX, ev.clientY, html);
    dirty = true;
  } else {
    S.hoverX = null;
    hideTooltip();
    dirty = true;
  }
});

mainCanvas.addEventListener('pointerup', ev => {
  if (S.stroking) {
    S.stroking = false;
    S.lastDrawIdx = null;
    rebuild();
    S.k = 0;
    if (!REDUCED_MOTION) setPlaying(true);
  }
  S.scrubbing = false;
});

mainCanvas.addEventListener('pointerleave', () => {
  S.hoverX = null;
  hideTooltip();
  dirty = true;
});

/* ---------------- interaction: state bars ---------------- */

stateCanvas.addEventListener('pointermove', ev => {
  const r = stateCanvas.getBoundingClientRect();
  const px = ev.clientX - r.left;
  const m = { l: 10, r: 10 };
  const pw = r.width - m.l - m.r;
  const i = Math.floor((px - m.l) / (pw / S.N));
  if (i >= 0 && i < S.N && S.states) {
    const v = stateAt(S.k)[i];
    showTooltip(ev.clientX, ev.clientY,
      '<span class="tt-mut">c[' + i + '] = </span>' + v.toFixed(3));
  } else hideTooltip();
});
stateCanvas.addEventListener('pointerleave', hideTooltip);

/* ---------------- interaction: matrix ---------------- */

matrixCanvas.addEventListener('pointermove', ev => {
  if (!matrixLayout) return;
  const r = matrixCanvas.getBoundingClientRect();
  const px = ev.clientX - r.left, py = ev.clientY - r.top;
  const { ox, oy, cell, side, bx, N } = matrixLayout;
  const row = Math.floor((py - oy) / cell);
  if (row < 0 || row >= N) { hideTooltip(); return; }
  if (px >= ox && px < ox + side) {
    const col = Math.floor((px - ox) / cell);
    if (col >= 0 && col < N) {
      showTooltip(ev.clientX, ev.clientY,
        '<span class="tt-mut">A[' + row + ',' + col + '] = </span>' +
        effectiveA(row, col).toFixed(2));
      return;
    }
  }
  if (px >= bx && px < bx + cell) {
    showTooltip(ev.clientX, ev.clientY,
      '<span class="tt-mut">B[' + row + '] = </span>' + effectiveB(row).toFixed(2));
    return;
  }
  hideTooltip();
});
matrixCanvas.addEventListener('pointerleave', hideTooltip);

/* ---------------- controls ---------------- */

$('measureSeg').addEventListener('click', ev => {
  const btn = ev.target.closest('button[data-measure]');
  if (!btn || btn.dataset.measure === S.measure) return;
  S.measure = btn.dataset.measure;
  for (const b of $('measureSeg').querySelectorAll('button'))
    b.classList.toggle('on', b === btn);
  updateCaptions();
  rebuild();
});

$('signalSel').addEventListener('change', ev => {
  loadSignal(ev.target.value);
  rebuild();
  S.k = 0;
  if (!REDUCED_MOTION) setPlaying(true); else { S.k = STEPS; dirty = true; }
});

$('nRange').addEventListener('input', ev => {
  S.N = parseInt(ev.target.value, 10);
  updateCaptions();
  rebuild();
});

$('pRange').addEventListener('input', ev => {
  const v = parseFloat(ev.target.value);
  if (S.measure === 'legt') S.theta = v; else S.rate = v;
  $('paramLabel').textContent = fmtParamLabel();
  rebuild();
});

$('drawBtn').addEventListener('click', () => {
  S.drawMode = !S.drawMode;
  $('drawBtn').setAttribute('aria-pressed', String(S.drawMode));
  $('mainWrap').classList.toggle('drawing', S.drawMode);
  if (S.drawMode) {
    setPlaying(false);
    const wasDrawn = S.signalName === 'drawn';
    loadSignal('drawn');
    $('signalSel').value = 'drawn';
    if (!wasDrawn) rebuild();
    S.hoverX = null;
    hideTooltip();
    dirty = true;
  }
});

$('playBtn').addEventListener('click', () => setPlaying(!S.playing));

$('restartBtn').addEventListener('click', () => {
  S.k = 0;
  acc = 0;
  setPlaying(true);
  dirty = true;
});

$('speedSel').addEventListener('change', ev => {
  S.speed = parseFloat(ev.target.value);
});

$('timeRange').addEventListener('input', ev => {
  S.k = parseInt(ev.target.value, 10);
  S.playing = false;
  setPlaying(false);
  dirty = true;
});

$('errChk').addEventListener('change', ev => {
  S.showError = ev.target.checked;
  dirty = true;
});

/* ---------------- guided experiments ---------------- */

for (const btn of document.querySelectorAll('.exp-load')) {
  btn.addEventListener('click', () => {
    const d = btn.dataset;
    S.measure = d.measure;
    for (const b of $('measureSeg').querySelectorAll('button'))
      b.classList.toggle('on', b.dataset.measure === d.measure);
    S.N = parseInt(d.n, 10);
    $('nRange').value = S.N;
    if (d.p) {
      const v = parseFloat(d.p);
      if (d.measure === 'legt') S.theta = v; else if (d.measure === 'lagt') S.rate = v;
    }
    loadSignal(d.signal);
    $('signalSel').value = d.signal;
    updateCaptions();
    rebuild();

    const wantDraw = d.draw === '1';
    if (wantDraw !== S.drawMode) $('drawBtn').click();
    if (wantDraw || REDUCED_MOTION) {
      S.k = STEPS;
      setPlaying(false);
    } else {
      S.k = 0;
      acc = 0;
      setPlaying(true);
    }
    dirty = true;
    $('demo').scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth', block: 'start' });
  });
}

/* ---------------- resize ---------------- */

const ro = new ResizeObserver(() => { drawMatrix(); dirty = true; });
ro.observe(demoPanel);

/* ---------------- init ---------------- */

loadSignal('waves');
updateCaptions();
rebuild();
if (REDUCED_MOTION) S.k = STEPS;
setPlaying(S.playing);
requestAnimationFrame(frame);

})();
