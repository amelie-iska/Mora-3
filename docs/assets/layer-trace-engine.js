// Deterministic 4-channel, 2-head, 2-layer reference-sized numerical trace.
const width = 4, headWidth = 2, epsilon = 1e-5;
const vector = (length, seed, scale = .42) => Array.from({length}, (_, index) =>
  Math.round(100 * scale * Math.sin((index + 1) * (seed * .31 + .9) + seed)) / 100);
const matrix = (rows, columns, seed, scale) => Array.from({length: rows}, (_, i) =>
  vector(columns, seed + i * 3, scale));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const add = (a, b) => a.map((value, i) => value + b[i]);
const sum = (...rows) => rows.reduce(add);
const linear = (values, weights, bias = weights.map(() => 0)) => values.map(row =>
  weights.map((weightsRow, i) => dot(weightsRow, row) + bias[i]));
const maxPlus = (values, weights) => values.map(row => weights.map(weightsRow =>
  Math.max(...weightsRow.map((weight, i) => weight + row[i]))));
const zero = Array(width).fill(0);

function normParameters(seed) {
  return {gamma: add(Array(width).fill(1), vector(width, seed, .12)), beta: vector(width, seed + 1, .08)};
}

function normalize(rows, {gamma, beta}) {
  const stats = rows.map(row => {
    const mean = row.reduce((a, b) => a + b, 0) / width;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / width;
    return [mean, variance];
  });
  return {stats, values: rows.map((row, i) => row.map((x, j) =>
    (x - stats[i][0]) / Math.sqrt(stats[i][1] + epsilon) * gamma[j] + beta[j]))};
}

export function cdf(x) {
  // Bounded toy activations permit an accurate erf series for the exact GELU.
  const z = x / Math.SQRT2;
  let term = z;
  let result = z;
  for (let n = 1; n <= 32; n += 1) {
    term *= -z * z / n;
    result += term / (2 * n + 1);
  }
  return (1 + 2 / Math.sqrt(Math.PI) * result) / 2;
}

export const weights = {
  feature: matrix(width, 2, 1),
  incidence: matrix(width, 6, 3),
  role: matrix(2, width, 5, .19),
  graph: vector(width, 8, .16),
  layers: [0, 1].map(i => {
    const seed = 13 + i * 40;
    return {
      norm1: normParameters(seed), norm2: normParameters(seed + 2),
      query: matrix(headWidth, headWidth, seed + 4),
      key: matrix(headWidth, headWidth, seed + 6),
      value: matrix(headWidth, headWidth, seed + 8),
      output: matrix(width, width, seed + 10),
      dense: {
        qkv: matrix(3 * width, width, seed + 19),
        qkvBias: vector(3 * width, seed + 20, .09),
        output: matrix(width, width, seed + 23),
        outputBias: vector(width, seed + 24, .08),
      },
      ffnIn: matrix(6, width, seed + 12), ffnInBias: vector(6, seed + 14, .11),
      ffnOut: matrix(width, 6, seed + 16), ffnOutBias: vector(width, seed + 18, .09),
    };
  }),
  finalNorm: normParameters(100),
  affinity: {
    baseWidth: 2, outputMult: 1,
    weights: matrix(3, width, 110),
    bias: vector(3, 120, .12),
  },
};

const presets = {
  junction: {nodes: ["cell A", "cell B"], edges: [[0, 1], [1, 0]], nodeFeatures: [[.8, -.2], [-.1, .7]]},
  chain: {nodes: ["cell A", "cell B", "cell C"], edges: [[0, 1], [1, 2]], nodeFeatures: [[.7, .2], [.1, -.7], [-.3, .6]]},
  contact: {nodes: ["residue A", "residue B", "residue C"], edges: [[0, 1], [0, 2], [2, 1]], nodeFeatures: [[.9, .2], [.1, .8], [.4, -.3]]},
};

export function graphPreset(name, edgeStrength) {
  const {nodes, edges, nodeFeatures} = presets[name];
  const edgeFeatures = edges.map((_, i) => [i === 0 ? edgeStrength : .15 * (i + 1), .25 - i * .2]);
  return {nodes, edges, nodeFeatures, edgeFeatures};
}

export function graphFromInput({nodes, edges, nodeFeatures, edgeFeatures}) {
  const validFeature = row => Array.isArray(row) && row.length === 2
    && row.every(value => Number.isFinite(value) && Math.abs(value) <= 4);
  if (!Array.isArray(nodes) || nodes.length < 1 || nodes.length > 3
      || nodes.some(name => typeof name !== "string" || !name.trim() || name.length > 24 || /^\[(GRAPH|PAD)\]$/.test(name))
      || new Set(nodes).size !== nodes.length) {
    throw new Error("Use 1–3 distinct node names of 1–24 characters.");
  }
  if (!Array.isArray(edges) || edges.length > 5 || edges.some(edge => !Array.isArray(edge)
      || edge.length !== 2 || edge.some(index => !Number.isInteger(index) || index < 0 || index >= nodes.length))) {
    throw new Error("Use 0–5 edges with two valid zero-based node indices each.");
  }
  if (!Array.isArray(nodeFeatures) || nodeFeatures.length !== nodes.length || !nodeFeatures.every(validFeature)
      || !Array.isArray(edgeFeatures) || edgeFeatures.length !== edges.length || !edgeFeatures.every(validFeature)) {
    throw new Error("Supply two finite features between −4 and 4 for each node and edge.");
  }
  const identifiers = nodes.map((_, i) => Array.from({length: 3}, (__, j) => +(i === j)));
  const features = [...nodeFeatures, ...edgeFeatures];
  const endpoints = [...nodes.map((_, i) => [i, i]), ...edges];
  const roles = [...nodes.map(() => 0), ...edges.map(() => 1)];
  const tokens = ["[GRAPH]", ...nodes, ...edges.map(([a, b]) => nodes[a] + " → " + nodes[b]), "[PAD]"];
  return {tokens, features, endpoints, roles, identifiers, valid: tokens.map((_, i) => i < tokens.length - 1)};
}

export function makeGraph(name, edgeStrength) {
  return graphFromInput(graphPreset(name, edgeStrength));
}

function tropicalAttention(rows, valid, parameters, blockSize) {
  const logValues = rows.map(row => row.map(x => Math.log1p(Math.max(x, 0))));
  const heads = [0, 1].map(head => {
    const start = head * headWidth;
    const z = logValues.map(row => row.slice(start, start + headWidth));
    const [q, k, v] = [parameters.query, parameters.key, parameters.value].map(w => maxPlus(z, w));
    const differences = q.map(query => k.map(key => query.map((x, i) => x - key[i])));
    const scores = differences.map(row => row.map((diff, i) =>
      valid[i] ? Math.min(...diff) - Math.max(...diff) : -Infinity));
    const candidates = scores.map(row => row.map((score, i) => v[i].map(x => score + x)));
    const maxima = rows => Array.from({length: headWidth}, (_, channel) =>
      Math.max(...rows.map(values => values[channel])));
    const blocks = candidates.map(row => Array.from({length: Math.ceil(row.length / blockSize)}, (_, i) =>
      maxima(row.slice(i * blockSize, (i + 1) * blockSize))));
    const context = blocks.map(maxima);
    return {z, q, k, v, differences, scores, candidates, blocks, context};
  });
  const concatenated = rows.map((_, i) => heads.flatMap(head => head.context[i]));
  const decoded = concatenated.map(row => row.map(Math.expm1));
  const output = linear(decoded, parameters.output);
  return {logValues, heads, concatenated, decoded, output};
}

const zeros = (rows, cols) => Array.from({length: rows}, () => Array(cols).fill(0));
const extremum = (values, reduce) => values.indexOf(reduce(...values));

const transposeMultiply = (gradient, weights) => weights[0].map((_, i) =>
  gradient.reduce((total, value, j) => total + value * weights[j][i], 0));

function normGradient(input, normalized, parameters, gradient) {
  return input.map((row, token) => {
    const [mean, variance] = normalized.stats[token];
    const std = Math.sqrt(variance + epsilon);
    const standardized = row.map(value => (value - mean) / std);
    const gained = gradient[token].map((value, i) => value * parameters.gamma[i]);
    const average = gained.reduce((sum, value) => sum + value, 0) / width;
    const projection = dot(gained, standardized) / width;
    return gained.map((value, i) => (value - average - standardized[i] * projection) / std);
  });
}

function tropicalAttentionGradient(layer, parameters, gradient) {
  const attention = layer.attention;
  const heads = attention.heads.map((head, h) => {
    const count = head.z.length;
    const [dq, dk, dv, dz] = Array.from({length: 4}, () => zeros(count, headWidth));
    const routes = head.context.map((row, query) => row.map((context, output) => {
      const global = h * headWidth + output;
      const decode = Math.exp(context);
      const seed = gradient[query][global] * decode;
      // torch.max/min select the first index. Blocks retain the first winner on ties.
      const key = head.candidates[query].findIndex(row => row[output] === context);
      const difference = head.differences[query][key];
      const lo = extremum(difference, Math.min), hi = extremum(difference, Math.max);
      dq[query][lo] += seed;
      dq[query][hi] -= seed;
      dk[key][lo] -= seed;
      dk[key][hi] += seed;
      dv[key][output] += seed;
      return {key, lo, hi, seed, decode};
    }));
    for (const [derivative, projection] of [[dq, parameters.query], [dk, parameters.key], [dv, parameters.value]]) {
      derivative.forEach((row, token) => row.forEach((value, output) => {
        const terms = projection[output].map((weight, i) => weight + head.z[token][i]);
        dz[token][extremum(terms, Math.max)] += value;
      }));
    }
    return {routes, dq, dk, dv, dz};
  });
  const normalized = layer.firstNorm.values.map((row, token) => row.map((value, i) =>
    value > 0 ? heads[Math.floor(i / headWidth)].dz[token][i % headWidth] / (1 + value) : 0));
  return {heads, normalized};
}

function denseAttentionGradient(layer, parameters, gradient) {
  const {heads, scale} = layer.attention;
  const count = gradient.length;
  const packed = zeros(count, 3 * width);
  heads.forEach((head, h) => {
    const start = h * headWidth;
    head.context.forEach((context, query) => {
      const incoming = gradient[query].slice(start, start + headWidth);
      // Softmax JVP: p * (dL/dp − E_p[dL/dp]).
      const average = dot(incoming, context);
      head.probabilities[query].forEach((probability, key) => {
        const scoreGradient = probability * (dot(incoming, head.v[key]) - average);
        incoming.forEach((g, channel) => {
          const i = start + channel;
          packed[query][i] += scoreGradient * scale * head.k[key][channel];
          packed[key][width + i] += scoreGradient * head.scaled[query][channel];
          packed[key][2 * width + i] += probability * g;
        });
      });
    });
  });
  return {normalized: packed.map(row => transposeMultiply(row, parameters.dense.qkv))};
}

function attentionResidualGradient(layer, parameters, upstream, kind) {
  const projection = kind === "dense" ? parameters.dense.output : parameters.output;
  const joined = upstream.map(row => transposeMultiply(row, projection));
  const {heads, normalized} = kind === "dense"
    ? denseAttentionGradient(layer, parameters, joined)
    : tropicalAttentionGradient(layer, parameters, joined);
  const input = normGradient(layer.input, layer.firstNorm, parameters.norm1, normalized);
  return {heads, normalized, residual: input.map((row, i) => add(row, upstream[i]))};
}

export function tropicalResidualGradient(layer, parameters, query, channel) {
  const upstream = zeros(layer.input.length, width);
  upstream[query][channel] = 1;
  const result = attentionResidualGradient(layer, parameters, upstream, "tropical");
  return {...result, heads: result.heads.map((head, h) => ({...head,
    routes: head.routes[query].map((route, c) => ({...route, weight: parameters.output[channel][h * headWidth + c]})),
  }))};
}

function denseAttention(rows, valid, {qkv, qkvBias, output, outputBias}, attnMult) {
  const packed = linear(rows, qkv, qkvBias);
  const scale = Math.sqrt(attnMult) / headWidth;
  const heads = [0, 1].map(head => {
    const start = head * headWidth;
    const [q, k, v] = [0, 1, 2].map(part => packed.map(row =>
      row.slice(part * width + start, part * width + start + headWidth)));
    const scaled = q.map(row => row.map(value => value * scale));
    const logits = scaled.map(query => k.map(key => dot(query, key)));
    const scores = logits.map(row => row.map((value, i) => valid[i] ? value : -Infinity));
    const maxima = scores.map(row => Math.max(...row));
    const exponentials = scores.map((row, i) => row.map(value => Math.exp(value - maxima[i])));
    const totals = exponentials.map(row => row.reduce((total, value) => total + value, 0));
    const probabilities = exponentials.map((row, i) => row.map(value => value / totals[i]));
    const candidates = probabilities.map(row => row.map((value, j) => v[j].map(channel => value * channel)));
    const context = candidates.map(row => sum(...row));
    return {q, k, v, scaled, logits, scores, maxima, exponentials, totals, probabilities, candidates, context};
  });
  const concatenated = rows.map((_, i) => heads.flatMap(head => head.context[i]));
  return {packed, scale, heads, concatenated, output: linear(concatenated, output, outputBias)};
}

// Axes are separate even when two predictions use the same encoder state.
export const feedbackAxes = [
  {label: "ΔG · kcal/mol", target: -.65, scale: 1.5, group: "experimental"},
  {label: "UMA · eV", target: -.25, scale: .5, group: "teacher"},
  {label: "eSEN · eV", target: -.4, scale: .5, group: "teacher"},
];
const masks = {both: [true, true, true], experimental: [true, false, false],
  teacher: [false, true, true], none: [false, false, false]};
export const feedbackWeight = .25;

export function affinityObjective(predictions, coverage) {
  const rows = feedbackAxes.map((axis, i) => {
    const active = masks[coverage][i];
    const residual = active ? (predictions[i] - axis.target) / axis.scale : null;
    return {...axis, active, prediction: predictions[i], residual, loss: active ? residual ** 2 : null};
  });
  const reduce = group => {
    const terms = rows.filter(row => row.active && row.group === group);
    return terms.length ? terms.reduce((total, row) => total + row.loss, 0) / terms.length : null;
  };
  const experimental = reduce("experimental"), computed = reduce("teacher");
  const total = experimental === null && computed === null ? null
    : (experimental ?? 0) + feedbackWeight * (computed ?? 0);
  return {rows, experimental, computed, total};
}

export function runLayer(rows, parameters, valid, blockSize, kind, attnMult) {
  const firstNorm = normalize(rows, parameters.norm1);
  const attention = kind === "dense"
    ? denseAttention(firstNorm.values, valid, parameters.dense, attnMult)
    : tropicalAttention(firstNorm.values, valid, parameters, blockSize);
  const residual = rows.map((row, i) => add(row, attention.output[i]));
  const secondNorm = normalize(residual, parameters.norm2);
  const ffnPre = linear(secondNorm.values, parameters.ffnIn, parameters.ffnInBias);
  const ffnActive = ffnPre.map(row => row.map(x => x * cdf(x)));
  const ffnOutput = linear(ffnActive, parameters.ffnOut, parameters.ffnOutBias);
  const output = residual.map((row, i) => add(row, ffnOutput[i]));
  return {input: rows, firstNorm, attention, residual, secondNorm, ffnPre, ffnActive, ffnOutput, output};
}

export function runTrace(graph, blockSize = graph.tokens.length, {kind = "tropical", attnMult = 1} = {}) {
  const incidence = graph.endpoints.map(pair => pair.flatMap(node => graph.identifiers[node]));
  const features = linear(graph.features, weights.feature);
  const incidences = linear(incidence, weights.incidence);
  const roles = graph.roles.map(role => weights.role[role]);
  const embedded = features.map((feature, i) => sum(feature, incidences[i], roles[i]));
  const input = [weights.graph, ...embedded, zero];
  let rows = input;
  const layers = weights.layers.map(parameters => {
    const layer = runLayer(rows, parameters, graph.valid, blockSize, kind, attnMult);
    rows = layer.output;
    return layer;
  });
  const final = normalize(rows, weights.finalNorm);
  const masked = final.values.map((row, i) => graph.valid[i] ? row : zero);
  const head = weights.affinity;
  const contextScale = head.outputMult / (width / head.baseWidth);
  const scaledContext = masked[0].map(value => value * contextScale);
  const [predictions] = linear([scaledContext], head.weights, head.bias);
  return {incidence, features, incidences, roles, input, layers, final, masked,
    affinity: {contextScale, scaledContext, predictions}};
}

// Differentiate the shared-context synthetic objective through both encoder layers.
export function affinityLossGradient(trace, coverage, kind) {
  const objective = affinityObjective(trace.affinity.predictions, coverage);
  if (objective.total === null) return null;
  const output = objective.rows.map(row => {
    if (!row.active) return 0;
    const count = objective.rows.filter(other => other.active && other.group === row.group).length;
    const multiplier = row.group === "teacher" ? feedbackWeight : 1;
    return 2 * multiplier * row.residual / (row.scale * count);
  });
  const context = transposeMultiply(output, weights.affinity.weights)
    .map(value => value * trace.affinity.contextScale);
  const finalOutput = zeros(trace.masked.length, width);
  finalOutput[0] = context;
  const finalInput = normGradient(trace.layers.at(-1).output, trace.final, weights.finalNorm, finalOutput);
  let upstream = finalInput;
  const layers = [];
  for (let i = trace.layers.length - 1; i >= 0; i -= 1) {
    const layer = trace.layers[i];
    const parameters = weights.layers[i];
    const hidden = upstream.map((row, token) => transposeMultiply(row, parameters.ffnOut).map((g, j) => {
      const x = layer.ffnPre[token][j];
      return g * (cdf(x) + x * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI));
    }));
    const ffnInput = hidden.map(row => transposeMultiply(row, parameters.ffnIn));
    const ffnNorm = normGradient(layer.residual, layer.secondNorm, parameters.norm2, ffnInput);
    const postAttention = ffnNorm.map((row, j) => add(row, upstream[j]));
    const attention = attentionResidualGradient(layer, parameters, postAttention, kind);
    layers.push({index: i, output: upstream, hidden, postAttention,
      normalized: attention.normalized, input: attention.residual});
    upstream = attention.residual;
  }
  const feature = upstream.slice(1, -1).map(row => transposeMultiply(row, weights.feature));
  return {output, context, finalInput, layers, embedding: upstream, feature};
}
