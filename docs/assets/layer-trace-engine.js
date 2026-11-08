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

function gelu(x) {
  // Bounded toy activations permit an accurate erf series for the exact GELU.
  const z = x / Math.SQRT2;
  let term = z;
  let result = z;
  for (let n = 1; n <= 32; n += 1) {
    term *= -z * z / n;
    result += term / (2 * n + 1);
  }
  return x * (1 + 2 / Math.sqrt(Math.PI) * result) / 2;
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
      ffnIn: matrix(6, width, seed + 12), ffnInBias: vector(6, seed + 14, .11),
      ffnOut: matrix(width, 6, seed + 16), ffnOutBias: vector(width, seed + 18, .09),
    };
  }),
  finalNorm: normParameters(100),
  readout: vector(width, 110),
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

function runAttention(rows, valid, parameters, blockSize) {
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

export function runTrace(graph, blockSize = graph.tokens.length) {
  const incidence = graph.endpoints.map(pair => pair.flatMap(node => graph.identifiers[node]));
  const features = linear(graph.features, weights.feature);
  const incidences = linear(incidence, weights.incidence);
  const roles = graph.roles.map(role => weights.role[role]);
  const embedded = features.map((feature, i) => sum(feature, incidences[i], roles[i]));
  const input = [weights.graph, ...embedded, zero];
  let rows = input;
  const layers = weights.layers.map(parameters => {
    const firstNorm = normalize(rows, parameters.norm1);
    const attention = runAttention(firstNorm.values, graph.valid, parameters, blockSize);
    const residual = rows.map((row, i) => add(row, attention.output[i]));
    const secondNorm = normalize(residual, parameters.norm2);
    const ffnPre = linear(secondNorm.values, parameters.ffnIn, parameters.ffnInBias);
    const ffnActive = ffnPre.map(row => row.map(gelu));
    const ffnOutput = linear(ffnActive, parameters.ffnOut, parameters.ffnOutBias);
    const output = residual.map((row, i) => add(row, ffnOutput[i]));
    const layer = {input: rows, firstNorm, attention, residual, secondNorm, ffnPre, ffnActive, ffnOutput, output};
    rows = output;
    return layer;
  });
  const final = normalize(rows, weights.finalNorm);
  const masked = final.values.map((row, i) => graph.valid[i] ? row : zero);
  const readout = dot(masked[0], weights.readout);
  return {incidence, features, incidences, roles, input, layers, final, masked, readout};
}
