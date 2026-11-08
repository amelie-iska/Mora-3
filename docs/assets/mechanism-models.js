// Synthetic examples. Nothing in this module loads a checkpoint or a dataset.
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const norm = vector => Math.hypot(...vector);
const average = vectors => vectors[0].map((_, i) =>
  vectors.reduce((total, vector) => total + vector[i], 0) / vectors.length);

const references = [
  {name: "Matched pair 1", before: [.5, .7, .2, .3], after: [1.8, 1.2, .9, .4]},
  {name: "Matched pair 2", before: [.8, 1.1, -.2, .3], after: [1.7, 1.75, .5, .55]},
];
const queryBefore = [1.15, 2.12, .4, -.25];
const assay = "junction conductance";
const candidates = [
  {id: "A", state: [2.27, 2.73, 1.13, -.08], relation: "receiver → donor", context: "matched", assay},
  {id: "B", state: [2.26, 2.69, -.04, .42], relation: "donor → receiver", context: "matched", assay},
  {id: "C", state: [2.39, 2.82, 1.12, -.07], relation: "donor → receiver", context: "matched", assay},
  {id: "D", state: [2.25, 2.71, 1.11, -.08], relation: "donor → receiver", context: "matched", assay: "membrane voltage"},
  {id: "E", state: [1.27, 2.18, .43, -.24], relation: "donor → receiver", context: "matched", assay},
  {id: "F", state: [2.17, 2.59, 1.04, -.05], relation: "donor → receiver", context: "different cell state", assay},
];

function rejection(candidate) {
  if (candidate.relation !== "donor → receiver") return "Reversed assay roles";
  if (candidate.context !== "matched") return "Cell-state mismatch";
  if (candidate.assay !== assay) return "Different endpoint";
  return null;
}

export function analogy(scale, referenceMode) {
  const pairs = referenceMode === 0 ? references : [references[referenceMode - 1]];
  const displacement = average(pairs.map(pair => subtract(pair.after, pair.before)));
  const query = queryBefore.map((value, i) => value + scale * displacement[i]);
  const ranked = candidates.map(candidate => {
    const difference = subtract(candidate.state, query);
    return {...candidate, projected: norm(difference.slice(0, 2)), full: norm(difference), refusal: rejection(candidate)};
  }).sort((a, b) => a.full - b.full);
  const selected = ranked.find(item => !item.refusal);
  const projected = ranked.reduce((best, item) => item.projected < best.projected ? item : best);
  const baseline = ranked.filter(item => !item.refusal).reduce((best, item) =>
    norm(subtract(item.state, queryBefore)) < norm(subtract(best.state, queryBefore)) ? item : best);
  return {pairs, queryBefore, displacement, query, ranked, selected, projected, baseline};
}

const logDensity = (value, mean, std) => value.reduce((total, coordinate, i) =>
  total - ((coordinate - mean[i]) / std[i]) ** 2 / 2 - Math.log(std[i]) - Math.log(2 * Math.PI) / 2, 0);

export function trajectory(logReward, length) {
  const states = [[0, 0], [1.05, .8], [2.05, 1.55], [3.32, 1.92]].slice(0, length + 1);
  const stopProbabilities = [0, .28, .61, 1];
  const terms = states.slice(1).map((state, i) => {
    const prev = states[i];
    const forward = logDensity(state, [prev[0] + .95, prev[1] + .55], [.48, .35]);
    const backward = i === 0 ? 0 : logDensity(prev, [state[0] - .86, state[1] - .58], [.52, .41]);
    const continuation = Math.log1p(-stopProbabilities[i]);
    return {forward, backward, continuation};
  });
  const stop = Math.log(stopProbabilities[length]);
  const logZ = .85;
  const balance = terms.reduce((total, step) => total + step.forward + step.continuation - step.backward,
    logZ + stop - logReward);
  return {states, terms, stop, balance, logZ};
}

export function currentVelocity(coupling) {
  const xyz = [[0, 0, .1], [1.36, .55, -.18], [2.47, -.28, .28], [3.75, .57, -.1]];
  const free = [false, true, true, true];
  const velocities = xyz.map((point, i) => {
    const contributions = xyz.map((neighbor, j) => {
      const relative = subtract(neighbor, point);
      const distance = norm(relative);
      const scalar = .18 * Math.exp(-distance) + (Math.abs(i-j) === 1 ? coupling : 0);
      return relative.map(value => scalar * value);
    });
    return point.map((_, axis) => free[i]
      ? contributions.reduce((total, row) => total + row[axis], 0) / xyz.length : 0);
  });
  const next = xyz.map((point, i) => point.map((value, j) => value + .4 * velocities[i][j]));
  return {xyz, free, velocities, next, distance: norm(subtract(next[1], next[2]))};
}
