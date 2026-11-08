// Synthetic examples. Nothing in this module loads a checkpoint or a dataset.
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const norm = vector => Math.hypot(...vector);
const average = vectors => vectors[0].map((_, i) =>
  vectors.reduce((total, vector) => total + vector[i], 0) / vectors.length);

const assay = "junction conductance";
const relation = "donor → receiver";
const stage = "early epithelial";
const protocol = "junction-opening pulse";
const matched = {relation, stage, assay, unit: "nS", protocol};
const references = [
  {...matched, id: "R1", before: [.5, .7, .2, .3], after: [1.8, 1.2, .9, .4], group: "reference culture 1"},
  {...matched, id: "R2", before: [.8, 1.1, -.2, .3], after: [1.7, 1.75, .5, .55], group: "reference culture 2"},
  {...matched, id: "R3", before: [.6, .9, .1, -.1], after: [1.2, 1.55, -.2, .1],
    stage: "late epithelial", group: "reference culture 3"},
];
const queryBefore = [1.15, 2.12, .4, -.25];
const candidates = [
  {...matched, id: "A", state: [2.27, 2.73, 1.13, -.08], relation: "receiver → donor"},
  {...matched, id: "B", state: [2.26, 2.69, -.04, .42]},
  {...matched, id: "C", state: [2.39, 2.82, 1.12, -.07]},
  {...matched, id: "D", state: [2.25, 2.71, 1.11, -.08], assay: "membrane voltage", unit: "mV"},
  {...matched, id: "E", state: [1.27, 2.18, .43, -.24]},
  {...matched, id: "F", state: [2.17, 2.59, 1.04, -.05], stage: "late epithelial"},
];

function rejection(item, query) {
  const reasons = [];
  if (item.relation !== query.relation) reasons.push("Role order");
  if (item.stage !== query.stage) reasons.push("Cell state");
  if (item.assay !== query.assay || item.unit !== query.unit) reasons.push("Endpoint or unit");
  if (item.protocol !== query.protocol) reasons.push("Intervention");
  return reasons.join(", ");
}

export function analogy(scale, referenceMode, scenario = 0) {
  const querySpec = {...matched};
  if (scenario === 1) querySpec.stage = "late epithelial";
  if (scenario === 2) querySpec.relation = "receiver → donor";
  // The explicit choice is made before the compatibility check. R3 is shown
  // for audit and can be chosen separately for the later-stage query.
  const chosen = referenceMode === 0 ? ["R1", "R2"] : [references[referenceMode - 1].id];
  const referenceAudit = references.map(ref => ({...ref,
    disposition: rejection(ref, querySpec) || (chosen.includes(ref.id) ? "Used" : "Outside chosen set"),
  }));
  const pairs = referenceAudit.filter(ref => ref.disposition === "Used");
  const displacement = pairs.length ? average(pairs.map(pair => subtract(pair.after, pair.before))) : null;
  const query = displacement ? queryBefore.map((value, i) => value + scale * displacement[i]) : queryBefore;
  // These are visible proposal descriptors. Query outcomes are absent from
  // every candidate record and never participate in the distance or gates.
  const ranked = candidates.map(candidate => {
    const difference = subtract(candidate.state, query);
    return {...candidate, projected: norm(difference.slice(0, 2)), full: norm(difference),
      refusal: rejection(candidate, querySpec), masked: true};
  }).sort((a, b) => a.full - b.full || a.id.localeCompare(b.id));
  const eligible = ranked.filter(item => !item.refusal);
  const selected = displacement ? eligible[0] : null;
  const projected = ranked.reduce((best, item) => item.projected < best.projected ? item : best);
  const baseline = [...eligible].sort((a,b) =>
    norm(subtract(a.state, queryBefore)) - norm(subtract(b.state, queryBefore)) || a.id.localeCompare(b.id))[0];
  const projectedEligible = [...eligible].sort((a,b) => a.projected-b.projected || a.id.localeCompare(b.id))[0];
  return {pairs, referenceAudit, querySpec, queryBefore, displacement, query, ranked, selected, projected,
    projectedEligible, baseline};
}

export function segmentBytes(payload, learned) {
  const tokens = [...learned, ...new Set(payload)].sort((a,b) => b.length-a.length);
  const costs = Array(payload.length).fill(Infinity).concat(0);
  const next = [];
  for (let i = payload.length - 1; i >= 0; i--) {
    for (const token of tokens) {
      if (!payload.startsWith(token,i)) continue;
      const cost = 1 + costs[i+token.length];
      if (cost < costs[i] || (cost === costs[i] && token.length > next[i].length)) {
        costs[i]=cost;
        next[i]=token;
      }
    }
  }
  const path=[], greedy=[];
  for (let i=0;i<payload.length;i+=next[i].length) path.push(next[i]);
  for (let i=0;i<payload.length;) {
    const token=tokens.find(token=>payload.startsWith(token,i));
    greedy.push(token);
    i+=token.length;
  }
  return {path,costs,greedy};
}

export function hilbertScore(query, key) {
  const differences=subtract(query,key);
  return Math.min(...differences)-Math.max(...differences);
}

export function ripsAt(states, radius) {
  if (states.length!==4) throw new RangeError("The worked filtration uses four vertices");
  const roots=states.map((_,i)=>i);
  const root=i=>roots[i]===i ? i : root(roots[i]);
  const edges=[], faces=[];
  const edge=(i,j)=>norm(subtract(states[i],states[j]))<=radius;
  for (let i=0;i<states.length;i++) {
    for (let j=i+1;j<states.length;j++) {
      if (!edge(i,j)) continue;
      edges.push([i,j]);
      roots[root(j)]=root(i);
      for (let k=j+1;k<states.length;k++) {
        if (edge(i,k) && edge(j,k)) faces.push([i,j,k]);
      }
    }
  }
  const components=new Set(roots.map((_,i)=>root(i))).size;
  // This demo has four vertices. Only the four-face boundary of a tetrahedron
  // produces a dependency among the triangular boundaries.
  const boundaryRank=faces.length===4 ? 3 : faces.length;
  return {edges,faces,components,loops:edges.length-states.length+components-boundaryRank};
}

export function ripsLoopIntervals(states) {
  const critical=[...new Set(states.flatMap((a,i)=>states.slice(i+1).map(b=>norm(subtract(a,b)))))].sort((a,b)=>a-b);
  const intervals=[];
  let birth=null;
  for (const radius of critical) {
    const {loops}=ripsAt(states,radius);
    if (loops && birth===null) birth=radius;
    if (!loops && birth!==null) {
      intervals.push([birth,radius]);
      birth=null;
    }
  }
  return intervals;
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

export function currentVelocity(coupling, frame = 0) {
  const original = [[0, 0, .1], [1.36, .55, -.18], [2.47, -.28, .28], [3.75, .57, -.1]];
  // Proper rotation about the chain axis, plus a frame translation.
  const rotate = ([x,y,z]) => frame ? [x,-z,y] : [x,y,z];
  const xyz = original.map(point => rotate(point).map((value,i)=>value+(frame ? [.1,-.35,.15][i] : 0)));
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
