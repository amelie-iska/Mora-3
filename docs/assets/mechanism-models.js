// Synthetic examples. Nothing in this module loads a checkpoint or a dataset.
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const norm = vector => Math.hypot(...vector);
const average = vectors => vectors[0].map((_, i) =>
  vectors.reduce((total, vector) => total + vector[i], 0) / vectors.length);

const assay = "junction conductance";
const relation = "donor → receiver";
const stage = "early epithelial";
const protocol = "junction-opening pulse";
export const coolingGaps = [160, 80, 40, 20];
export const coolingTemperatures = coolingGaps.map((_, i) =>
  290 + coolingGaps.slice(i).reduce((total, gap) => total + gap, 0)).concat(290);
const kBoltzmannEv = 8.617333262145e-5;
export const energyExamples = [
  {name: "UMA illustration", energies: [-2, -1.97, -1.94]},
  {name: "eSEN illustration", energies: [-2.2, -2.145, -2.11]},
];
export function boltzmannStates(energies, baseWeights, temperature, reference = Math.min(...energies)) {
  if (!energies.length || energies.length !== baseWeights.length || !energies.every(Number.isFinite)
      || !baseWeights.every(weight => Number.isFinite(weight) && weight > 0)
      || !Number.isFinite(reference) || !Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("Finite energies, positive base weights, and positive temperature required.");
  }
  const kT = kBoltzmannEv * temperature;
  const logReward = energies.map(energy => -(energy - reference) / kT);
  if (!logReward.every(Number.isFinite)) throw new RangeError("Unrepresentable log reward.");
  const baseScale = Math.max(...baseWeights);
  const logBaseNormalizer = Math.log(baseScale) + Math.log(baseWeights.reduce((sum, weight) => sum + weight / baseScale, 0));
  const logMass = logReward.map((reward, i) => reward + Math.log(baseWeights[i]) - logBaseNormalizer);
  const shift = Math.max(...logMass);
  const weights = logMass.map(value => Math.exp(value - shift));
  const sum = weights.reduce((total, weight) => total + weight, 0);
  return {reference, kT, logReward, logMass, logNormalizer: shift + Math.log(sum), probabilities: weights.map(weight => weight / sum)};
}

const matched = {relation, stage, assay, unit: "nS", protocol};
const references = [
  {...matched, id: "R1", before: [.5, .7, .2, .3], after: [1.8, 1.2, .9, .4], group: "reference culture 1"},
  {...matched, id: "R2", before: [.8, 1.1, -.2, .3], after: [1.7, 1.75, .5, .55], group: "reference culture 2"},
  {...matched, id: "R3", before: [.6, .9, .1, -.1], after: [1.2, 1.55, -.2, .1],
    stage: "late epithelial", group: "reference culture 3"},
];
const discordantReference = {...matched, id: "R4", before: [1.4, 1.3, .9, .2], after: [.5, .72, .24, .12],
  group: "reference culture 4"};
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

// Typed roles and edges are explicit inputs to the synthetic match.
export const sourceMotif = {
  types: ["donor", "junction", "receiver"],
  edges: [[0, 1, "opens"], [1, 2, "conducts"]],
};
export function checkMotif(source, target, mapping) {
  if (mapping.length !== source.types.length || target.types.length !== mapping.length
      || new Set(mapping).size !== mapping.length
      || mapping.some(i => !Number.isInteger(i) || i < 0 || i >= target.types.length))
    return {ok:false, reasons:["Correspondence is not a bijection"], matched:0};
  const reasons=[];
  if (source.types.some((type,i)=>type !== target.types[mapping[i]])) reasons.push("Node type mismatch");
  const expected = source.edges.map(([a,b,type])=>JSON.stringify([mapping[a],mapping[b],type]));
  const actual = target.edges.map(edge=>JSON.stringify(edge));
  const matched = expected.filter(edge=>actual.includes(edge)).length;
  if (matched !== expected.length || actual.length !== expected.length || new Set(actual).size !== actual.length)
    reasons.push("Directed typed edge mismatch");
  return {ok:!reasons.length,reasons,matched};
}
export function matchMotifByRole(source, target) {
  const mapping = source.types.map(type => target.types.indexOf(type));
  if (new Set(source.types).size !== source.types.length || new Set(target.types).size !== target.types.length
      || source.types.length !== target.types.length || mapping.includes(-1)) {
    return {mapping:null, ok:false, reasons:["Unique role correspondence unavailable"], matched:0};
  }
  return {mapping, ...checkMotif(source, target, mapping)};
}

function candidateMotif(id) {
  if (id === "C") {
    return {types:["receiver","donor","junction"], edges:[[1,2,"opens"],[2,0,"conducts"]]};
  }
  const edges = id === "B" ? [[0,1,"opens"],[0,2,"conducts"]]
    : id === "A" ? [[1,0,"opens"],[2,1,"conducts"]] : sourceMotif.edges;
  return {types:sourceMotif.types, edges};
}

export function topologyComparison(radius, scenario=0) {
  const source=[[0,0],[1,0],[1,1],[0,1]];
  const target=scenario===1 ? [[0,0],[.4,0],[.8,0],[1.2,0]]
    : scenario===2 ? source.map(p=>[...p]) : [[0,0],[1.2,.1],[1.35,.85],[.14,.95]];
  const mapping=scenario===2 ? [0,2,1,3] : [0,1,2,3];
  const a=ripsAt(source,radius), b=ripsAt(target,radius);
  const key=(i,j)=>[Math.min(i,j),Math.max(i,j)].join(":");
  const mapped=new Set(a.edges.map(([i,j])=>key(mapping[i],mapping[j])));
  const targetEdges=new Set(b.edges.map(([i,j])=>key(i,j)));
  const missing=[...mapped].filter(k=>!targetEdges.has(k));
  const extra=[...targetEdges].filter(k=>!mapped.has(k));
  const distances=source.flatMap((p,i)=>source.slice(i+1).map((q,k)=>{
    const j=i+k+1, left=norm(subtract(p,q)),right=norm(subtract(target[mapping[i]],target[mapping[j]]));
    return {pair:`${i}↔${j}`,left,right,error:Math.abs(left-right)};
  }));
  return {source,target,mapping,a,b,missing,extra,distances,
    distortion:Math.max(...distances.map(d=>d.error)),
    sameBetti:a.components===b.components && a.loops===b.loops,
    preservesEdges:!missing.length && !extra.length};
}

export function analogy(scale, referenceMode, scenario = 0) {
  if (!Number.isFinite(scale) || !Number.isInteger(referenceMode) || referenceMode < 0 || referenceMode > 3
      || ![0,1,2,3].includes(scenario)) throw new RangeError("Invalid analogy controls");
  const querySpec = {...matched};
  if (scenario === 1) querySpec.stage = "late epithelial";
  if (scenario === 2) querySpec.relation = "receiver → donor";
  // The explicit choice is made before the compatibility check. R3 is shown
  // for audit and can be chosen separately for the later-stage query.
  const chosen = referenceMode === 0 ? (scenario === 3 ? ["R1", "R2", "R4"] : ["R1", "R2"])
    : [references[referenceMode - 1].id];
  const available = scenario === 3 ? [...references, discordantReference] : references;
  const referenceAudit = available.map(ref => ({...ref,
    disposition: rejection(ref, querySpec) || (chosen.includes(ref.id) ? "Used" : "Outside chosen set"),
  }));
  const pairs = referenceAudit.filter(ref => ref.disposition === "Used");
  const transfer = active => {
    const deltas = active.map(pair => subtract(pair.after, pair.before));
    const agreement = [];
    for (const [i, first] of deltas.entries()) {
      for (let j=i+1;j<deltas.length;j++) {
        const second = deltas[j];
        const cosine = first.reduce((sum, value, axis) => sum + value * second[axis], 0) / (norm(first) * norm(second));
        agreement.push({references:[active[i].id, active[j].id], cosine});
      }
    }
    const conflicts = agreement.filter(({cosine}) => !Number.isFinite(cosine) || cosine <= 0);
    const transferReason = !active.length ? "no compatible reference pairs"
      : conflicts.length ? "conflicting reference directions" : null;
    const displacement = transferReason ? null : average(deltas);
    const query = displacement ? queryBefore.map((value, i) => value + scale * displacement[i]) : queryBefore;
    return {displacement, query, agreement, conflicts, transferReason};
  };
  const {displacement, query, agreement, conflicts, transferReason} = transfer(pairs);
  // These are visible proposal descriptors. Query outcomes are absent from
  // every candidate record and never participate in the distance or gates.
  const ranked = candidates.map(candidate => {
    const difference = subtract(candidate.state, query);
    const targetMotif = candidateMotif(candidate.id);
    const motif = matchMotifByRole(sourceMotif,targetMotif);
    return {...candidate, motif, targetMotif, difference, squared:difference.map(v=>v*v), projected: norm(difference.slice(0, 2)), full: norm(difference),
      refusal: [rejection(candidate, querySpec), ...motif.reasons].filter(Boolean).join(", "), masked: true};
  }).sort((a, b) => a.full - b.full || a.id.localeCompare(b.id));
  const eligible = ranked.filter(item => !item.refusal);
  const selected = displacement ? eligible[0] : null;
  const projected = ranked.reduce((best, item) => item.projected < best.projected ? item : best);
  const baseline = [...eligible].sort((a,b) =>
    norm(subtract(a.state, queryBefore)) - norm(subtract(b.state, queryBefore)) || a.id.localeCompare(b.id))[0];
  const projectedEligible = [...eligible].sort((a,b) => a.projected-b.projected || a.id.localeCompare(b.id))[0];
  const sensitivity = pairs.map(omitted => {
    const remaining = pairs.filter(pair => pair !== omitted);
    const {displacement, query: alternative} = transfer(remaining);
    const ranks = eligible.map(item => ({id: item.id, distance: norm(subtract(item.state, alternative))}))
      .sort((a,b) => a.distance-b.distance || a.id.localeCompare(b.id));
    const pick = displacement ? ranks[0] : null;
    return {omitted: omitted.id, remaining: remaining.map(item => item.id), selected: pick?.id ?? null,
      distance: pick?.distance ?? null, margin: pick && ranks.length > 1 ? ranks[1].distance - pick.distance : null};
  });
  return {pairs, referenceAudit, querySpec, queryBefore, displacement, query, ranked, selected, projected, agreement, conflicts, transferReason,
    projectedEligible, baseline, sensitivity,
    margin: selected && eligible.length > 1 ? eligible[1].full-selected.full : null};
}

export function segmentBytes(payload, learned) {
  const tokens = [...new Set([...learned, ...payload])].sort((a,b) => b.length-a.length);
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
  const firstChoices = tokens.filter(token => payload.startsWith(token)).map(token => ({
    token, suffixCost:costs[token.length], chosen:token === next[0],
  }));
  return {path,costs,greedy,firstChoices};
}

export function hilbertTerms(query, key) {
  const differences=subtract(query,key);
  const min=Math.min(...differences), max=Math.max(...differences);
  return {differences, min, max, score:min-max};
}
export const hilbertScore = (query, key) => hilbertTerms(query, key).score;

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
    return {forward, backward, continuation, contribution: forward + continuation - backward};
  });
  const stop = Math.log(stopProbabilities[length]);
  const logZ = .85;
  const pathSum = terms.reduce((total, step) => total + step.contribution, 0);
  const balance = logZ + pathSum + stop - logReward;
  return {states, terms, stop, balance, logZ, pathSum};
}

const rotateQuarterTurn = ([x,y,z]) => [x,-z,y];

export function currentVelocity(coupling, frame = 0) {
  const original = [[0, 0, .1], [1.36, .55, -.18], [2.47, -.28, .28], [3.75, .57, -.1]];
  // Proper rotation about the chain axis, plus a frame translation.
  const xyz = frame ? original.map(point => rotateQuarterTurn(point).map((value,i)=>value+[.1,-.35,.15][i])) : original;
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

export function frameComparison(coupling) {
  const original = currentVelocity(coupling);
  const transformed = currentVelocity(coupling,1);
  const velocityError = Math.max(...original.velocities.map((v,i)=>norm(subtract(rotateQuarterTurn(v),transformed.velocities[i]))));
  const distanceError = Math.max(...original.next.flatMap((p,i)=>original.next.slice(i+1).map((q,j)=>
    Math.abs(norm(subtract(p,q)) - norm(subtract(transformed.next[i],transformed.next[i+j+1]))))));
  return {original, transformed, velocityError, distanceError};
}
