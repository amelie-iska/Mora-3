const axisNames = ["x", "y", "z"];
const atoms = [
  {name: "A₀", fixed: [0, 0, 0]},
  {name: "A₁", token: [0.60, -0.30, 0.50], center: [0.95, 0.55, 0.10], widths: [2, 3, 1.5]},
  {name: "A₂", token: [-0.40, 0.70, 0.35], center: [-0.65, 0.85, 0.25], widths: [1.5, 2.5, 1]},
];
const heads = {
  forward: {
    weights: [[1.2, -.4, .5], [-.3, .7, .8], [.6, .2, -.4], [.2, .5, -.3], [-.4, .1, .4], [.3, -.2, .6]],
    bias: [.10, -.12, .06, -.35, -.48, -.56],
  },
  backward: {
    weights: [[-.7, .6, .3], [.4, -.5, .6], [-.2, .9, .1], [.6, -.4, .2], [.2, .3, -.5], [-.1, .7, .4]],
    bias: [-.12, .16, -.08, -.52, -.31, -.62],
  },
};
const readoutScale = .8 / 2;
const temperature = 350;
const energyOffset = .020;
const horizon = 4;
const energyTerm = -energyOffset / (8.617333262145e-5 * temperature);
const axes = atoms.slice(1).flatMap(atom => axisNames.map(axis => atom.name + " · " + axis));
const logSigmoid = x => -Math.max(0, -x) - Math.log1p(Math.exp(-Math.abs(x)));
const fmt = value => value.toFixed(3);
const triple = values => "(" + values.map(fmt).join(", ") + ")";
const code = "https://github.com/amelie-iska/Mora-3/blob/7f016ccb2b0f93aa67e7fd21f4cb70e72ba6a57b/src/mora3/";

export function cartesianWalkthroughState({direction = "forward", axis = 0, epsilon = .75, boxScale = 1, optionalStop = false} = {}) {
  if (!Object.hasOwn(heads, direction) || !Number.isInteger(axis) || axis < 0 || axis >= axes.length
    || !Number.isFinite(epsilon) || !Number.isFinite(boxScale) || boxScale <= 0 || typeof optionalStop !== "boolean") {
    throw new RangeError("Choose a finite Gaussian draw and a positive Cartesian box scale.");
  }

  const head = heads[direction];
  const headRows = [];
  const coordinates = [];
  const positions = atoms.map((atom, atomIndex) => {
    if (atom.fixed) return [...atom.fixed];

    const projected = head.weights.map((weights, index) =>
      weights.reduce((sum, weight, j) => sum + weight * atom.token[j] * readoutScale, head.bias[index]));
    const mean = projected.slice(0, 3);
    const logScale = projected.slice(3);
    headRows.push({atom: atom.name, token: [...atom.token], mean, logScale});
    return mean.map((value, j) => {
      const index = 3 * (atomIndex - 1) + j;
      const draw = index === axis ? epsilon : 0;
      const z = value + Math.exp(logScale[j]) * draw;
      const width = atom.widths[j] * boxScale;
      const lower = atom.center[j] - width / 2;
      const upper = atom.center[j] + width / 2;
      const fraction = Math.exp(logSigmoid(z));
      const x = lower + width * fraction;
      if (!(Number.isFinite(x) && lower < x && x < upper)) {
        throw new RangeError("The sample cannot be represented inside the open box.");
      }
      const baseTerm = logSigmoid(z) + logSigmoid(-z);
      const logWidth = Math.log(width);
      coordinates.push({
        label: axes[index], mean: value, logScale: logScale[j], draw, z, lower, upper, x, fraction,
        logWidth, baseTerm, logJacobian: logWidth + baseTerm,
        logDensityZ: -.5 * draw * draw - logScale[j] - .5 * Math.log(2 * Math.PI),
      });
      return x;
    });
  });

  const sum = key => coordinates.reduce((total, row) => total + row[key], 0);
  const logDensityZ = sum("logDensityZ");
  const logJacobian = sum("logJacobian");
  const logVolume = sum("logWidth");
  const baseCorrection = sum("baseTerm");
  const layerTerm = optionalStop ? -Math.log(horizon) : 0;
  return {
    headRows, coordinates, positions, selected: coordinates[axis], logDensityZ, logJacobian, logVolume,
    baseCorrection, logDensityX: logDensityZ - logJacobian, energyTerm, layerTerm,
    terminalLogReward: energyTerm + baseCorrection + layerTerm,
  };
}

export function renderCartesianWalkthrough(document) {
  const element = (tag, content, className) => {
    const node = document.createElement(tag);
    if (content !== undefined) node.textContent = content;
    if (className) node.className = className;
    return node;
  };
  const panel = element("section", undefined, "cartesian-walkthrough");
  panel.setAttribute("aria-label", "Native Cartesian GFlowNet walkthrough");
  const title = element("h3", "From atom tokens to bounded XYZ");
  const intro = element("p", "Synthetic calculation with toy encoded tokens and head weights. In the native policy, TokenGT encodes the complete current system. The Gaussian heads then read only the free atom tokens.");
  const control = element("fieldset", undefined, "cartesian-controls");
  control.append(element("legend", "Change the sample"));
  const select = (label, key, items) => {
    const field = element("label", label);
    const input = element("select");
    input.setAttribute("data-cartesian-control", key);
    for (const [value, text] of items) {
      const option = element("option", text);
      option.value = value;
      input.append(option);
    }
    field.append(input);
    control.append(field);
    return input;
  };
  const range = (label, key, min, max, step, value) => {
    const field = element("label", label);
    const input = element("input");
    Object.assign(input, {type: "range", min, max, step, value});
    input.setAttribute("data-cartesian-control", key);
    const output = element("output", String(value));
    output.setAttribute("aria-hidden", "true");
    field.append(input, output);
    control.append(field);
    return {input, output};
  };
  const direction = select("Gaussian head", "direction", [["forward", "Forward destination"], ["backward", "Backward predecessor"]]);
  const axis = select("Free coordinate", "axis", axes.map((name, index) => [String(index), name]));
  const draw = range("Gaussian residual ε", "epsilon", -2.5, 2.5, .25, .75);
  const width = range("Box width multiplier", "box-scale", .5, 2, .25, 1);
  const stopLabel = element("label");
  const stop = element("input");
  stop.type = "checkbox";
  stop.setAttribute("data-cartesian-control", "optional-stop");
  stopLabel.append(stop, element("span", "Allow stopping before layer 4"));
  control.append(stopLabel);

  const pipeline = element("ol", undefined, "cartesian-steps");
  pipeline.append(
    element("li", "The complete graph has 1 graph token, 3 atom tokens, and 4 directed bond tokens."),
    element("li", "The free atom mask [0, 1, 1] gathers two atom tokens. Each direction uses a shared linear head with six outputs per free atom."),
    element("li", "Split each head row into (μₓ, μᵧ, μ_z) and (log σₓ, log σᵧ, log σ_z). Flatten each block to six coordinates in atom order."),
    element("li", "Draw z = μ + exp(log σ) ε. The selected axis uses the slider; the other residuals are zero in this example."),
  );
  const headTable = element("table", undefined, "cartesian-head-table");
  headTable.append(element("caption", "The same linear head is applied to both free atom tokens"));
  const header = element("thead");
  const headerRow = element("tr");
  for (const label of ["Atom", "Toy encoded token", "Means μ", "Log scales log σ"]) {
    const cell = element("th", label);
    cell.scope = "col";
    headerRow.append(cell);
  }
  header.append(headerRow);
  const headBody = element("tbody");
  headTable.append(header, headBody);

  const bound = element("div", undefined, "cartesian-interval");
  const position = element("p", undefined, "cartesian-coordinates");
  const interval = element("p");
  const meter = element("meter");
  meter.min = 0;
  meter.max = 1;
  meter.setAttribute("aria-label", "Selected coordinate as a fraction of its open box interval");
  const sample = element("p");
  sample.setAttribute("aria-live", "polite");
  bound.append(element("h4", "Open Cartesian box"), interval, meter, sample, position,
    element("p", "x = lower + (upper − lower) sigmoid(z). Fixed A₀ stays at its supplied XYZ and has no Gaussian outputs. The native map refuses numerical saturation at a boundary."));

  const terms = element("dl", undefined, "cartesian-measures");
  const term = (label, key) => {
    const value = element("dd");
    value.setAttribute("data-cartesian-result", key);
    terms.append(element("dt", label), value);
    return value;
  };
  const densityZ = term("Joint Gaussian log density in logits, log p(z)", "log-density-z");
  const jacobian = term("Cartesian log Jacobian, log |det ∂x/∂z|", "log-jacobian");
  const volume = term("Free box log volume in Å⁶, log V", "log-volume");
  const densityX = term("Transition log density per Cartesian volume, log p(x)", "log-density-x");
  const correction = term("Normalized-uniform-box correction, log |det J| − log V", "base-correction");
  const terminal = term("Terminal log reward for the toy energy offset", "log-reward");
  const measures = element("div", undefined, "cartesian-density");
  measures.append(element("h4", "Transition density and target base measure"), terms,
    element("p", "The transition density transforms as log p(x) = log p(z) − log |det J|. The target instead uses the normalized box base dx/V. Its pullback adds log |det J| − log V to the energy term. At fixed z, scaling the box changes log |det J| and log V by the same amount."),
    element("p", "The box control holds the toy encoder outputs fixed to isolate this cancellation. Native box bounds are inputs to the encoder. The example uses ΔE = 0.020 eV and 350 K. Its four layers are generative steps."));

  const sources = element("p", "Source ");
  const sourceLinks = [
    ["free-token heads", "model/cartesian_gfn_policy.py#L293-L304"],
    ["box transform and reward", "model/cartesian_gfn_policy.py#L49-L88"],
    ["box support", "boltzmann_target.py#L14-L51"],
  ];
  for (const [index, [label, path]] of sourceLinks.entries()) {
    if (index) sources.append(element("span", " · "));
    const link = element("a", label);
    link.href = code + path;
    sources.append(link);
  }

  const update = () => {
    const state = cartesianWalkthroughState({
      direction: direction.value, axis: Number(axis.value), epsilon: Number(draw.input.value),
      boxScale: Number(width.input.value), optionalStop: stop.checked,
    });
    draw.output.textContent = Number(draw.input.value).toFixed(2);
    width.output.textContent = Number(width.input.value).toFixed(2) + "×";
    headBody.replaceChildren(...state.headRows.map(row => {
      const tr = element("tr");
      const name = element("th", row.atom);
      name.scope = "row";
      tr.append(name, ...[row.token, row.mean, row.logScale].map(value => element("td", triple(value))));
      return tr;
    }));
    const selected = state.selected;
    meter.value = selected.fraction;
    interval.textContent = selected.label + " ∈ (" + fmt(selected.lower) + ", " + fmt(selected.upper) + ") Å";
    sample.textContent = "z = " + fmt(selected.mean) + " + exp(" + fmt(selected.logScale) + ") × " + fmt(selected.draw)
      + " = " + fmt(selected.z) + ". The resulting coordinate is " + fmt(selected.x) + " Å.";
    position.textContent = "XYZ in atom order · " + state.positions.map((xyz, index) => atoms[index].name + " " + triple(xyz)).join(" · ") + " Å";
    densityZ.textContent = fmt(state.logDensityZ);
    jacobian.textContent = fmt(state.logJacobian);
    volume.textContent = fmt(state.logVolume);
    densityX.textContent = fmt(state.logDensityX);
    correction.textContent = fmt(state.baseCorrection);
    terminal.textContent = fmt(state.terminalLogReward) + " = " + fmt(state.energyTerm) + " energy + " + fmt(state.baseCorrection)
      + " box" + (stop.checked ? " − log(4) layer" : "");
  };
  for (const input of [direction, axis, stop]) input.addEventListener("change", update);
  for (const {input} of [draw, width]) input.addEventListener("input", update);
  panel.append(title, intro, control, pipeline, headTable,
    element("p", "The toy head uses W · (0.8/2 × token) + b. The μP readout scales the input and leaves the bias unscaled."),
    bound, measures, sources);
  update();
  return panel;
}
