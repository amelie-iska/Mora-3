"use strict";

const ui = Object.fromEntries(["method", "title", "status", "description", "formula", "control", "result", "source", "plot", "caption"]
  .map(name => [name, document.getElementById("lab-" + name)]));
const code = "https://github.com/amelie-iska/Mora-3/blob/7f016ccb2b0f93aa67e7fd21f4cb70e72ba6a57b/src/mora3/";
const implemented = "Implemented method · synthetic example";
const intended = "Research direction · synthetic example";
const label = (x, y, value, style = "ink") => '<text x="' + x + '" y="' + y + '" class="' + style + '">' + value + "</text>";
const line = (x1, y1, x2, y2, style = "stroke") => '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + style + '"/>';
const dot = (x, y, style = "aqua", r = 6) => '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" class="' + style + '"/>';
const box = (x, y, w, h, style = "box") => '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="5" class="' + style + '"/>';
const arrow = (x1, y1, x2, y2) => line(x1, y1, x2, y2).replace("/>", ' marker-end="url(#lab-arrow)"/>');
const grid = () => [80, 140, 200, 260].map(y => line(42, y, 520, y, "grid-line")).join("")
  + [100, 190, 280, 370, 460].map(x => line(x, 48, x, 285, "grid-line")).join("");
const point = (z, name, style) => dot(...z, style) + label(z[0] + 9, z[1] - 10, name, style);
const scalar = (value, digits = 2) => Number(value).toFixed(digits);
const xy = ([x, y]) => [68 + 85 * x, 278 - 70 * y];
const result = (figure, summary, caption) => ({figure, summary, caption});

const inputs = [
  {graph: ["C", "N", "O"], names: ["atom", "bond", "atom"], output: ["assay head", "affinity scalar", "target, method, unit"]},
  {graph: ["A", "G", "V"], names: ["residue", "contact", "residue"], output: ["structure head", "Cartesian coordinates", "atom IDs, frame, Å"]},
  {graph: ["A", "B", "C"], names: ["cell", "adjacency", "cell"], output: ["response head", "expression vector", "genes, condition, time"]},
  {graph: ["R", "T", "P"], names: ["reactant", "relation", "product"], output: ["reaction head", "geometry / features", "atom map, role, frame"]}
];
const methods = {
  inputs: {
    title: "Graph inputs and outputs", status: "Implemented components · synthetic examples",
    description: "The graph encoder represents typed nodes and edges. A task-specific readout preserves the fields needed to interpret its output.",
    formula: "nodes + edges → token states → task readout",
    source: "outputs.html", sourceLabel: "Output types and evaluation",
    control: {label: "Example route", options: ["Molecular complex", "Protein structure", "Cell perturbation", "Reaction"]},
    render(index) {
      const entry = inputs[index];
      const positions = [[62, 108], [139, 158], [63, 211]];
      const graph = line(62, 108, 139, 158) + line(139, 158, 63, 211) + line(62, 108, 63, 211)
        + positions.map((p, i) => dot(...p, i === 1 ? "amber" : "aqua", 19) + label(p[0] - 5, p[1] + 6, entry.graph[i], "dark-ink")).join("");
      const tokens = entry.names.map((name, i) => box(206, 97 + i * 43, 125, 33)
        + label(214, 119 + i * 43, name, i === 1 ? "amber" : "aqua")).join("");
      return result(label(30, 58, "GRAPH", "mute") + label(208, 58, "TOKEN STATES", "mute")
        + label(388, 58, "READOUT", "mute") + graph + arrow(155, 158, 198, 158)
        + tokens + arrow(336, 158, 379, 158) + box(387, 130, 155, 54)
        + label(399, 163, entry.output[0], "blue") + label(30, 279, entry.output[1], "ink")
        + label(30, 307, entry.output[2], "mute"), entry.output[1] + " with " + entry.output[2] + ".",
        "Green nodes and amber edge or relation features feed a blue readout.");
    }
  },
  tokenizer: {
    title: "ConvexTok segmentation", status: implemented,
    description: "Vocabulary fitting uses a continuous relaxation and rounds its scores into a bounded vocabulary. Encoding uses shortest-path segmentation over retained byte strings and the byte fallback.",
    formula: "d[j] = min { d[i] + 1 | bytes[i:j] in vocabulary }",
    source: code + "tokenization/shortest_path.py", sourceLabel: "Shortest-path encoder source",
    control: {label: "Available learned tokens", options: ["Byte fallback", "Add AC and GT", "Add ACGT too"]},
    render(index) {
      const payload = "ACGTACGT";
      const learned = ["AC", "GT", "ACGT"].slice(0, [0, 2, 3][index]);
      const tokens = [...learned, "A", "C", "G", "T"];
      const paths = Array(payload.length + 1).fill(null);
      paths[0] = [];
      for (let start = 0; start < payload.length; start += 1) {
        for (const token of tokens) {
          if (!payload.startsWith(token, start)) continue;
          const end = start + token.length;
          const path = [...paths[start], token];
          if (paths[end] === null || path.length < paths[end].length) paths[end] = path;
        }
      }
      const path = paths[payload.length];
      const cells = [...payload].map((c, i) => box(36 + 61 * i, 62, 52, 50) + label(58 + 61 * i, 93, c)).join("");
      let offset = 0;
      const spans = path.map(token => {
        const x = 36 + 61 * offset;
        offset += token.length;
        return box(x, 162, 61 * token.length - 9, 49, token.length > 1 ? "aqua" : "blue")
          + label(x + 12, 192, token, "dark-ink");
      }).join("");
      return result(label(36, 41, "INPUT BYTES", "mute") + cells + arrow(271, 116, 271, 153) + spans
        + label(36, 262, "SHORTEST PATH · " + path.length + " TOKENS", "mute"),
        path.join(" | ") + " uses " + path.length + " tokens and decodes to " + payload + ".",
        "The lower row spans the original bytes assigned to each emitted token.");
    }
  },
  attention: {
    title: "Tropical attention", status: implemented,
    description: "Negative Hilbert range scores each key. Each output channel independently picks the largest score-plus-value among valid keys.",
    formula: "sⱼ = −(max(q−kⱼ) − min(q−kⱼ)); yᵣ = maxⱼ(sⱼ + vⱼᵣ)",
    source: code + "model/tropical.py", sourceLabel: "Tropical attention source",
    control: {label: "Second query coordinate", min: -2, max: 2, step: 0.25, value: 0},
    render(value) {
      const keys = [[0, 0], [1, -1], [1, 1]];
      const values = [[0, 1], [2, -0.5], [-0.5, 2]];
      const scores = keys.map(([x, y]) => -Math.abs((1 - x) - (value - y)));
      const channels = [0, 1].map(i => scores.map((score, j) => score + values[j][i]));
      const maxima = channels.map(column => Math.max(...column));
      const headers = label(32, 68, "KEY", "mute") + label(165, 68, "SCORE", "mute")
        + label(280, 68, "S + V₀", "mute") + label(403, 68, "S + V₁", "mute");
      const rows = scores.map((score, j) => {
        const y = 111 + j * 49;
        return label(32, y, "k" + j + " (" + keys[j].join(",") + ")")
          + label(165, y, scalar(score), "blue")
          + channels.map((column, i) => label(280 + 123 * i, y, scalar(column[j]),
            column[j] === maxima[i] ? "aqua" : "mute")).join("") + line(32, y + 15, 514, y + 15, "grid-line");
      }).join("");
      const selected = maxima.map(s => scalar(s)).join(", ");
      return result(headers + rows + label(32, 279, "MAX PER CHANNEL → (" + selected + ")", "aqua"),
        "With q = (1, " + scalar(value) + "), the output is (" + selected + ").",
        "Green cells attain the channel maximum. Ties remain visible.");
    }
  },
  ring: {
    title: "Tropical ring reduction", status: intended,
    description: "A proposed ring keeps a query shard local while key and value blocks circulate. A channel-wise maximum merges incoming partial results. The local computation uses the implemented tropical operator.",
    formula: "mₜ = max(mₜ₋₁, maxⱼ∈Bₜ(sⱼ + vⱼ))",
    source: "architecture.html", sourceLabel: "Attention design and evidence",
    control: {label: "Blocks seen by query shard 0", min: 1, max: 4, step: 1, value: 2},
    render(count) {
      const blocks = [[0.3, 1.2], [1.5, 0.1], [-0.4, 1.7], [1.3, 0.8]];
      const slots = [[184, 84], [312, 137], [184, 218], [56, 137]];
      const visible = blocks.slice(0, count);
      const max = [0, 1].map(i => Math.max(...visible.map(block => block[i])));
      const links = slots.map(([x, y], i) => {
        const [toX, toY] = slots[(i + 1) % slots.length];
        return arrow(x + 0.18 * (toX - x), y + 0.18 * (toY - y), x + 0.78 * (toX - x), y + 0.78 * (toY - y));
      }).join("");
      const shards = slots.map(([x, y], i) => dot(x, y, i < count ? "aqua" : "blue", 21)
        + label(x - 11, y + 6, "B" + i, "dark-ink")).join("");
      return result(label(42, 36, "KEY–VALUE BLOCK RING", "mute") + links + shards
        + label(141, 147, "query 0", "ink") + box(355, 99, 164, 95)
        + label(369, 130, "RUNNING MAX", "mute") + label(369, 166, max.map(v => scalar(v, 1)).join("  /  "), "aqua")
        + label(42, 290, "MERGED " + count + " OF " + blocks.length + " BLOCKS", "mute"),
        "After " + count + " blocks, the two channel maxima are " + max.map(v => scalar(v, 1)).join(" and ") + ".",
        "Green shows which blocks reached the local query. This is a reduction example with toy partial maxima.");
    }
  },
  concept: {
    title: "Latent concept steering", status: implemented,
    description: "The concept editor learns an affine transform of a fixed input basis and normalizes the resulting directions. A signed step scales one direction and adds it to the latent state before a readout.",
    formula: "z′ = z + α dᵢ,  ‖dᵢ‖₂ = 1",
    source: code + "reasoning/graphcg.py", sourceLabel: "GraphCG concept editor source",
    control: {label: "Signed steering strength α", min: -2, max: 2, step: 0.1, value: 1},
    render(value) {
      const original = [2.1, 1.5];
      const direction = [0.8, 0.6];
      const edit = original.map((coordinate, i) => coordinate + value * direction[i]);
      return result(grid() + arrow(...xy(original), ...xy(edit))
        + point(xy(original), "z", "blue") + point(xy(edit), "z′", "aqua")
        + label(48, 34, "2D LATENT SLICE", "mute") + label(339, 314, "α = " + scalar(value, 1), "amber"),
        "The synthetic state moves to (" + edit.map(v => scalar(v)).join(", ") + ").",
        "The displayed direction is synthetic. Biological meaning requires independent calibration.");
    }
  },
  gfn: {
    title: "Latent GFlowNet", status: implemented,
    description: "A policy makes continuous latent moves and chooses when to stop. Trajectory balance compares path log densities with terminal log reward and counts stop mass once.",
    formula: "residual = log Z + Σ(log Pᶠ − log Pᵇ) + log Pstop − log R",
    source: code + "reasoning/continuous_gfn.py", sourceLabel: "Trajectory balance source",
    control: {label: "Terminal log reward", min: -2, max: 2, step: 0.1, value: 0.5},
    render(reward) {
      const balance = 1.5 - 0.3 - 0.45 + 0.1 - 0.35 - reward;
      return result(grid() + arrow(72, 202, 217, 137) + arrow(217, 137, 385, 101)
        + arrow(217, 137, 373, 230) + point([72, 202], "z₀", "blue") + point([217, 137], "z₁", "blue")
        + point([385, 101], "stop", "aqua") + point([373, 230], "continue", "amber")
        + label(52, 34, "ILLUSTRATIVE LATENT TRAJECTORY", "mute")
        + label(52, 313, "LOG BALANCE RESIDUAL = " + scalar(balance), "aqua"),
        "Residual " + scalar(balance) + " yields squared loss " + scalar(balance ** 2, 3) + " for this path.",
        "The green stop ends the selected path. Amber shows an alternative continuation.");
    }
  },
  forest: {
    title: "Forest-of-thought", status: implemented,
    description: "A bounded search can expand several roots in latent space. Score-ordered selection keeps branches that satisfy a minimum distance from previously selected states.",
    formula: "retain candidates by score, subject to a distance threshold and beam budget",
    source: code + "reasoning/fot_selection.py", sourceLabel: "Branch selection source",
    control: {label: "Minimum latent separation", min: 0, max: 3.5, step: 0.1, value: 1},
    render(threshold) {
      const roots = [[0.5, 1.1], [0.5, 2.4]];
      const candidates = [
        {state: [2.6, 2.8], score: 0.94, root: 1},
        {state: [2.8, 2.6], score: 0.90, root: 1},
        {state: [3.4, 1.25], score: 0.82, root: 0},
        {state: [4.6, 0.72], score: 0.71, root: 0}
      ];
      const selected = [];
      candidates.forEach(candidate => {
        if (selected.length === 3) return;
        if (selected.every(kept => Math.hypot(...kept.state.map((v, i) => v - candidate.state[i])) >= threshold)) selected.push(candidate);
      });
      const branches = candidates.map((candidate, index) => {
        const to = xy(candidate.state);
        const style = selected.includes(candidate) ? "aqua" : "mute";
        const selectedLabelOffset = index === 1 ? 27 : -10;
        return arrow(...xy(roots[candidate.root]), ...to) + dot(...to, style)
          + label(to[0] + 9, to[1] + selectedLabelOffset, scalar(candidate.score), style);
      }).join("");
      return result(grid() + branches + roots.map((root, i) => point(xy(root), "root " + (i + 1), "blue")).join("")
        + label(48, 34, "CANDIDATE SCORES", "mute"), selected.length + " of 4 branches retained at minimum separation " + scalar(threshold, 1) + ".",
        "Selected branches are green. Gray branches fell outside the retained beam.");
    }
  },
  analogy: {
    title: "Latent analogy", status: intended,
    description: "A proposed retrieval test applies a reference displacement to a different source. Nearest candidates need independent evidence that the transformation transfers across contexts.",
    formula: "query = z(C) + β[z(B) − z(A)]",
    source: "architecture.html", sourceLabel: "Architecture and reasoning evidence",
    control: {label: "Reference displacement scale β", min: 0, max: 1.5, step: 0.1, value: 1},
    render(scale) {
      const a = [0.4, 0.5], b = [2.1, 1.2], c = [0.8, 1.9];
      const target = c.map((coordinate, i) => coordinate + scale * (b[i] - a[i]));
      const candidates = [[1.35, 2.2], [2.6, 2.8], [3.7, 2.95]];
      const distances = candidates.map(z => Math.hypot(...z.map((v, i) => v - target[i])));
      const nearest = distances.indexOf(Math.min(...distances));
      return result(grid() + arrow(...xy(a), ...xy(b)) + arrow(...xy(c), ...xy(target))
        + point(xy(a), "A", "blue") + point(xy(b), "B", "blue") + point(xy(c), "C", "blue")
        + candidates.map((z, i) => point(xy(z), "D" + (i + 1), i === nearest ? "aqua" : "mute")).join("")
        + dot(...xy(target), "amber", 5) + label(48, 34, "NEAREST LATENT CANDIDATE", "mute"),
        "D" + (nearest + 1) + " is nearest to the synthetic query at distance " + scalar(distances[nearest]) + ".",
        "Amber marks the extrapolated query. Green marks a retrieved candidate.");
    }
  },
  topology: {
    title: "Topological analogy check", status: intended,
    description: "The research plan studies whether a mapping between latent neighborhoods preserves connectivity and loops across scales. This small Vietoris–Rips filtration compares two four-point neighborhoods.",
    formula: "include edge (i, j) when ‖zᵢ − zⱼ‖₂ ≤ ε; fill every clique",
    source: "architecture.html", sourceLabel: "Reasoning design and evidence",
    control: {label: "Filtration radius ε", min: 0.6, max: 2.2, step: 0.05, value: 1.15},
    render(radius) {
      const neighborhoods = [
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [1.2, 0.1], [1.35, 0.85], [0.14, 0.95]]
      ];
      const panels = neighborhoods.map((states, group) => {
        const position = states.map(([x, y]) => [70 + group * 255 + 116 * x, 229 - 120 * y]);
        const adjacent = states.map((a, i) => states.map((b, j) => i !== j
          && Math.hypot(a[0] - b[0], a[1] - b[1]) <= radius));
        const roots = states.map((_, i) => i);
        const root = i => roots[i] === i ? i : root(roots[i]);
        let edges = 0;
        let triangles = 0;
        let strokes = "";
        let faces = "";
        for (let i = 0; i < states.length; i += 1) {
          for (let j = i + 1; j < states.length; j += 1) {
            if (!adjacent[i][j]) continue;
            edges += 1;
            roots[root(j)] = root(i);
            strokes += line(...position[i], ...position[j]);
            for (let k = j + 1; k < states.length; k += 1) {
              if (!adjacent[i][k] || !adjacent[j][k]) continue;
              triangles += 1;
              faces += '<polygon class="lab-face" points="' + [i, j, k].map(index => position[index].join(",")).join(" ") + '"/>';
            }
          }
        }
        const components = new Set(states.map((_, i) => root(i))).size;
        const rank = triangles === 4 ? 3 : triangles;
        const loops = edges - states.length + components - rank;
        const textX = 56 + 255 * group;
        return {
          figure: label(textX, 58, "NEIGHBORHOOD " + (group ? "B" : "A"), "mute") + faces + strokes
            + position.map(p => dot(...p)).join("") + label(textX, 284, "β₀ " + components + "    β₁ " + loops, "aqua"),
          summary: " " + (group ? "B" : "A") + " has " + components + " component(s) and " + loops + " loop(s)."
        };
      });
      return result(line(278, 47, 278, 291, "grid-line") + panels.map(p => p.figure).join(""),
        "At radius " + scalar(radius) + "," + panels.map(p => p.summary).join(""),
        "β₀ counts components. β₁ counts independent unfilled loops. Comparing these curves is a proposed diagnostic.");
    }
  },
  geometry: {
    title: "Structure and dynamics", status: "Implemented components · synthetic geometry",
    description: "A structure output specifies atom identities and a coordinate frame. A coordinate-velocity head can act on the current frame. Flow time in this view indexes an illustrative path through coordinate space.",
    formula: "x(t) = (1 − t)x₀ + tx₁,  dx/dt = x₁ − x₀",
    source: code + "model/current_frame_velocity.py", sourceLabel: "Coordinate-velocity readout source",
    control: {label: "Flow interpolation time t", min: 0, max: 1, step: 0.05, value: 0.35},
    render(time) {
      const start = [[112, 164], [231, 124], [288, 212], [415, 171]];
      const end = [[112, 164], [192, 81], [327, 126], [416, 213]];
      const current = start.map((p, i) => p.map((x, j) => x + time * (end[i][j] - x)));
      const bonds = positions => positions.slice(1).map((p, i) => line(...positions[i], ...p)).join("");
      const guides = end.map(p => '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="13" class="target-ring"/>').join("");
      const velocities = current.slice(1).map((p, i) => {
        const atom = i + 1;
        return arrow(...p, ...p.map((v, j) => v + 0.29 * (end[atom][j] - start[atom][j])));
      }).join("");
      const distance = Math.hypot(...current[1].map((v, i) => v - current[2][i])) / 50;
      return result(label(42, 42, "PROJECTED COORDINATE FRAME", "mute") + guides + bonds(current) + velocities
        + current.map((p, i) => dot(...p, i === 0 ? "blue" : "aqua", 10) + label(p[0] - 5, p[1] + 32, String(i + 1))).join("")
        + label(42, 291, "FLOW TIME " + scalar(time) + "    PAIR 2–3 " + scalar(distance) + " TOY UNITS", "mute"),
        "The anchor is fixed. Projected atoms 2 and 3 are " + scalar(distance) + " toy units apart.",
        "Arrows show the interpolation velocity. Dashed rings mark target coordinates. Flow time has no physical units.");
    }
  }
};

function show(name) {
  const method = methods[name];
  if (!method) return;
  const {control} = method;
  ui.title.textContent = method.title;
  ui.status.textContent = method.status;
  ui.description.textContent = method.description;
  ui.formula.textContent = method.formula;
  ui.source.href = method.source;
  ui.source.textContent = method.sourceLabel;
  const picker = document.createElement(control.options ? "select" : "input");
  const output = document.createElement("output");
  const pickerLabel = document.createElement("label");
  const holder = document.createElement("div");
  holder.className = "lab-controls";
  picker.id = "lab-parameter";
  pickerLabel.htmlFor = picker.id;
  pickerLabel.textContent = control.label;
  if (control.options) {
    control.options.forEach((choice, i) => picker.add(new Option(choice, i)));
  } else {
    Object.assign(picker, {type: "range", min: control.min, max: control.max, step: control.step, value: control.value});
    output.htmlFor = picker.id;
  }
  holder.append(pickerLabel, picker);
  if (!control.options) holder.append(output);
  ui.control.replaceChildren(holder);
  const update = () => {
    output.value = picker.value;
    const display = method.render(Number(picker.value));
    ui.plot.innerHTML = '<defs><marker id="lab-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6a91a4"/></marker></defs>' + display.figure;
    ui.plot.setAttribute("aria-label", method.title + ". " + display.summary);
    ui.result.textContent = display.summary;
    ui.caption.textContent = display.caption;
  };
  picker.addEventListener("input", update);
  update();
}

ui.method.addEventListener("change", () => show(ui.method.value));
show(ui.method.value);
