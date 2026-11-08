import {analogy, trajectory, currentVelocity} from "./mechanism-models.js";

const ui = Object.fromEntries(["method", "title", "status", "description", "formula", "control", "result", "source", "plot", "caption", "evidence"]
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
      const positions = [[56, 158], [157, 158]];
      const graph = arrow(76,158,133,158)
        + box(81,105,54,32) + label(90,127,entry.graph[1],"amber")
        + positions.map((point,i) => dot(...point,"aqua",19)
          + label(point[0]-5,point[1]+6,entry.graph[i*2],"dark-ink")).join("");
      const tokens = ["[GRAPH]", entry.names[0], entry.names[2], entry.names[1]].map((name, i) =>
        box(206, 84 + i * 39, 125, 31) + label(214, 105 + i * 39, name, i === 3 ? "amber" : "aqua")).join("");
      return result(label(30, 58, "GRAPH", "mute") + label(208, 58, "TOKEN STATES", "mute")
        + label(388, 58, "READOUT", "mute") + graph + arrow(155, 158, 198, 158)
        + tokens + arrow(336, 158, 379, 158) + box(387, 130, 155, 54)
        + label(399, 163, entry.output[0], "blue") + label(30, 279, entry.output[1], "ink")
        + label(30, 307, entry.output[2], "mute"), entry.output[1] + " with " + entry.output[2] + ".",
        "The directed edge has its own token. The encoder prepends a learned graph token. A task selects the appropriate state or states for its readout.");
    }
  },
  tokenizer: {
    title: "ConvexTok segmentation", status: implemented,
    description: "Vocabulary fitting uses a continuous relaxation and rounds its scores into a bounded vocabulary. Encoding uses shortest-path segmentation over retained byte strings and the byte fallback.",
    formula: "d[i] = min { 1 + d[j] | bytes[i:j] in vocabulary };  d[n] = 0",
    source: code + "tokenization/shortest_path.py", sourceLabel: "Shortest-path encoder source",
    control: {label: "Available learned tokens", options: ["Byte fallback", "Add AC and GT", "Add ACGT too"]},
    render(index) {
      const payload = "ACGTACGT";
      const learned = ["AC", "GT", "ACGT"].slice(0, [0, 2, 3][index]);
      const tokens = [...learned, "A", "C", "G", "T"];
      const costs = Array(payload.length).fill(Infinity).concat(0);
      const next = [];
      for (let start = payload.length - 1; start >= 0; start--) {
        for (const token of tokens) {
          if (!payload.startsWith(token,start)) continue;
          const cost = 1 + costs[start+token.length];
          if (cost < costs[start] || (cost === costs[start] && token.length > next[start].length)) {
            costs[start]=cost;
            next[start]=token;
          }
        }
      }
      const path=[];
      for (let i=0;i<payload.length;i+=next[i].length) path.push(next[i]);
      const cells = [...payload].map((c, i) => box(36 + 61 * i, 62, 52, 50) + label(58 + 61 * i, 93, c)).join("");
      let offset = 0;
      const spans = path.map(token => {
        const x = 36 + 61 * offset;
        offset += token.length;
        return box(x, 162, 61 * token.length - 9, 49, token.length > 1 ? "aqua" : "blue")
          + label(x + 12, 192, token, "dark-ink");
      }).join("");
      return result(label(36, 41, "INPUT BYTES", "mute") + cells + arrow(271, 116, 271, 153) + spans
        + label(36, 256, "SUFFIX COST d[i] AT EACH BYTE OFFSET", "mute")
        + costs.map((cost,i)=>label(41 + 59*i, 289, String(cost), "blue")).join(""),
        path.join(" | ") + " uses " + path.length + " tokens and decodes to " + payload + ".",
        "The suffix costs are computed right to left. Equal-cost paths choose the longer first token. Controls are separate from payload bytes.");
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
      const keyBlocks = [
        [[0,.1,.4,1.3],[.9,.3,.2,.8]],
        [[.7,.6,1.8,.3],[0,.75,1.1,.85]],
        [[.4,.9,.1,2.2],[.8,.4,-.1,1.0]],
        [[1.0,.7,1.6,.95],[.1,.4,.2,1.1]]
      ];
      const blocks=keyBlocks.map(keys => [2,3].map(channel => Math.max(...keys.map(key => {
        const differences=[.6-key[0],.8-key[1]];
        return Math.min(...differences)-Math.max(...differences)+key[channel];
      }))));
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
        "Green shows which blocks reached the local query. Each block reduces two synthetic keys with the same Hilbert score as the attention demo. The distributed ring is a design sketch.");
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
      const basis = [[1,0],[0,1]];
      const projection = [[.72,.54],[.48,.78]], bias=[.08,.12];
      const raw = basis.map(row => projection.map((w,i)=>w.reduce((sum,x,j)=>sum+x*row[j],bias[i])));
      const [direction,other] = raw.map(row=>row.map(x=>x/Math.hypot(...row)));
      const cosine=direction.reduce((sum,x,i)=>sum+x*other[i],0);
      const edit = original.map((coordinate, i) => coordinate + value * direction[i]);
      return result(grid() + arrow(...xy(original), ...xy(original.map((x,i)=>x+other[i]))) + arrow(...xy(original), ...xy(edit))
        + point(xy(original), "z", "blue") + point(xy(edit), "z′", "aqua")
        + label(48, 34, "2D LATENT SLICE", "mute") + label(339, 314, "α = " + scalar(value, 1), "amber"),
        "The state moves to (" + edit.map(v => scalar(v)).join(", ") + "). Direction cosine: " + scalar(cosine,3) + ".",
        "Both directions have unit norm after the same affine projection. The high cosine illustrates why normalization alone does not yield independent directions.");
    }
  },
  gfn: {
    title: "Latent GFlowNet", status: implemented,
    description: "Score a fixed path with two-dimensional Gaussian forward and reverse kernels. A noninitial move includes continuation probability. The final exit contributes stop probability once, and the reverse step to the origin is deterministic.",
    formula: "residual = log Z + Σ(log Pᶠ − log Pᵇ) + log Pstop − log R",
    source: code + "reasoning/gfn_sampling.py", sourceLabel: "Path scoring source",
    control: {label: "Terminal log reward", min: -2, max: 2, step: 0.1, value: 0.5,
      choices:["Stop at step 2", "Forced stop at horizon 3", "Stop at step 1"]},
    render(reward, choice) {
      const model=trajectory(reward,[2,3,1][choice]);
      return {...result(grid() + model.states.slice(1).map((state,i) => arrow(...xy(model.states[i]),...xy(state))).join("")
        + model.states.map((state,i) => point(xy(state),"z"+i,i===model.states.length-1?"aqua":"blue")).join("")
        + label(48, 34, "LAYERED LATENT PATH", "mute")
        + label(48, 315, "STOP LOG MASS " + scalar(model.stop) + "   LOG Z " + scalar(model.logZ), "aqua"),
        "Residual " + scalar(model.balance) + " yields squared loss " + scalar(model.balance ** 2, 3) + ".",
        "Densities describe continuous latent moves. Stop and continue are probability masses, with forced exit at the horizon."),
        table:{caption:"Terms in trajectory balance",columns:["Move","Forward density","Continue","Reverse density"],
          rows:model.terms.map((term,i)=>[String(i+1),...["forward","continuation","backward"].map(key=>scalar(term[key],3))])},
      };
    }
  },
  forest: {
    title: "Forest-of-thought", status: implemented,
    description: "Two trees keep independent beams. At each layer, filter invalid candidates, sort by score, then greedily enforce minimum separation. A retained terminal leaf is recorded and does not expand.",
    formula: "per tree, per layer: valid → stable score order → distance guard → beam",
    source: code + "reasoning/fot_forest.py", sourceLabel: "Bounded forest source",
    control: {label: "Minimum latent separation", min: 0, max: 1.4, step: 0.05, value: 0.45},
    render(threshold) {
      const panels=[[74,122],[74,245]].map(([x,y],tree) => {
        const root={id:"R"+(tree+1),xy:[x,y]};
        const nodes=[];
        let active=[root];
        for (let depth=1;depth<=2;depth++) {
          const layer=active.flatMap((parent,j)=>[0,1,2].map(i=>{
            const state=[depth*1.3+(i%2)*.2, j*.6+i*.27];
            return {id:parent.id+"."+i, parent, state, depth, xy:[x+depth*152,y-32+j*16+i*20],
              score:[.85,.96,.72][i]-.04*j-.03*depth, valid:i!==1 || tree===0 || depth===2,
              terminal: i===0};
          }));
          const selected=[];
          for (const item of layer.filter(item=>item.valid).sort((a,b)=>b.score-a.score)) {
            if (selected.length===2) break;
            if (selected.every(other=>Math.hypot(...item.state.map((v,j)=>v-other.state[j]))>=threshold)) {
              item.selected=true;
              selected.push(item);
            }
          }
          nodes.push(...layer);
          active=selected.filter(item=>!item.terminal);
          if (!active.length) break;
        }
        return {root,nodes};
      });
      const branches=panels.map(({root,nodes}) => point(root.xy,root.id,"blue") + nodes.map(item=>{
        const tint=!item.valid?"amber":item.selected?"aqua":"mute";
        return line(...item.parent.xy,...item.xy,item.selected?"stroke":"grid-line")
          + dot(...item.xy,tint,item.terminal?7:4) + label(item.xy[0]+10,item.xy[1]+4,scalar(item.score),tint);
      }).join("")).join("");
      const all=panels.flatMap(panel=>panel.nodes);
      return result(label(43,35,"ROOTS","mute") + label(198,35,"DEPTH 1","mute") + label(347,35,"DEPTH 2","mute") + branches,
        all.filter(item=>item.selected).length + " of " + all.length + " proposals retained across both trees.",
        "Green marks retained branches. Amber marks invalid proposals. Larger markers are terminal and do not expand.");
    }
  },
  analogy: {
    title: "Transfer a typed relation", status: intended,
    description: "Two synthetic pairs show the same change in a matched junction-conductance assay. Estimate the direction from either pair or both, then apply it to a new donor–receiver assay. Rank all four embedding coordinates and check roles, cell state, and the measured endpoint independently.",
    formula: "Δ = mean(z(after) − z(before)); q = z(new before) + βΔ",
    source: "architecture.html", sourceLabel: "Architecture and reasoning evidence",
    control: {label: "Reference displacement scale β", min: 0, max: 1.5, step: 0.05, value: 1,
      choices: ["Both reference pairs", "Matched pair 1", "Matched pair 2"]},
    render(scale, referenceMode) {
      const model = analogy(scale, referenceMode);
      const project = ([x,y]) => [56 + 138*x, 286 - 72*y];
      const marks = model.pairs.map(pair => {
        const id=pair.name.at(-1);
        const a=project(pair.before), b=project(pair.after);
        return arrow(...a,...b) + point(a, "R"+id+" pre", "blue") + point(b, "R"+id+" post", "blue");
      }).join("");
      const target = project(model.query), start = project(model.queryBefore);
      const marksById = model.ranked.map(candidate => {
        const [x,y]=project(candidate.state);
        const style=candidate === model.selected ? "aqua" : candidate.refusal ? "amber" : "mute";
        const offsets={A:[-36,-24], B:[-19,30], C:[25,-17], D:[19,33], E:[15,20], F:[-43,11]};
        const [dx,dy]=offsets[candidate.id];
        return line(x,y,x+dx,y+dy,"grid-line") + dot(x,y,style,candidate === model.selected ? 7 : 4)
          + label(x+dx,y+dy,candidate.id,style);
      }).join("");
      const display = result(grid() + label(44, 28, "COORDINATES 1–2 OF 4", "mute") + marks + arrow(...start,...target)
        + point(start,"new pre","blue") + marksById + '<circle cx="'+target[0]+'" cy="'+target[1]+'" r="10" class="target-ring"/>'
        + label(target[0]+13,target[1]-14,"q","amber") + label(44, 316, "Blue: reference   Green: selected   Amber: refused", "mute"),
        "2D nearest " + model.projected.id + " · 4D nearest " + model.ranked[0].id + " · eligible " + model.selected.id + " · no-transfer baseline " + model.baseline.id + ".",
        "Fixture embeddings are hand-authored. Closeness proposes a candidate for validation. It cannot establish a transferred effect.");
      return {...display, table: {
        caption: "Candidate audit in distance order · query (" + model.query.map(value=>scalar(value)).join(", ") + ")",
        columns: ["Candidate", "2D distance", "4D distance", "Eligibility"],
        rows: model.ranked.map(item => [item.id, scalar(item.projected,3), scalar(item.full,3), item.refusal ?? (item === model.selected ? "Selected" : "Eligible")]),
      }};
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
        // On four vertices, the four triangular boundaries have one dependency.
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
    title: "Current-frame structure and dynamics", status: "Diagnostic implementation · synthetic geometry",
    description: "Compute scalar coefficients from distance and bond features of a four-atom input. Multiply each relative displacement by its coefficient, average across neighbors including self, and zero the anchor velocity. The ghost frame shows one illustrative Euler step.",
    formula: "vᵢ = freeᵢ · (1/N) Σⱼ wᵀfᵢⱼ(xⱼ − xᵢ);  x′ = x + 0.4v",
    source: code + "model/current_frame_velocity.py", sourceLabel: "Coordinate-velocity readout source",
    control: {label: "Bond feature coefficient", min: -0.8, max: 1.6, step: 0.05, value: 0.9},
    render(coupling) {
      const {xyz,free,velocities,next,distance} = currentVelocity(coupling);
      const project = ([x,y,z]) => [79 + 98*x + 23*z, 176 - 100*y + 48*z];
      const bonds = xyz.slice(1).map((point,i) => line(...project(xyz[i]), ...project(point))).join("");
      const ghosts = next.map(point => {
        const [x,y]=project(point);
        return '<circle cx="'+x+'" cy="'+y+'" r="12" class="target-ring"/>';
      }).join("");
      return {...result(label(42, 42, "3D INPUT PROJECTED TO THE PAGE", "mute") + bonds + ghosts
        + xyz.map((point,i) => {
          const from=project(point);
          return (free[i] ? arrow(...from,...project(point.map((x,j)=>x+1.8*velocities[i][j]))) : "")
            + dot(...from, free[i] ? "aqua" : "blue", 9) + label(from[0]-5, from[1]+32, String(i+1));
        }).join("") + label(42, 291, "PAIR 2–3 AFTER STEP  " + scalar(distance,3) + " Å", "mute"),
        "Atom 1 is fixed. Atom 2 has speed " + scalar(Math.hypot(...velocities[1]),3) + " Å per toy step unit.",
        "Arrows magnify the velocity. Rings show the Euler update. This diagnostic has no calibrated physical time."),
        table:{caption:"Current-frame velocities",columns:["Atom","vₓ","vᵧ","v_z","Mask"],
          rows:velocities.map((v,i)=>[String(i+1),...v.map(x=>scalar(x,3)),free[i]?"Free":"Fixed"])},
      };
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
  const secondary = document.createElement("select");
  if (control.choices) {
    const secondaryLabel = document.createElement("label");
    secondary.id = "lab-choice";
    secondaryLabel.htmlFor = secondary.id;
    secondaryLabel.textContent = name === "analogy" ? "Reference pairs" : "Terminal step";
    control.choices.forEach((choice, i) => secondary.add(new Option(choice, i)));
    holder.append(secondaryLabel, secondary);
  }
  ui.control.replaceChildren(holder);
  const update = () => {
    output.value = picker.value;
    const display = method.render(Number(picker.value), Number(secondary.value));
    ui.plot.innerHTML = '<defs><marker id="lab-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6a91a4"/></marker></defs>' + display.figure;
    ui.plot.setAttribute("aria-label", method.title + ". " + display.summary);
    ui.result.textContent = display.summary;
    ui.caption.textContent = display.caption;
    ui.evidence.replaceChildren();
    if (display.table) {
      const table=document.createElement("table");
      table.createCaption().textContent=display.table.caption;
      const head=table.createTHead().insertRow();
      for (const title of display.table.columns) {
        const cell=document.createElement("th");
        cell.scope="col";
        cell.textContent=title;
        head.append(cell);
      }
      const body=table.createTBody();
      for (const values of display.table.rows) {
        const row=body.insertRow();
        for (const value of values) row.insertCell().textContent=value;
      }
      ui.evidence.append(table);
    }
  };
  picker.addEventListener("input", update);
  secondary.addEventListener("change", update);
  update();
}

function selectMethod() {
  const requested=location.hash.slice(1);
  ui.method.value=Object.hasOwn(methods,requested) ? requested : "inputs";
  show(ui.method.value);
}
ui.method.addEventListener("change", () => {
  history.replaceState(null,"","#"+ui.method.value);
  show(ui.method.value);
});
window.addEventListener("hashchange", selectMethod);
selectMethod();
