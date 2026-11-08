import {analogy, sourceMotif, trajectory, frameComparison, segmentBytes, hilbertTerms, hilbertScore, ripsAt, ripsLoopIntervals, topologyComparison, boltzmannStates, coolingTemperatures, energyExamples} from "./mechanism-models.js";

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

function motifDiagram(candidate) {
  const mapping = candidate.motif.mapping;
  if (!mapping) return "";
  const columns = [35, 210, 385];
  const roleStyles = ["aqua", "blue", "amber"];
  const roleCards = (types, y) => types.map((role, i) =>
    box(columns[i], y, 140, 32) + label(columns[i] + 12, y + 22, i + " " + role,
      roleStyles[sourceMotif.types.indexOf(role)])).join("");
  return '<svg class="lab-motif-figure" viewBox="0 0 560 166" role="img" aria-label="Role correspondence to candidate '
    + candidate.id + '. ' + mapping.map((j,i) => i + ' to ' + j).join(', ') + '">'
    + label(35, 22, "SOURCE MOTIF ORDER", "mute")
    + mapping.map((j,i) => line(columns[i] + 70, 64, columns[j] + 70, 104, "lab-role-link " + roleStyles[i])).join("")
    + roleCards(sourceMotif.types, 32) + roleCards(candidate.targetMotif.types, 104)
    + label(35, 158, "CANDIDATE " + candidate.id + " ORDER", "mute") + "</svg>";
}

const inputs = [
  {graph: ["C", "bond", "O"], names: ["atom", "bond", "atom"], output: ["assay head", "assay value or class", "endpoint, protocol, unit"]},
  {graph: ["A", "link", "V"], names: ["residue", "contact", "residue"], output: ["structure head", "Cartesian coordinates", "atom IDs, frame, Å"]},
  {graph: ["A", "adj", "C"], names: ["cell", "adjacency", "cell"], output: ["response head", "expression vector", "genes, condition, time"]},
  {graph: ["R", "rxn", "P"], names: ["reactant", "relation", "product"], output: ["reaction head", "product molecular string", "substrate, protein, product grammar"]}
];
const methods = {
  inputs: {
    title: "Graph inputs and outputs", status: "Implemented components · synthetic examples",
    description: "TokenGT adds a projected entity feature vector, a projection of two concatenated orthogonal endpoint IDs, and a learned node or edge type vector. A task readout chooses its output representation.",
    formula: "hᵢⱼ = Wfeat fᵢⱼ + Wid [pᵢ ∥ pⱼ] + e(type)",
    source: "outputs.html", sourceLabel: "Output types and evaluation",
    control: {label: "Example route", options: ["Molecular complex", "Protein structure", "Cell perturbation", "Reaction"]},
    render(index) {
      const entry = inputs[index];
      const positions = [[56, 158], [157, 158]];
      const graph = arrow(76,158,133,158)
        + box(81,105,54,32) + label(90,127,entry.graph[1],"amber")
        + positions.map((point,i) => dot(...point,"aqua",19)
          + label(point[0]-5,point[1]+6,entry.graph[i*2],"dark-ink")).join("");
      const tokens = ["[GRAPH]", entry.names[0] + " [i,i]", entry.names[2] + " [j,j]", entry.names[1] + " [i,j]"].map((name, i) =>
        box(206, 84 + i * 39, 125, 31) + label(214, 105 + i * 39, name, i === 3 ? "amber" : "aqua")).join("");
      return {...result(label(30, 58, "GRAPH", "mute") + label(208, 58, "TOKEN STATES", "mute")
        + label(388, 58, "READOUT", "mute") + graph + arrow(155, 158, 198, 158)
        + tokens + arrow(336, 158, 379, 158) + box(387, 130, 155, 54)
        + label(399, 163, entry.output[0], "blue") + label(30, 279, entry.output[1], "ink")
        + label(30, 307, entry.output[2], "mute"), entry.output[1] + " with " + entry.output[2] + ".",
        "Node tokens repeat their identifier. Directed edge tokens retain endpoint order. The learned graph token is separate. Different tasks can use pooled or per-entity states."),
        table:{caption:"Endpoint projection inputs for this graph",
          columns:["Token", "Type", "Packed endpoint IDs"],
          rows:[["[GRAPH]", "Special", "Separate learned vector"],
            [entry.graph[0], "Node", "(1, 0 | 1, 0)"], [entry.graph[2], "Node", "(0, 1 | 0, 1)"],
            [entry.graph[0] + " → " + entry.graph[2], "Edge", "(1, 0 | 0, 1)"]]},
        notes:[["Toy ID basis", "pᵢ = (1, 0), pⱼ = (0, 1). Reversing the edge would swap the two halves to (0, 1 | 1, 0)."]],
      };
    }
  },
  tokenizer: {
    title: "ConvexTok segmentation", status: implemented,
    description: "Vocabulary fitting uses a continuous relaxation and rounds its scores into a bounded vocabulary. Encoding uses shortest-path segmentation over retained byte strings and the byte fallback.",
    formula: "d[i] = min { 1 + d[j] | bytes[i:j] in vocabulary };  d[n] = 0",
    source: code + "tokenization/shortest_path.py", sourceLabel: "Shortest-path encoder source",
    control: {label: "Available learned tokens", options: ["Byte fallback", "Add AC and GT", "Add ACGT too", "Greedy trap: ACGT / ACG / TAC"]},
    render(index) {
      const payload = index === 3 ? "ACGTAC" : "ACGTACGT";
      const learned = index === 3 ? ["ACGT", "ACG", "TAC"] : ["AC", "GT", "ACGT"].slice(0, [0, 2, 3][index]);
      const {path,costs,greedy,firstChoices} = segmentBytes(payload,learned);
      const cells = [...payload].map((c, i) => box(36 + 61 * i, 62, 52, 50) + label(58 + 61 * i, 93, c)).join("");
      let offset = 0;
      const spans = path.map(token => {
        const x = 36 + 61 * offset;
        offset += token.length;
        return box(x, 162, 61 * token.length - 9, 49, token.length > 1 ? "aqua" : "blue")
          + label(x + 12, 192, token, "dark-ink");
      }).join("");
      return {...result(label(36, 41, "INPUT BYTES", "mute") + cells + arrow(271, 116, 271, 153) + spans
        + label(36, 256, "SUFFIX COST d[i] AT EACH BYTE OFFSET", "mute")
        + costs.map((cost,i)=>label(41 + 59*i, 289, String(cost), "blue")).join(""),
        "Shortest path " + path.join(" | ") + " uses " + path.length + " tokens. Greedy " + greedy.join(" | ") + " uses " + greedy.length + ".",
        "The suffix costs run right to left and resolve ties with the longer first token. Longest-first greed can lose. Each symbol here is one ASCII byte. The example takes the learned vocabulary as given and shows encoding."),
        table:{caption:"Prefix choices at byte offset 0",
          columns:["Prefix", "Next offset", "Suffix cost", "Total tokens", "Choice"],
          rows:firstChoices.map(({token,suffixCost,chosen})=>[token,token.length,suffixCost,1+suffixCost,chosen?"Shortest path":"Alternative"])},
        notes:[["Byte round trip", path.join("") === payload
          ? "Concatenating the chosen tokens reconstructs all " + payload.length + " input bytes in order."
          : "The chosen tokens fail to reconstruct the input."]],
      };
    }
  },
  attention: {
    title: "Tropical attention", status: implemented,
    description: "Negative Hilbert range scores each key. Each output channel independently picks the largest score-plus-value among valid keys.",
    formula: "sⱼ = −(max(q−kⱼ) − min(q−kⱼ)); yᵣ = maxⱼ(sⱼ + vⱼᵣ)",
    source: code + "model/tropical.py", sourceLabel: "Tropical attention source",
    control: {label: "Second query coordinate", min: -2, max: 2, step: 0.25, value: 0,
      secondaryLabel: "Key mask", choices:["All keys valid", "Mask k₁", "Mask k₂"]},
    render(value, mask) {
      const keys = [[0, 0], [1, -1], [1, 1]];
      const values = [[0, 1], [2, -0.5], [-0.5, 2]];
      const terms = keys.map(key => hilbertTerms([1,value],key));
      const scores = terms.map(({score},i) => i === mask && mask > 0 ? -Infinity : score);
      const channels = [0, 1].map(i => scores.map((score, j) => score + values[j][i]));
      const maxima = channels.map(column => Math.max(...column));
      const winners = channels.map((column,i)=>column.indexOf(maxima[i]));
      const headers = label(32, 68, "KEY", "mute") + label(165, 68, "SCORE", "mute")
        + label(280, 68, "S + V₀", "mute") + label(403, 68, "S + V₁", "mute");
      const rows = scores.map((score, j) => {
        const y = 111 + j * 49;
        return label(32, y, "k" + j + " (" + keys[j].join(",") + ")")
          + label(165, y, Number.isFinite(score) ? scalar(score) : "MASK", "blue")
          + channels.map((column, i) => label(280 + 123 * i, y, Number.isFinite(column[j]) ? scalar(column[j]) : "−∞",
            j === winners[i] ? "aqua" : column[j] === maxima[i] ? "amber" : "mute")).join("") + line(32, y + 15, 514, y + 15, "grid-line");
      }).join("");
      const selected = maxima.map(s => scalar(s)).join(", ");
      return {...result(headers + rows + label(32, 279, "MAX PER CHANNEL → (" + selected + ")", "aqua"),
        "With q = (1, " + scalar(value) + "), the output is (" + selected + "). Winning key indices: " + winners.join(", ") + ".",
        "Green is the first winning key per channel. Amber shows later ties. The two channels can choose different keys."),
        table:{caption:"Hilbert score before the key mask",
          columns:["Key", "q − k", "Min", "Max", "Min − max", "Valid"],
          rows:terms.map(({differences,min,max,score},i)=>["k"+i,differences.map(v=>scalar(v)).join(", "),
            scalar(min),scalar(max),scalar(score),Number.isFinite(scores[i])?"Yes":"Masked → −∞"])},
        notes:[["Next operation", "The full layer joins all heads, applies expm1, and projects the result. These two channels become ("
          + maxima.map(v=>scalar(Math.expm1(v),3)).join(", ") + ") before projection."]],
      };
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
        return hilbertScore([.6,.8],key.slice(0,2))+key[channel];
      }))));
      const slots = [[184, 84], [312, 137], [184, 218], [56, 137]];
      const visible = blocks.slice(0, count);
      let running = [-Infinity, -Infinity];
      const history = visible.map(block => {
        running = block.map((value,i)=>Math.max(value,running[i]));
        return running;
      });
      const max = history.at(-1);
      const dense = [2,3].map(channel => Math.max(...keyBlocks.flat().map(key =>
        hilbertScore([.6,.8],key.slice(0,2))+key[channel])));
      const links = slots.map(([x, y], i) => {
        const [toX, toY] = slots[(i + 1) % slots.length];
        return arrow(x + 0.18 * (toX - x), y + 0.18 * (toY - y), x + 0.78 * (toX - x), y + 0.78 * (toY - y));
      }).join("");
      const shards = slots.map(([x, y], i) => dot(x, y, i < count ? "aqua" : "blue", 21)
        + label(x - 11, y + 6, "B" + i, "dark-ink")).join("");
      const channels = pair => pair.map(value=>scalar(value,2)).join(" / ");
      return {...result(label(42, 36, "KEY–VALUE BLOCK RING", "mute") + links + shards
        + label(111, 124, "query stays at B0", "ink") + box(355, 99, 164, 95)
        + label(369, 130, "RUNNING MAX", "mute") + label(369, 166, channels(max), "aqua")
        + label(42, 290, "MERGED " + count + " OF " + blocks.length + " BLOCKS", "mute")
        + label(42, 318, "DENSE CHECK  " + channels(dense), "blue"),
        "After " + count + " blocks, the two channel maxima are " + max.map(v => scalar(v, 2)).join(" and ") + ".",
        "Green shows which blocks reached the local query. Each block reduces two synthetic keys with the same Hilbert score as the attention demo. The distributed ring is a design sketch."),
        table:{caption:"Merge history for the selected query",
          columns:["Arrival", "Block max · channels 0 / 1", "Running max · channels 0 / 1"],
          rows:visible.map((block,i)=>["B"+i,channels(block),channels(history[i])])},
        notes:[["Full reduction check", count === blocks.length ? "Running and unsharded maxima agree in both channels."
          : "Unsharded minus running maxima are " + channels(dense.map((v,i)=>v-max[i])) + ". Unseen blocks can only raise a maximum."]],
      };
    }
  },
  concept: {
    title: "Latent concept steering", status: implemented,
    description: "The concept editor learns an affine transform of a fixed input basis and normalizes the resulting directions. A signed step scales one direction and adds it to the latent state. This panel shows the edit before any decoder or downstream measurement.",
    formula: "z′ = z + α dᵢ,  ‖dᵢ‖₂ = 1",
    source: code + "reasoning/graphcg.py", sourceLabel: "GraphCG concept editor source",
    control: {label: "Signed steering strength α", min: -2, max: 2, step: 0.1, value: 1},
    render(value) {
      const original = [2.1, 1.5];
      const basis = [[1,0],[0,1]];
      const projection = [[.72,.54],[.48,.78]], bias=[.08,.12];
      const raw = basis.map(row => projection.map((w,i)=>w.reduce((sum,x,j)=>sum+x*row[j],bias[i])));
      const rawNorm = raw.map(row=>Math.hypot(...row));
      const [direction,other] = raw.map((row,i)=>row.map(x=>x/rawNorm[i]));
      const cosine=direction.reduce((sum,x,i)=>sum+x*other[i],0);
      const delta = direction.map(x=>value*x);
      const edit = original.map((coordinate, i) => coordinate + delta[i]);
      const vector = row => "(" + row.map(x=>scalar(x,3)).join(", ") + ")";
      return {...result(grid() + arrow(...xy(original), ...xy(original.map((x,i)=>x+other[i]))) + arrow(...xy(original), ...xy(edit))
        + point(xy(original), "z", "blue") + point(xy(edit), "z′", "aqua")
        + label(48, 34, "2D LATENT SLICE", "mute") + label(339, 314, "α = " + scalar(value, 1), "amber"),
        "The state moves to (" + edit.map(v => scalar(v)).join(", ") + "). Edit norm " + scalar(Math.abs(value),2) + " · direction cosine " + scalar(cosine,3) + ".",
        "Both directions have unit norm after the same affine projection. The second arrow shows the other unit direction for comparison."),
        table:{caption:"Affine transform and normalization of the fixed basis",
          columns:["Basis", "Affine output", "Raw norm", "Unit direction"],
          rows:[direction,other].map((unit,i)=>[vector(basis[i]),vector(raw[i]),scalar(rawNorm[i],3),vector(unit)])},
        notes:[["Signed edit", vector(original) + " + " + vector(delta) + " = " + vector(edit) + " (rounded). The step scales the first unit direction."],
          ["Cross-talk in latent space", "The signed edit projects " + scalar(value*cosine,3)
            + " units onto the second normalized direction. Unit normalization permits correlated directions."]],
      };
    }
  },
  gfn: {
    title: "Latent GFlowNet", status: implemented,
    description: "Score a fixed path with two-dimensional Gaussian forward and reverse kernels. A noninitial move includes continuation probability. The final exit contributes stop probability once, and the reverse step to the origin is deterministic.",
    formula: "residual = log Z + Σ(log pᶠ + log pcontinue − log pᵇ) + log pstop − log R",
    source: code + "reasoning/gfn_sampling.py", sourceLabel: "Path scoring source",
    control: {label: "Terminal log reward", min: -2, max: 2, step: 0.1, value: 0.5,
      secondaryLabel: "Terminal step", choices:["Stop at step 2", "Forced stop at horizon 3", "Stop at step 1"]},
    render(reward, choice) {
      const model=trajectory(reward,[2,3,1][choice]);
      return {...result(grid() + model.states.slice(1).map((state,i) => arrow(...xy(model.states[i]),...xy(state))).join("")
        + model.states.map((state,i) => point(xy(state),"z"+i,i===model.states.length-1?"aqua":"blue")).join("")
        + label(48, 34, "LAYERED LATENT PATH", "mute")
        + label(48, 315, "STOP LOG MASS " + scalar(model.stop) + "   LOG Z " + scalar(model.logZ), "aqua"),
        "Residual " + scalar(model.balance) + " yields squared loss " + scalar(model.balance ** 2, 3) + ".",
        "Densities describe continuous latent moves. Stop and continue are probability masses, with forced exit at the horizon."),
        table:{caption:"Log terms in trajectory balance",columns:["Move","Forward density","Continue mass","Reverse density or mass","Signed contribution"],
          rows:model.terms.map((term,i)=>[String(i+1),...["forward","continuation","backward","contribution"].map(key=>scalar(term[key],3))])},
        notes:[["Balance", "log Z (" + scalar(model.logZ,3) + ") + path sum (" + scalar(model.pathSum,3)
          + ") + log stop (" + scalar(model.stop,3) + ") − log reward (" + scalar(reward,3) + ") = " + scalar(model.balance,3) + " (rounded)."],
          ["Exit accounting", model.stop === 0 ? "Horizon exit is forced. Its log stop mass is zero." : "Optional exit contributes log stop " + scalar(model.stop,3) + " once."],
          ["Origin reverse", "Move 1 reverses to a Dirac origin, so its reverse log mass is zero."],
          ["Balance ratio", "exp(residual) = " + scalar(Math.exp(model.balance),3)
            + ". A ratio of 1 balances this path. A single fixture path cannot establish the terminal distribution."]],
      };
    }
  },
  forest: {
    title: "Forest-of-thought", status: implemented,
    description: "Two trees keep independent beams. At each layer, filter invalid candidates, sort by score, then greedily enforce minimum separation. A retained terminal leaf is recorded and does not expand.",
    formula: "per tree, per layer: valid → stable score order → distance guard → beam",
    source: code + "reasoning/fot_forest.py", sourceLabel: "Bounded forest source",
    control: {label: "Minimum latent separation", min: 0, max: 1.4, step: 0.05, value: 0.45,
      secondaryLabel: "Per-tree candidate budget", choices: ["9 proposals", "5 proposals", "3 proposals"]},
    render(threshold, budgetChoice) {
      const budget=[9,5,3][budgetChoice];
      const panels=[[74,122],[74,245]].map(([x,y],tree) => {
        const root={id:"R"+(tree+1),xy:[x,y]};
        const nodes=[];
        let active=[root], reason="depth";
        const leaves=[];
        for (let depth=1;depth<=2;depth++) {
          const layer=[];
          for (const [j,parent] of active.entries()) {
            const count=Math.min(3,budget-nodes.length-layer.length);
            for (let i=0;i<count;i++) {
              const state=[depth*1.3+(i%2)*.2, j*.6+i*.27];
              layer.push({id:parent.id+"."+i, parent, state, depth, xy:[x+depth*152,y-32+j*16+i*20],
                score:[.85,.96,.72][i]-.04*j-.03*depth, valid:i!==1 || tree===0 || depth===2,
                terminal:i===0});
            }
          }
          const selected=[];
          layer.filter(item=>!item.valid).forEach(item=>{ item.disposition="Invalid"; });
          for (const item of layer.filter(item=>item.valid).sort((a,b)=>b.score-a.score)) {
            if (selected.length===2) {
              item.disposition="Beam full";
              continue;
            }
            item.separation=Math.min(...selected.map(other=>Math.hypot(...item.state.map((v,j)=>v-other.state[j]))));
            if (item.separation<threshold) {
              item.disposition="Too close";
              continue;
            }
            item.selected=true;
            item.disposition=item.terminal?"Retained terminal":"Retained";
            selected.push(item);
          }
          nodes.push(...layer);
          leaves.push(...selected.filter(item=>item.terminal));
          active=selected.filter(item=>!item.terminal);
          if (!selected.length) {reason="no valid selections"; break;}
          if (!active.length) {reason="terminal"; break;}
          if (depth===2) {leaves.push(...active); break;}
          if (nodes.length===budget) {reason="budget"; leaves.push(...active); break;}
        }
        return {root,nodes,reason,leaves};
      });
      const branches=panels.map(({root,nodes}) => point(root.xy,root.id,"blue") + nodes.map(item=>{
        const tint=!item.valid?"amber":item.selected?"aqua":"mute";
        return line(...item.parent.xy,...item.xy,item.selected?"stroke":"grid-line")
          + dot(...item.xy,tint,item.terminal?7:4) + label(item.xy[0]+10,item.xy[1]+4,scalar(item.score),tint);
      }).join("")).join("");
      const all=panels.flatMap(panel=>panel.nodes);
      return {...result(label(43,35,"ROOTS","mute") + label(198,35,"DEPTH 1","mute") + label(347,35,"DEPTH 2","mute") + branches,
        all.filter(item=>item.selected).length + " of " + all.length + " proposals retained across both trees.",
        "Green marks retained branches. Amber marks invalid proposals. Larger markers are terminal. The separation test compares with proposals already retained at the same depth in the same tree."),
        table:{caption:"Candidate audit within each independent tree",
          columns:["Candidate", "Depth", "Score", "Nearest retained", "Decision"],
          rows:all.map(item=>[item.id,item.depth,scalar(item.score),Number.isFinite(item.separation)?scalar(item.separation,3):"—",item.disposition])},
        notes:[["Budget order", "Roots do not spend proposals. If a depth runs out of budget, earlier retained parents get the remaining expansions first."],
          ...panels.map(({root,nodes,leaves,reason})=>[root.id, nodes.length + "/" + budget + " proposals · stop by " + reason
          + " · terminal leaves " + (leaves.filter(item=>item.terminal).map(item=>item.id).join(", ") || "none")
          + " · unexpanded frontier " + (leaves.filter(item=>!item.terminal).map(item=>item.id).join(", ") || "none") + "."])],
      };
    }
  },
  analogy: {
    title: "Transfer a typed relation", status: intended,
    description: "Compare paired before and after records from independent reference cultures for a junction-opening pulse. Check whether the assay contexts match and the four-dimensional displacement directions agree. Match donor, junction and receiver by their roles, then verify directed edge types before ranking proposals. Candidate outcomes stay masked throughout.",
    formula: "dᵢ = z(ref after) − z(ref before); q = z(query before) + β mean(dᵢ) when the reference gate passes",
    source: code + "reasoning/fot_selection.py", sourceLabel: "Implemented eligibility selection (analogy gates here are synthetic)",
    control: {label: "Reference displacement scale β", min: -1, max: 1.5, step: 0.05, value: 1,
      secondaryLabel:"Reference pairs", choices: ["Early-stage reference set", "Matched pair 1", "Matched pair 2", "Later-stage pair 3"],
      scenarioLabel:"Counterfactual scenario", scenarios:["Original matched query", "Later receiver cell state", "Swap donor and receiver", "Discordant reference cultures", "Close eligible proposals"]},
    render(scale, referenceMode, scenario) {
      const m = analogy(scale,referenceMode,scenario);
      const project = ([x,y]) => [56 + 138*x, 286 - 72*y];
      const marks = m.pairs.map(pair => {
        const a=project(pair.before), b=project(pair.after);
        return arrow(...a,...b) + point(a, pair.id+" pre", "blue") + point(b, pair.id+" post", "blue");
      }).join("");
      const target = project(m.query), start = project(m.queryBefore);
      const envelope = m.envelope;
      const robust = envelope?.robust;
      const bounds = envelope?.bounds;
      const rectangle = bounds ? (() => {
        const [x1,y1]=project([bounds[0][0],bounds[1][1]]);
        const [x2,y2]=project([bounds[0][1],bounds[1][0]]);
        return box(x1,y1,x2-x1,y2-y1,robust ? "lab-envelope stable" : "lab-envelope");
      })() : "";
      const offsets={A:[-36,-24], B:[-19,30], C:[25,-17], D:[19,33], E:[15,20], F:[-43,11]};
      const marksById = m.ranked.map(candidate => {
        const [x,y]=project(candidate.state);
        const style=candidate === m.selected && robust !== false ? "aqua" : candidate.refusal ? "amber" : "mute";
        const [dx,dy]=offsets[candidate.id];
        return line(x,y,x+dx,y+dy,"grid-line") + dot(x,y,style,candidate === m.selected ? 7 : 4)
          + label(x+dx,y+dy,candidate.id,style);
      }).join("");
      const summary = envelope && !robust
        ? "Abstain under reference sensitivity · mean-transfer nearest eligible " + m.selected.id + "."
        : m.selected
        ? "2D nearest " + m.projected.id + " · 4D nearest " + m.ranked[0].id + " · eligible " + m.selected.id + " · no-transfer baseline " + m.baseline.id + "."
        : "Abstain · " + (m.transferReason || "no eligible proposal") + ". Candidate outcomes remain masked.";
      const example = m.selected || m.ranked.find(candidate => candidate.id === "C");
      const walk = edges => edges.map(([i,j,type]) => i + " → " + j + " (" + type + ")").join(", ");
      const pairAudit = m.agreement.map(({references,cosine}) => references.join(" / ") + " "
        + (Number.isFinite(cosine) ? scalar(cosine,3) : "undefined")).join(" · ");
      return {...result(grid() + rectangle + label(44, 28, "DISPLAY COORDINATES 1–2 OF 4", "mute") + marks
        + (scale !== 0 && m.displacement ? arrow(...start,...target) : "")
        + point(start,"query pre","blue") + marksById + '<circle cx="'+target[0]+'" cy="'+target[1]+'" r="10" class="target-ring"/>'
        + label(target[0]+13,target[1]-14,m.displacement ? "q" : "q₀","amber") + label(44, 316, "Blue refs · Green pick · Gold refused · Box spread", "mute"),
        summary,"Fixture vectors illustrate retrieval. Reference outcomes support the displacement. Candidate descriptors contain only visible proposal inputs. A retrieval result needs independent outcome validation."),
        diagram:motifDiagram(example),
        notes:[
          ["Query relation", m.querySpec.relation + " · " + m.querySpec.stage + " · " + m.querySpec.assay + " (" + m.querySpec.unit + ") · " + m.querySpec.protocol],
          ["Reference audit", m.referenceAudit.map(r=>r.id + " (" + r.group + ") · " + (r.disposition === "Used" ? "Compared" : r.disposition)).join(" · ")],
          ["Typed correspondence", "Source edges " + walk(sourceMotif.edges) + ". Candidate " + example.id + " edges "
            + walk(example.targetMotif.edges) + ". Role mapping " + example.motif.mapping.map((j,i)=>i + "→" + j).join(" · ")
            + " preserves both directed edge types. A reverses the edges. B rewires the conducting edge from the donor."],
          ["Shared representation", "All z vectors are manually assigned four-dimensional fixtures. Native routes share a TokenGT encoder. Cross-modal calibration would require a separate evaluation."],
          ["Direction agreement", pairAudit ? "4D pairwise cosines " + pairAudit + ". This fixture abstains if any cosine is undefined or ≤ 0."
            : m.pairs.length ? "One compatible pair supports a provisional transfer. Checking agreement requires a second culture."
            : "Checking agreement requires compatible reference pairs."],
          ["Matched displacement", m.displacement ? "Δ = (" + m.displacement.map(v=>scalar(v)).join(", ") + ")" : "No supported transfer direction."],
          ["Counterfactual", "β = 0 removes the transfer. Negative β reverses it. Switching to a later cell state invalidates the early pairs. Selecting R3 then tests recovery with matching evidence."],
          ["Reference sensitivity", m.sensitivity.length ? m.sensitivity.map(row => "Omit " + row.omitted + " → "
            + (row.selected ? row.selected + " (distance " + scalar(row.distance,3)
              + (row.margin === null ? "" : ", margin " + scalar(row.margin,3)) + ")" : "abstain")).join(" · ")
            : "No compatible reference subset to test."],
          ["Retrieval margin", m.margin === null ? "A runner-up margin requires two eligible proposals and a supported transfer."
            : "Runner-up minus selected distance = " + scalar(m.margin,3) + ". This measures geometric separation. Confidence calibration needs independent outcome data."],
          ["Reference spread", envelope
            ? "Projected box uses coordinate-wise minima and maxima of the compatible single-culture transfers. The decision uses all four dimensions. "
              + (robust ? "The same proposal stays nearest throughout the box." : "At least one eligible rival can tie or outrank the mean-transfer proposal within the box.")
            : "At least two compatible reference cultures and one eligible proposal are needed for the spread check."],
          ["Worst-case gap", envelope?.rivals.length
            ? envelope.rivals.map(rival=>rival.id + " vs " + m.selected.id + " · [" + scalar(rival.minGap,3)
              + ", " + scalar(rival.maxGap,3) + "]").join(" · ")
              + " in squared-distance units. The box allows combinations of reference coordinates never observed together."
            : "A gap interval requires an eligible rival. The spread check is a sensitivity analysis."],
          ["Projection check", m.projectedEligible ? "Among eligible proposals, the 2D view prefers " + m.projectedEligible.id + ". The selection rule uses all four coordinates." : "No eligible 2D comparison."]
        ],
        table: {
          caption: "Candidate audit in 4D distance order · " + (m.displacement ? "transferred q" : "unshifted q₀") + " (" + m.query.map(value=>scalar(value)).join(", ") + ")",
          columns: ["Candidate", "2D distance", "4D distance", "Squared residuals · four axes", "Motif edges", "Gate", "Held-out outcome"],
          rows: m.ranked.map(item => [item.id, scalar(item.projected,3), scalar(item.full,3), item.squared.map(v=>scalar(v,4)).join(" + "), item.motif.matched + "/2",
            item.refusal || (item === m.selected ? (robust === false ? "Nearest at mean · unstable" : "Selected") : "Passes proposal gates"), "Masked"]),
        }};
    }
  },
  topology: {
    title: "Topological analogy check", status: intended,
    description: "The research plan studies whether a mapping between latent neighborhoods preserves connectivity and loops across scales. This Vietoris–Rips example compares two four-point neighborhoods at the selected radius and shows the scale interval where each has an unfilled loop.",
    formula: "include edge (i, j) when ‖zᵢ − zⱼ‖₂ ≤ ε; fill every clique",
    source: "architecture.html", sourceLabel: "Reasoning design and evidence",
    control: {label: "Filtration radius ε", min: 0.6, max: 2.2, step: 0.05, value: 1.15,
      secondaryLabel:"Proposed correspondence", choices:["Deformed square · identity mapping", "Collapsed line · identity mapping", "Same square · wrong mapping"]},
    render(radius, scenario) {
      const comparison=topologyComparison(radius,scenario);
      const neighborhoods = [comparison.source,comparison.target];
      const panels = neighborhoods.map((states, group) => {
        const position = states.map(([x, y]) => [70 + group * 255 + 116 * x, 229 - 120 * y]);
        const {edges,faces,components,loops}=ripsAt(states,radius);
        const strokes=edges.map(([i,j])=>line(...position[i],...position[j],
          group && comparison.extra.includes(i + ":" + j) ? "lab-extra-edge" : "stroke")).join("");
        const missing=group ? comparison.missing.map(pair => {
          const [i,j]=pair.split(":").map(Number);
          return line(...position[i],...position[j],"lab-missing-edge");
        }).join("") : "";
        const fills=faces.map(face=>'<polygon class="lab-face" points="'
          + face.map(index=>position[index].join(",")).join(" ") + '"/>').join("");
        const intervals=ripsLoopIntervals(states).map(([birth,death])=>"[" + scalar(birth,3) + ", " + scalar(death,3) + ")");
        const textX = 56 + 255 * group;
        return {
          figure: label(textX, 58, "NEIGHBORHOOD " + (group ? "B" : "A"), "mute") + fills + strokes + missing
            + position.map((p,i) => dot(...p) + label(p[0]+9,p[1]-9,String(i),"blue")).join("") + label(textX, 284, "β₀ " + components + "    β₁ " + loops, "aqua"),
          intervals, summary: " " + (group ? "B" : "A") + " has " + components + " component(s) and " + loops + " loop(s)."
        };
      });
      const legend = comparison.preservesEdges ? "" : line(52, 313, 79, 313, "lab-missing-edge")
        + label(90, 318, "Missing in B", "ink") + line(298, 313, 325, 313, "lab-extra-edge")
        + label(336, 318, "Extra in B", "ink");
      return {...result(line(278, 47, 278, 291, "grid-line") + panels.map(p => p.figure).join("") + legend,
        "At radius " + scalar(radius) + "," + panels.map(p => p.summary).join("") + (comparison.preservesEdges ? " Supplied map preserves this edge set." : " REFUSE supplied map: edge mismatch."),
        "β₀ counts components. β₁ counts independent unfilled loops. A loop dies when filling triangles remove the independent cycle. Dashed overlays mark missing edges and are excluded from the counts."),
        table:{caption:"All six distances under the supplied map", columns:["Source pair", "Source distance", "Mapped target distance", "Absolute error"], rows:comparison.distances.map(d=>[d.pair,scalar(d.left,3),scalar(d.right,3),scalar(d.error,3)])},
        notes:[["Supplied map", comparison.mapping.map((j,i)=>i+"→"+j).join(" · ")],
          ["Correspondence audit", "Equal Betti counts: " + comparison.sameBetti + "; missing mapped edges: " + (comparison.missing.join(", ") || "none") + "; extra target edges: " + (comparison.extra.join(", ") || "none") + ". Counts alone cannot authenticate a correspondence."],
          ["Metric distortion", "max |d(source pair) − d(mapped target pair)| = " + scalar(comparison.distortion,3) + ". No learned acceptance threshold is implied."],
          ["Scope", "Four-point clique complexes only. Undirected proximity edges are not causal, chemical, or typed semantic edges. Even a passing map is no proof of a biological analogy."],
          ...panels.map((p,i)=>["Neighborhood " + (i===0 ? "A" : "B") + " loop interval",
          p.intervals.join(" · ") || "No unfilled loop across the filtration."])],
      };
    }
  },
  ensemble: {
    title: "Five-stage Boltzmann target", status: "Implemented reward equation · synthetic ensemble",
    description: "Keep the molecular system fixed while cooling. Select a teacher-specific toy energy vector, then compute log rewards and normalize the mass over three illustrative conformational states. Each state has an explicit base weight.",
    formula: "log Rᵢ = −(Eᵢ − Eref)/(kᴮT);  pᵢ ∝ wᵢ exp(log Rᵢ)",
    source: "https://github.com/amelie-iska/Mora-3/blob/f74ff02d88bb9e12b02e835b541c27486298f580/src/mora3/boltzmann_target.py", sourceLabel: "Native Boltzmann log reward",
    control: {label: "Cooling stage", options: coolingTemperatures.map((t,i) => "Stage " + (i+1) + " · " + t + " K"),
      secondaryLabel: "Independent energy illustration", choices: energyExamples.map(example => example.name),
      scenarioLabel: "Relative base weights", scenarios: ["Equal weight", "Third basin has 4× base weight"]},
    render(stage, teacher, measure) {
      const example = energyExamples[teacher];
      const temperature = coolingTemperatures[stage];
      const base = measure === 0 ? [1,1,1] : [1,1,4];
      const {reference, kT, logReward, logMass, logNormalizer, probabilities} = boltzmannStates(example.energies, base, temperature);
      const pct = value => scalar(100 * value,1) + "%";
      const colors = ["aqua","blue","amber"];
      const plot = coolingTemperatures.map((t,i) => {
        const values = boltzmannStates(example.energies,base,t).probabilities;
        const x = 45 + i*102;
        let top = 261;
        return values.map((value,j) => {
          const height = value*182;
          top -= height;
          return box(x,top,54,height,colors[j]);
        }).join("") + label(x,286,t + " K",i===stage?"aqua":"mute")
          + (i===stage ? line(x,299,x+54,299,"stroke") : "");
      }).join("");
      const gap = stage ? coolingTemperatures[stage-1]-temperature : null;
      const entropy = -probabilities.reduce((total,p) => total + (p ? p*Math.log(p) : 0),0);
      const meanEnergy = probabilities.reduce((total,p,i) => total + p*example.energies[i],0);
      return {...result(label(42,36,"NORMALIZED PROBABILITY BY COOLING STAGE","mute") + plot
        + label(42,324,"Green A   Blue B   Gold C","mute"),
        "Stage " + (stage+1) + " · " + temperature + " K" + (gap === null ? "" : " · drop " + gap + " K")
          + ". State masses " + probabilities.map(pct).join(" / ") + ".",
        "The bars show a three-state approximation with invented energies. The same reward equation is used for the continuous coordinate target. Sampling steps do not specify physical elapsed time."),
        table:{caption:"Energy and target mass at the selected stage",
          columns:["State","Energy (eV)","Relative base weight","log reward","log mass","Probability"],
          rows:example.energies.map((energy,i)=>[["A","B","C"][i],scalar(energy,3),String(base[i]),scalar(logReward[i],3),scalar(logMass[i],3),pct(probabilities[i])])},
        notes:[["Energy scale", "Eref = " + scalar(reference,3) + " eV and kᴮT = " + scalar(kT,4)
          + " eV at this stage. A shared change to Eref shifts all log rewards equally and leaves probabilities unchanged."],
          ["Normalization", "The relative base weights sum to 1 after normalization. Log mass adds the log of that normalized base weight to the log reward. The log-sum-exp of the three log masses is " + scalar(logNormalizer,3) + "."],
          ["C to A odds", "Base-weight ratio " + scalar(base[2]/base[0],3) + " × energy factor "
            + scalar(Math.exp(logReward[2]-logReward[0]),3) + " gives p(C)/p(A) = " + scalar(probabilities[2]/probabilities[0],3) + " (rounded)."],
          ["Ensemble statistics", "The three-state Shannon entropy is " + scalar(entropy,3)
            + " nats and the probability-weighted mean energy is " + scalar(meanEnergy,4) + " eV."],
          ["Teacher scope", "UMA and eSEN illustrate separate energy vectors. The browser runs no potential, DFT calculation, force evaluation, or kinetic integrator."]],
      };
    }
  },
  geometry: {
    title: "Current-frame structure and dynamics", status: "Diagnostic implementation · synthetic geometry",
    description: "Compute scalar coefficients from distance and bond features of a four-atom input. Multiply each relative displacement by its coefficient, average across neighbors including self, and zero the anchor velocity. The ghost frame shows one illustrative Euler step.",
    formula: "vᵢ = freeᵢ · (1/N) Σⱼ wᵀfᵢⱼ(xⱼ − xᵢ);  x′ = x + 0.4v",
    source: code + "model/current_frame_velocity.py", sourceLabel: "Coordinate-velocity readout source",
    control: {label: "Bond feature coefficient", min: -0.8, max: 1.6, step: 0.05, value: 0.9,
      secondaryLabel:"Current frame", choices:["Original coordinates", "Rotate and translate frame"],
      scenarioLabel:"Atom to inspect", scenarios:["Atom 2 · free", "Atom 1 · anchor", "Atom 3 · free", "Atom 4 · free"]},
    render(coupling, frame, atomChoice) {
      const {original,transformed,velocityError,distanceError} = frameComparison(coupling);
      const {xyz,free,pairs,velocities,next,distance} = frame ? transformed : original;
      const atom = [1,0,2,3][atomChoice];
      const vector = values => values.map(value=>scalar(value,3)).join(", ");
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
            + dot(...from, free[i] ? "aqua" : "blue", i === atom ? 13 : 9) + label(from[0]-5, from[1]+32, String(i+1));
        }).join("") + label(42, 291, "PAIR 2–3 AFTER STEP  " + scalar(distance,3) + " Å", "mute"),
        "Atom 1 is fixed. Atom 2 has speed " + scalar(Math.hypot(...velocities[1]),3) + " Å per toy step unit.",
        "Arrows magnify the velocity and rings show the Euler update. The anchor still supplies context to its neighbors. The frame selector applies a quarter-turn about x and a translation. Time uses an illustrative scale."),
        table:{caption:"Pair arithmetic for selected atom " + (atom+1),
          columns:["Neighbor", "Distance (Å)", "Bond", "Scalar coefficient", "Relative xyz (Å)", "Weighted xyz"],
          rows:pairs[atom].map((pair,i)=>[String(i+1),scalar(pair.distance,3),pair.bonded?"Yes":"No",
            scalar(pair.coefficient,3),vector(pair.relative),vector(pair.weighted)])},
        notes:[["Velocity and Euler update", "Selected atom " + (atom+1) + " · " + (free[atom] ? "free" : "fixed")
          + " · velocity (" + vector(velocities[atom]) + "). Current coordinates (" + vector(xyz[atom])
          + ") become (" + vector(next[atom]) + "). The self-pair has a zero displacement and is included in the mean."],
          ["Rigid-frame check", "Recomputing in the transformed frame gives max ‖v′ − Rv‖ = "
          + velocityError.toExponential(2) + ". The maximum change across all six updated pair distances is "
          + distanceError.toExponential(2) + " Å."]],
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
    secondaryLabel.textContent = control.secondaryLabel;
    control.choices.forEach((choice, i) => secondary.add(new Option(choice, i)));
    holder.append(secondaryLabel, secondary);
  }
  const scenario = document.createElement("select");
  if (control.scenarios) {
    const scenarioLabel=document.createElement("label");
    scenario.id="lab-scenario";
    scenarioLabel.htmlFor=scenario.id;
    scenarioLabel.textContent=control.scenarioLabel;
    control.scenarios.forEach((choice,i)=>scenario.add(new Option(choice,i)));
    holder.append(scenarioLabel,scenario);
  }
  ui.control.replaceChildren(holder);
  const update = () => {
    output.value = picker.value;
    const display = method.render(Number(picker.value), Number(secondary.value), Number(scenario.value));
    ui.plot.innerHTML = '<defs><marker id="lab-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6a91a4"/></marker></defs>' + display.figure;
    ui.plot.setAttribute("aria-label", method.title + ". " + display.summary);
    ui.result.textContent = display.summary;
    ui.caption.textContent = display.caption;
    ui.evidence.replaceChildren();
    if (display.diagram) ui.evidence.insertAdjacentHTML("beforeend", display.diagram);
    if (display.notes) {
      const notes=document.createElement("dl");
      notes.className="lab-audit-notes";
      for (const [heading,body] of display.notes) {
        const term=document.createElement("dt"), explanation=document.createElement("dd");
        term.textContent=heading;
        explanation.textContent=body;
        notes.append(term,explanation);
      }
      ui.evidence.append(notes);
    }
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
  scenario.addEventListener("change", update);
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
