import {weights, makeGraph, runTrace} from "./layer-trace-engine.js";

const names = ["input","strength","strength-value","token","head","channel","steps","progress","prev","next",
  "title","explanation","equation","numerical","tensors"];
const ui = Object.fromEntries(names.map(name => [name, document.getElementById("trace-" + name)]));
const fmt = value => value === -Infinity ? "−∞" : typeof value === "number"
  ? (Math.abs(value) < .00005 ? "0.0000" : value.toFixed(4)) : value;
const showRow = row => "[" + row.map(fmt).join(", ") + "]";
const columns = count => Array.from({length: count}, (_, index) => "c" + (index + 1));
function node(tag, content) {
  const item = document.createElement(tag);
  if (content !== undefined) item.textContent = content;
  return item;
}
function matrix(title, values, {rows = [], cols = columns(values[0].length), selected = -1, channel = -1,
  winner = -1, weight = false} = {}) {
  const figure = node("figure");
  figure.className = "trace-matrix" + (weight ? " is-weights" : "");
  figure.append(node("figcaption", title + " · " + values.length + " × " + values[0].length));
  const viewport = node("div");
  viewport.className = "trace-matrix-scroll";
  viewport.tabIndex = 0;
  viewport.setAttribute("aria-label", title);
  const table = node("table");
  const head = table.createTHead().insertRow();
  head.append(node("th", "Row"));
  cols.forEach(label => {
    const th = node("th", label);
    th.scope = "col";
    head.append(th);
  });
  const body = table.createTBody();
  values.forEach((row, i) => {
    const tr = body.insertRow();
    if (i === selected) tr.className = "is-selected";
    if (i === winner) tr.classList.add("is-winner");
    const th = node("th", rows[i] ?? String(i + 1));
    th.scope = "row";
    tr.append(th);
    row.forEach((value, j) => {
      const td = tr.insertCell();
      if (j === channel) td.className = "is-channel";
      td.textContent = fmt(value);
    });
  });
  viewport.append(table);
  figure.append(viewport);
  return figure;
}

const steps = [
  {title:"Graph and token embedding", text:"TokenGT builds a row for every node and directed edge. Node endpoints repeat the same ID. Each row adds a feature projection, two endpoint-ID projections packed into one matrix, and a learned node/edge role. The graph token has its own vector. Padding fills the batch shape.",
    equation:"xᵢ = Wfeature fᵢ + Wincidence [p(srcᵢ); p(dstᵢ)] + eroleᵢ", render(c) {
      const {graph, trace: t, query, tokenRows} = c;
      const rows = graph.tokens.slice(1,-1);
      const chosen = query === 0 ? "The first row is the independent learned graph token."
        : query === graph.tokens.length - 1 ? "The initial padding row is zero and its key is masked."
        : "Selected row = " + showRow(t.features[query-1]) + " + " + showRow(t.incidences[query-1])
          + " + " + showRow(t.roles[query-1]) + " = " + showRow(t.input[query]) + ".";
      return [chosen,
        matrix("Graph rows and valid roles", graph.endpoints.map(([a,b],i) => [a,b,graph.roles[i],...graph.features[i]]),
          {rows,cols:["source ID","target ID","role ID","feature 1","feature 2"],selected:query-1}),
        matrix("Orthonormal node identifiers",graph.identifiers,{rows:graph.tokens.slice(1,1+graph.identifiers.length)}),
        matrix("Feature projection weights",weights.feature,{weight:true}),
        matrix("Incidence projection weights",weights.incidence,{weight:true}),
        matrix("Role embeddings",weights.role,{rows:["node","edge"],weight:true}),
        matrix("Packed endpoint IDs",t.incidence,{rows,selected:query-1}),
        matrix("Feature contributions",t.features,{rows,selected:query-1}),
        matrix("Incidence contributions",t.incidences,{rows,selected:query-1}),
        matrix("Role contributions",t.roles,{rows,selected:query-1}),
        matrix("Transformer input",t.input,tokenRows)];
    }},
];

const layerSteps = [
  {title:"Pre-attention LayerNorm", text:"Normalize each token over its four channels, then apply learned gain and offset. The example uses population variance and ε = 10⁻⁵.",
    equation:"nᵢ = γ ⊙ (xᵢ − μᵢ)/√(σ²ᵢ + ε) + β",render(c) {
      return normView(c,c.layer.input,c.layer.firstNorm,c.parameters.norm1);
    }},
  {title:"Finite transform and head split", text:"Apply ReLU and log1p to the normalized values. Split the four channels into two head-local vectors. This finite transform is the implemented code path.",
    equation:"z = reshape_heads(log(1 + max(n, 0)))",render(c) {
      return ["The selected head sees " + showRow(c.head.z[c.query]) + ".",
        matrix("Normalized rows",c.layer.firstNorm.values,c.tokenRows),
        matrix("All transformed channels",c.layer.attention.logValues,c.tokenRows),
        matrix("Selected head input",c.head.z,c.tokenRows)];
    }},
  {title:"Max-plus queries, keys, and values", text:"Each head applies the same three learned head-local matrices to its own slice. For each output coordinate, add each input coordinate to its projection weight and take the maximum.",
    equation:"qᵢⱼ = maxᵣ(zᵢᵣ + WQⱼᵣ), with the same operation for K and V",render(c) {
      const terms=c.head.z[c.query].map((value,i)=>value+c.parameters.query[c.channel][i]);
      return ["Selected query channel = max " + showRow(terms) + " = " + fmt(Math.max(...terms)) + ".",
        matrix("Query weights shared by heads",c.parameters.query,{weight:true,selected:c.channel}),
        matrix("Key weights shared by heads",c.parameters.key,{weight:true}),
        matrix("Value weights shared by heads",c.parameters.value,{weight:true}),
        matrix("Queries",c.head.q,c.tokenRows),matrix("Keys",c.head.k,c.tokenRows),matrix("Values",c.head.v,c.tokenRows)];
    }},
  {title:"Hilbert scores and key mask", text:"Subtract each key vector from the selected query. Negate the range across head-local channels. The padding key receives negative infinity, so it cannot win the reduction.",
    equation:"sᵢⱼ = minᵣ(qᵢᵣ − kⱼᵣ) − maxᵣ(qᵢᵣ − kⱼᵣ)",render(c) {
      const terms=c.head.differences[c.query].map((diff,i)=>[...diff,Math.min(...diff),Math.max(...diff),c.graph.valid[i]?"valid":"masked",c.head.scores[c.query][i]]);
      return ["Query " + c.graph.tokens[c.query] + " = " + showRow(c.head.q[c.query]) + ".",
        matrix("Selected query versus each key",terms,{rows:c.graph.tokens,cols:["difference 1","difference 2","min","max","key mask","score"]}),
        matrix("All scores for the selected head",c.head.scores,{rows:c.graph.tokens,cols:c.graph.tokens,selected:c.query})];
    }},
  {title:"Max-plus context per channel", text:"Add the query–key score to each value channel. A channel selects its own maximizing key. In a tie, the highlighted key is the first maximizer in token order.",
    equation:"cᵢᵣ = maxⱼ(sᵢⱼ + vⱼᵣ)",render(c) {
      const candidates=c.head.candidates[c.query], max=c.head.context[c.query][c.channel];
      const winner=candidates.findIndex(row=>row[c.channel]===max);
      return ["Channel " + (c.channel+1) + " selects " + c.graph.tokens[winner] + " at " + fmt(max) + ".",
        matrix("Score-plus-value candidates",candidates,{rows:c.graph.tokens,channel:c.channel,winner}),
        matrix("Reduced context",c.head.context,c.tokenRows)];
    }},
  {title:"Join heads, decode, project, and add", text:"Concatenate the two contexts in head order, then apply expm1 and a bias-free output projection. Add the result to the layer input on the residual path.",
    equation:"a = Wo expm1(concat(c¹,c²)); r = x + a",render(c) {
      const l=c.layer;
      return ["Selected row = " + showRow(l.input[c.query]) + " + " + showRow(l.attention.output[c.query]) + " = " + showRow(l.residual[c.query]) + ".",
        ...l.attention.heads.map((head,i)=>matrix("Head "+(i+1)+" context",head.context,c.tokenRows)),
        matrix("Concatenated heads",l.attention.concatenated,c.tokenRows),
        matrix("Decoded channels",l.attention.decoded,c.tokenRows),
        matrix("Output projection weights",c.parameters.output,{weight:true}),
        matrix("Attention output",l.attention.output,c.tokenRows),
        matrix("Post-attention residual",l.residual,c.tokenRows)];
    }},
  {title:"Pre-FFN LayerNorm", text:"A second learned LayerNorm operates on the post-attention residual before the feed-forward network.",
    equation:"n₂ = LayerNorm₂(r)",render(c) {
      return normView(c,c.layer.residual,c.layer.secondNorm,c.parameters.norm2);
    }},
  {title:"GELU feed-forward and residual", text:"Project four channels to six hidden features, apply the exact GELU, then project back to four channels. Add the previous residual row. This output becomes the next layer input.",
    equation:"x′ = r + W₂ GELU(W₁ n₂ + b₁) + b₂",render(c) {
      const l=c.layer, p=c.parameters;
      return ["Selected layer output " + showRow(l.output[c.query]) + ".",
        matrix("Expansion weights and bias",p.ffnIn.map((w,i)=>[...w,p.ffnInBias[i]]),{weight:true,cols:[...columns(4),"bias"]}),
        matrix("Expanded activations",l.ffnPre,c.tokenRows),matrix("After GELU",l.ffnActive,c.tokenRows),
        matrix("Readback weights and bias",p.ffnOut.map((w,i)=>[...w,p.ffnOutBias[i]]),{weight:true,cols:[...columns(6),"bias"]}),
        matrix("Feed-forward output",l.ffnOutput,c.tokenRows),matrix("Layer output",l.output,c.tokenRows)];
    }},
];

function normView(c, input, norm, params) {
  const [mean,variance]=norm.stats[c.query];
  return ["Selected row mean " + fmt(mean) + " and variance " + fmt(variance) + ".",
    matrix("Input",input,c.tokenRows),
    matrix("Per-token statistics",norm.stats,{...c.tokenRows,cols:["mean","variance"]}),
    matrix("Gain and offset",[params.gamma,params.beta],{rows:["gain γ","offset β"],weight:true}),
    matrix("Normalized output",norm.values,c.tokenRows)];
}

for (let layer = 0; layer < 2; layer += 1) {
  steps.push(...layerSteps.map(step => ({...step, layer})));
}
steps.push({title:"Final normalization and padding mask",text:"Normalize the last layer output. Set the padded output row to zero. All valid token representations are available to downstream heads.",
  equation:"h = mask(LayerNormfinal(x₂))",render(c) {
    return [...normView(c,c.trace.layers[1].output,c.trace.final,weights.finalNorm),matrix("Final valid-token representation",c.trace.masked,c.tokenRows)];
  }});
steps.push({title:"Illustrative graph readout",text:"A demonstration linear readout selects the graph token. Actual Mora-3 output heads select and decode representations according to their own modality and task.",
  equation:"ŷ = w · h[GRAPH]",render(c) {
    const terms=c.trace.masked[0].map((value,i)=>value*weights.readout[i]);
    return ["Illustrative output = sum " + showRow(terms) + " = " + fmt(c.trace.readout) + ".",
      matrix("Graph representation",[c.trace.masked[0]],{rows:["[GRAPH]"]}),
      matrix("Demonstration readout weights",[weights.readout],{weight:true}),
      matrix("Readout contributions",[terms])];
  }});

let index = 0, graph, trace;
const buttons = steps.map((step, i) => {
  const button=node("button",String(i+1).padStart(2,"0")+" · "+(step.layer===undefined?"":"L"+(step.layer+1)+" ")+step.title);
  button.type="button";
  button.addEventListener("click",()=>select(i,true));
  return button;
});
ui.steps.replaceChildren(...buttons);

function render() {
  const step=steps[index], query=Number(ui.token.value), channel=Number(ui.channel.value);
  const title=(step.layer===undefined?"":"Layer "+(step.layer+1)+" · ")+step.title;
  const layer=trace.layers[step.layer];
  ui.title.textContent=title;
  ui.explanation.textContent=step.text;
  ui.equation.textContent=step.equation;
  ui.progress.textContent="Step "+(index+1)+" of "+steps.length;
  ui.prev.disabled=index===0;
  ui.next.disabled=index===steps.length-1;
  const tokenRows={rows:graph.tokens,selected:query};
  const [note,...figures]=step.render({graph,trace,query,channel,layer,tokenRows,
    parameters:weights.layers[step.layer],head:layer?.attention.heads[Number(ui.head.value)]});
  ui.numerical.textContent=note;
  ui.tensors.replaceChildren(...figures);
}
function select(i, focus=false) {
  index=i;
  buttons.forEach((button,j)=>{if(j===i)button.setAttribute("aria-current","step");else button.removeAttribute("aria-current");});
  render();
  if(focus)ui.title.focus({preventScroll:window.innerWidth>760});
}
function recompute() {
  const prior=Number(ui.token.value);
  const strength=Number(ui.strength.value);
  ui["strength-value"].value=strength.toFixed(2);
  graph=makeGraph(ui.input.value,strength);
  trace=runTrace(graph);
  ui.token.replaceChildren(...graph.tokens.map((token,i)=>new Option(token,i)));
  ui.token.value=Math.min(prior,graph.tokens.length-1);
  render();
}
ui.prev.addEventListener("click",()=>select(index-1,true));
ui.next.addEventListener("click",()=>select(index+1,true));
ui.input.addEventListener("change",recompute);
ui.strength.addEventListener("input",recompute);
[ui.token,ui.head,ui.channel].forEach(control=>control.addEventListener("change",render));
recompute();
select(0);
