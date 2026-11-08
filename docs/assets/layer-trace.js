import {tensorScene} from "./layer-trace-scene.js";
import {weights, graphPreset, graphFromInput, runTrace, cdf} from "./layer-trace-engine.js";

const names = ["input","strength","strength-value","token","head","channel","steps","progress","prev","next",
  "title","explanation","equation","numerical","tensors","map","key","reduction","json","apply","input-status","scene","angle","pitch",
  "kernel","multiplier","multiplier-value","block-setting","multiplier-setting","hidden-setting","hidden-channel"];
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
  winners = [], weight = false} = {}) {
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
    if (winners.includes(i)) tr.classList.add("is-winner");
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

function linearTerms(title, input, weight, bias, result) {
  const rows = input.map((x,i) => [x,weight[i],x*weight[i]]);
  const labels = columns(input.length);
  rows.push([0,bias,bias],[null,null,result]);
  return matrix(title,rows,{rows:[...labels,"bias","output"],cols:["input","weight","contribution"],selected:rows.length-1});
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
        ...(query>0 && graph.valid[query] ? [
          linearTerms("Selected feature projection",graph.features[query-1],weights.feature[c.global],0,t.features[query-1][c.global]),
          linearTerms("Selected endpoint projection",t.incidence[query-1],weights.incidence[c.global],0,t.incidences[query-1][c.global]),
        ] : []),
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
      const n = c.layer.firstNorm.values[c.query][c.global];
      return ["The selected head sees " + showRow(c.head.z[c.query]) + ".",
        matrix("Finite transform for the selected channel",[[n,Math.max(n,0),c.head.z[c.query][c.channel]]],
          {rows:[c.graph.tokens[c.query]],cols:["normalized","ReLU","log1p"]}),
        matrix("Normalized rows",c.layer.firstNorm.values,c.tokenRows),
        matrix("All transformed channels",c.layer.attention.logValues,c.tokenRows),
        matrix("Selected head input",c.head.z,c.tokenRows)];
    }},
  {title:"Max-plus queries, keys, and values", text:"Each head applies the same three learned head-local matrices to its own slice. For each output coordinate, add each input coordinate to its projection weight and take the maximum.",
    equation:"qᵢⱼ = maxᵣ(zᵢᵣ + WQⱼᵣ), with the same operation for K and V",render(c) {
      const terms=c.head.z[c.query].map((value,i)=>value+c.parameters.query[c.channel][i]);
      const ops = [
        ["Q at query", c.query, c.parameters.query],
        ["K at key", c.key, c.parameters.key],
        ["V at key", c.key, c.parameters.value],
      ];
      return ["Selected query channel = max " + showRow(terms) + " = " + fmt(Math.max(...terms)) + ".",
        matrix("Max-plus terms for selected channel", ops.map(([,row,w]) => {
          const terms = c.head.z[row].map((value, i) => value + w[c.channel][i]);
          return [...terms, Math.max(...terms)];
        }), {rows:ops.map(([name]) => name),cols:["z₁ + W₁","z₂ + W₂","maximum"]}),
        matrix("Query weights shared by heads",c.parameters.query,{weight:true,selected:c.channel}),
        matrix("Key weights shared by heads",c.parameters.key,{weight:true}),
        matrix("Value weights shared by heads",c.parameters.value,{weight:true}),
        matrix("Queries",c.head.q,c.tokenRows),matrix("Keys",c.head.k,c.tokenRows),matrix("Values",c.head.v,c.tokenRows)];
    }},
  {title:"Hilbert scores and key mask", text:"Subtract each key vector from the selected query. Negate the range across head-local channels. The padding key receives negative infinity, so it cannot win the reduction.",
    equation:"sᵢⱼ = minᵣ(qᵢᵣ − kⱼᵣ) − maxᵣ(qᵢᵣ − kⱼᵣ)",render(c) {
      const terms=c.head.differences[c.query].map((diff,i)=>[...diff,Math.min(...diff),Math.max(...diff),c.graph.valid[i]?"valid":"masked",c.head.scores[c.query][i]]);
      const diff=c.head.differences[c.query][c.key];
      const score=c.head.scores[c.query][c.key];
      return ["Selected Q − K = " + showRow(diff) + ". "
        + (c.graph.valid[c.key] ? "Score = " + fmt(Math.min(...diff)) + " − " + fmt(Math.max(...diff)) + " = " + fmt(score) + "." : "Key mask sets the score to −∞."),
        matrix("Selected query versus each key",terms,{rows:c.graph.tokens,selected:c.key,cols:["difference 1","difference 2","min","max","key mask","score"]}),
        matrix("All scores for the selected head",c.head.scores,{rows:c.graph.tokens,cols:c.graph.tokens,selected:c.query})];
    }},
  {title:"Max-plus context per channel", text:"Add the query–key score to each value channel. A channel has its own maximizing keys. Every exact tie is highlighted. The blockwise variant first reduces within each key block, then merges those maxima.",
    equation:"cᵢᵣ = maxⱼ(sᵢⱼ + vⱼᵣ)",render(c) {
      const candidates=c.head.candidates[c.query], max=c.head.context[c.query][c.channel];
      const winners=candidates.flatMap((row, i) => row[c.channel] === max ? [i] : []);
      const blocks=c.head.blocks[c.query];
      return ["Channel " + (c.channel+1) + " reaches " + fmt(max) + " at " + winners.map(i => c.graph.tokens[i]).join(", ") + "."
        + (winners.length > 1 ? " The source sends the max gradient through the first matching key." : ""),
        matrix("Score-plus-value candidates",candidates,{rows:c.graph.tokens,selected:c.key,channel:c.channel,winners}),
        matrix("Partial maximum by key block",blocks,{channel:c.channel}),
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
        linearTerms("Selected attention output projection",l.attention.decoded[c.query],c.parameters.output[c.global],0,l.attention.output[c.query][c.global]),
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
        linearTerms("Selected hidden projection",l.secondNorm.values[c.query],p.ffnIn[c.hidden],p.ffnInBias[c.hidden],l.ffnPre[c.query][c.hidden]),
        matrix("Expanded activations",l.ffnPre,c.tokenRows),matrix("After GELU",l.ffnActive,c.tokenRows),
        matrix("GELU factors",l.ffnPre[c.query].map((x,i)=>[x,cdf(x),l.ffnActive[c.query][i]]),
          {rows:columns(l.ffnPre[c.query].length),cols:["x","Φ(x)","x · Φ(x)"],selected:c.hidden}),
        matrix("Readback weights and bias",p.ffnOut.map((w,i)=>[...w,p.ffnOutBias[i]]),{weight:true,cols:[...columns(6),"bias"]}),
        linearTerms("Selected FFN output projection",l.ffnActive[c.query],p.ffnOut[c.global],p.ffnOutBias[c.global],l.ffnOutput[c.query][c.global]),
        matrix("Selected FFN residual addition",[[l.residual[c.query][c.global],l.ffnOutput[c.query][c.global],l.output[c.query][c.global]]],
          {rows:[c.graph.tokens[c.query]],cols:["residual","FFN output","sum"]}),
        matrix("Feed-forward output",l.ffnOutput,c.tokenRows),matrix("Layer output",l.output,c.tokenRows)];
    }},
];

const denseSteps = [
  {title:"Packed QKV projection",text:"A single affine projection maps each normalized token from four channels to twelve. Consecutive blocks hold all query, key and value channels. Each row of the weight matrix can see every input channel.",
    equation:"[Q; K; V]ᵢ = Wqkv nᵢ + bqkv",render(c) {
      const p=c.parameters.dense;
      const a=c.layer.attention;
      return ["The packed tensor stores 3 kinds × 2 heads × 2 channels per token.",
        matrix("Packed QKV weights and biases",p.qkv.map((w,i)=>[...w,p.qkvBias[i]]),
          {weight:true,cols:[...columns(4),"bias"]}),
        matrix("Packed QKV activation",a.packed,c.tokenRows),
        ...["Q","K","V"].map((name,part)=>{
          const row=part*4+c.global;
          const token=part===0?c.query:c.key;
          return linearTerms(name+" projection for selected channel",c.layer.firstNorm.values[token],p.qkv[row],p.qkvBias[row],a.packed[token][row]);
        })];
    }},
  {title:"Split heads and scale queries",text:"Read the head-local slices from the packed projection. Ordinary μP multiplies each query coordinate by √attn_mult divided by head width. Keys and values retain their projected values.",
    equation:"Qscaled = Q · √attn_mult / dhead",render(c) {
      const a=c.layer.attention;
      return ["Query multiplier " + fmt(a.scale) + " gives " + showRow(c.head.scaled[c.query]) + ".",
        matrix("Query slice",c.head.q,c.tokenRows),matrix("Scaled queries",c.head.scaled,c.tokenRows),
        matrix("Key slice",c.head.k,c.tokenRows),matrix("Value slice",c.head.v,c.tokenRows)];
    }},
  {title:"Dot-product logits and key mask",text:"Multiply matching query and key coordinates and sum them. The padding key's logit becomes negative infinity. The unmasked logits are retained here for inspection.",
    equation:"sᵢⱼ = Qscaledᵢ · Kⱼ; sᵢⱼ = −∞ for padding keys",render(c) {
      const q=c.head.scaled[c.query], k=c.head.k[c.key];
      const score=c.head.scores[c.query][c.key];
      return ["Selected score = " + (score===-Infinity ? "−∞ after masking." : fmt(score) + "."),
        matrix("Selected dot product",q.map((v,i)=>[v,k[i],v*k[i]]),{cols:["scaled Q","K","product"],channel:2,selected:c.channel}),
        matrix("Logits before the key mask",c.head.logits,{...c.tokenRows,cols:c.graph.tokens,channel:c.key}),
        matrix("Masked scores",c.head.scores,{...c.tokenRows,cols:c.graph.tokens,channel:c.key})];
    }},
  {title:"Softmax weights and value sum",text:"Subtract the largest valid score, exponentiate, and divide by the sum over keys. The padded key gets zero weight. Each context channel sums all weighted values from the same head.",
    equation:"pᵢⱼ = exp(sᵢⱼ − maxₖ sᵢₖ) / Σⱼ exp(sᵢⱼ − maxₖ sᵢₖ); cᵢ = Σⱼ pᵢⱼ Vⱼ",render(c) {
      const h=c.head,q=c.query;
      return ["The selected query uses softmax offset " + fmt(h.maxima[q]) + " and denominator " + fmt(h.totals[q]) + ".",
        matrix("Selected softmax and value terms",h.scores[q].map((score,i)=>[
          score,h.exponentials[q][i],h.probabilities[q][i],h.v[i][c.channel],h.candidates[q][i][c.channel]
        ]),{rows:c.graph.tokens,cols:["score","exp shifted","weight","V channel","weighted V"],selected:c.key}),
        matrix("Softmax weights",h.probabilities,{...c.tokenRows,cols:c.graph.tokens,channel:c.key}),
        matrix("Weighted sum by channel",h.context,c.tokenRows)];
    }},
  {title:"Join heads, project, and add",text:"Concatenate the two context vectors in head order, mix their channels with an affine projection, then add the residual from the layer input.",
    equation:"a = Wo concat(c¹,c²) + bo; r = x + a",render(c) {
      const l=c.layer,p=c.parameters.dense,a=l.attention;
      return ["Selected row = " + showRow(l.input[c.query]) + " + " + showRow(a.output[c.query]) + " = " + showRow(l.residual[c.query]) + ".",
        ...a.heads.map((h,i)=>matrix("Head "+(i+1)+" context",h.context,c.tokenRows)),
        matrix("Concatenated heads",a.concatenated,c.tokenRows),
        matrix("Output projection weights and biases",p.output.map((w,i)=>[...w,p.outputBias[i]]),
          {weight:true,cols:[...columns(4),"bias"]}),
        linearTerms("Selected attention output projection",a.concatenated[c.query],p.output[c.global],p.outputBias[c.global],a.output[c.query][c.global]),
        matrix("Attention output",a.output,c.tokenRows),matrix("Post-attention residual",l.residual,c.tokenRows)];
    }},
];

function normView(c, input, norm, params) {
  const [mean,variance]=norm.stats[c.query];
  const row=input[c.query];
  const denominator=Math.sqrt(variance+1e-5);
  const centered=row.map(value=>value-mean);
  return ["Selected row mean " + fmt(mean) + " and variance " + fmt(variance) + ".",
    matrix("Input",input,c.tokenRows),
    matrix("Per-token statistics",norm.stats,{...c.tokenRows,cols:["mean","variance"]}),
    matrix("Selected row normalization",row.map((value,i)=>[
      value,centered[i],centered[i]**2,centered[i]/denominator,params.gamma[i],params.beta[i],norm.values[c.query][i]
    ]),{rows:columns(row.length),cols:["input","centered","squared deviation","standardized","gain","offset","output"],selected:c.global}),
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

let index = 0, graph, trace, currentInput, sceneAxes = [];
const routes = [
  [0, "Embed", n => n + " × 4"],
  ...[0, 1].flatMap(layer => {
    const first = 1 + layer * layerSteps.length;
    return [
      [first, "Norm", n => n + " × 4"],
      [first + 1, "Split heads", n => "2 × " + n + " × 2"],
      [first + 2, "Q / K / V", n => "3 × 2 × " + n + " × 2"],
      [first + 3, "Score", n => "2 × " + n + " × " + n],
      [first + 4, "Reduce", n => "2 × " + n + " × 2"],
      [first + 5, "Merge + skip", n => n + " × 4"],
      [first + 6, "Norm", n => n + " × 4"],
      [first + 7, "FFN + skip", n => n + " × 6 → 4"],
    ];
  }),
  [17, "Final norm", n => n + " × 4"],
  [18, "Readout", () => "scalar"],
];
const makeButton = (i, label) => {
  const button = node("button", label);
  button.type = "button";
  button.addEventListener("click", () => select(i, true));
  return button;
};
const buttons = steps.map((step, i) =>
  makeButton(i, String(i+1).padStart(2,"0") + " · " + (step.layer === undefined ? "" : "L" + (step.layer+1) + " ") + step.title));
ui.steps.replaceChildren(...buttons);
const mapButtons = routes.map(([i, label]) => {
  const button = makeButton(i, label);
  const shape = node("small");
  const caption = node("span",label);
  button.replaceChildren(caption,shape);
  return {button, shape, caption};
});
for (const [name, from, to] of [["Input", 0, 1], ["Layer 1", 1, 9], ["Layer 2", 9, 17], ["Output", 17, 19]]) {
  const row = node("div");
  const track = node("div");
  row.className = "trace-map-row";
  track.className = "trace-map-track";
  track.append(...mapButtons.slice(from, to).map(item => item.button));
  row.append(node("strong", name), track);
  ui.map.append(row);
}
function drawScene() {
  const layer=trace.layers[steps[index].layer];
  const query=Number(ui.token.value), key=Number(ui.key.value);
  const head=Number(ui.head.value), channel=Number(ui.channel.value);
  const row=(label,values)=>[{label,values,selection:[query]}];
  const view=layer ? (index-1)%layerSteps.length : -1;
  sceneAxes = [];
  let tensors;
  if (view>=1 && view<=4 && ui.kernel.value==="dense") {
    const attention=layer.attention;
    if (view===1) {
      tensors=row("Packed Q · K · V",attention.packed);
      sceneAxes = [{row:"token",packed:true}];
    }
    else {
      const name=["","","scaled","scores","probabilities"][view];
      tensors=attention.heads.map((values,i)=>({label:"H"+(i+1)+" "+name,values:values[name],head:i,
        selection:i===head?[query,view>=3?key:channel]:null,
        mask:view>=3?(_,c)=>!graph.valid[c]:null}));
      sceneAxes = tensors.map((_,head)=>({row:"token",column:view>=3?"key":"channel",head}));
    }
  } else if (view>=1 && view<=4) {
    const heads=layer.attention.heads;
    if (view===2) {
      tensors=["q","k","v"].map(name=>({label:"H"+(head+1)+" "+name.toUpperCase(),values:heads[head][name],
        head,selection:[name==="q"?query:key,channel]}));
      sceneAxes = ["token","key","key"].map(row=>({row,column:"channel"}));
    } else {
      const name=["","z","","scores","context"][view];
      tensors=heads.map((values,i)=>({label:"H"+(i+1)+" "+name,values:values[name],head:i,
        selection:i===head?[query,view===3?key:channel]:null,
        mask:view===3?(_,c)=>!graph.valid[c]:null}));
      sceneAxes = tensors.map((_,head)=>({row:"token",column:view===3?"key":"channel",head}));
    }
  } else if (layer) {
    const [label,values]=({0:["Norm 1",layer.firstNorm.values],5:["Residual",layer.residual],
      6:["Norm 2",layer.secondNorm.values],7:["Layer output",layer.output]})[view];
    tensors=row(label,values);
    sceneAxes = [{row:"token",packed:true}];
  } else {
    tensors=index===0?row("Token embeddings",trace.input):index===steps.length-1
      ? [{label:"Graph readout",values:[[trace.readout]],selection:[0,0]}] : row("Final states",trace.masked);
    if(index!==steps.length-1) sceneAxes = [{row:"token",packed:true}];
  }
  tensorScene(ui.scene,tensors,head,Number(ui.angle.value),Number(ui.pitch.value));
  ui.scene.classList.toggle("is-selectable",sceneAxes.length>0);
}
function render() {
  const selected=steps[index], view=selected.layer===undefined?-1:(index-1)%layerSteps.length;
  const active=ui.kernel.value==="dense"?denseSteps[view-1]:null;
  const step=active?{...active,layer:selected.layer}:selected;
  const query=Number(ui.token.value), key=Number(ui.key.value), channel=Number(ui.channel.value);
  const head=Number(ui.head.value), global=head*2+channel;
  const title=(step.layer===undefined?"":"Layer "+(step.layer+1)+" · ")+step.title;
  const layer=trace.layers[step.layer];
  ui.title.textContent=title;
  ui.explanation.textContent=step.text;
  ui.equation.textContent=step.equation;
  ui.progress.textContent="Step "+(index+1)+" of "+steps.length;
  ui.prev.disabled=index===0;
  ui.next.disabled=index===steps.length-1;
  ui["hidden-setting"].hidden=view!==7;
  const tokenRows={rows:graph.tokens,selected:query};
  const [note,...figures]=step.render({graph,trace,query,key,channel,layer,tokenRows,
    global,hidden:Number(ui["hidden-channel"].value),
    parameters:weights.layers[step.layer],head:layer?.attention.heads[head]});
  ui.numerical.textContent=note;
  ui.tensors.replaceChildren(...figures);
  drawScene();
}
function select(i, focus=false) {
  index=i;
  for (const group of [buttons, mapButtons.map(item => item.button)]) {
    group.forEach((button,j)=>{if(j===i)button.setAttribute("aria-current","step");else button.removeAttribute("aria-current");});
  }
  render();
  if(focus) {
    history.replaceState(null, "", "#step-" + (i + 1));
    ui.title.focus({preventScroll:window.innerWidth>760});
  }
}
function runGraph(next) {
  graph = next;
  const attnMult=Number(ui.multiplier.value), dense=ui.kernel.value==="dense";
  ui["multiplier-value"].value=attnMult.toFixed(2);
  ui["multiplier-setting"].hidden=!dense;
  ui["block-setting"].hidden=dense;
  trace = runTrace(graph, ui.reduction.value === "full" ? graph.tokens.length : Number(ui.reduction.value),
    {kind:ui.kernel.value,attnMult});
  for (const select of [ui.token, ui.key]) {
    const prior=Number(select.value);
    select.replaceChildren(...graph.tokens.map((token,i)=>new Option(token,i)));
    select.value=Math.min(prior,graph.tokens.length-1);
  }
  const count=graph.tokens.length;
  const labels=["Packed QKV","Split + scale","Logits","Softmax + sum","Merge + skip"];
  const shapes=[count+" × 12","2 × "+count+" × 2","2 × "+count+" × "+count,"2 × "+count+" × 2",count+" × 4"];
  mapButtons.forEach(({shape,caption}, i) => {
    const layer=steps[i].layer, op=(i-1)%layerSteps.length-1;
    const alternate=dense && layer!==undefined ? denseSteps[op] : null;
    caption.textContent=alternate?labels[op]:routes[i][1];
    shape.textContent=alternate?shapes[op]:routes[i][2](count);
    buttons[i].textContent=String(i+1).padStart(2,"0")+" · "+(layer===undefined?"":"L"+(layer+1)+" ")+(alternate??steps[i]).title;
  });
  render();
}
function preset() {
  const strength=Number(ui.strength.value);
  applyInput(graphPreset(ui.input.value,strength));
  ui["input-status"].textContent="";
}
function applyInput(input) {
  const next=graphFromInput(input);
  currentInput=input;
  const first=input.edgeFeatures[0]?.[0];
  ui.strength.disabled=first===undefined;
  if(first!==undefined) ui.strength.value=first;
  ui["strength-value"].value=first===undefined?"No edge":fmt(first);
  ui.json.value=JSON.stringify(input,null,2);
  runGraph(next);
}
ui.apply.addEventListener("click", () => {
  try {
    const input = JSON.parse(ui.json.value);
    applyInput(input ?? {});
    ui["input-status"].textContent = "Custom graph applied.";
  } catch (error) {
    ui["input-status"].textContent = error instanceof SyntaxError ? "Enter valid JSON. The previous graph is still shown." : error.message;
  }
});
ui.angle.addEventListener("input",drawScene);
ui.pitch.addEventListener("input",drawScene);
ui.scene.addEventListener("click",event=>{
  const cell=event.target.closest("[data-cell]");
  if (!cell) return;
  const [tensor,row,column]=cell.dataset.cell.split(",").map(Number);
  const axis=sceneAxes[tensor];
  if (!axis) return;
  ui[axis.row].value=row;
  if (axis.column) ui[axis.column].value=column;
  if (axis.head!==undefined) ui.head.value=axis.head;
  if (axis.packed) {
    ui.head.value=Math.floor(column/2)%2;
    ui.channel.value=column%2;
  }
  render();
});
ui.prev.addEventListener("click",()=>select(index-1,true));
ui.next.addEventListener("click",()=>select(index+1,true));
ui.input.addEventListener("change",preset);
ui.strength.addEventListener("input",()=>{
  const edgeFeatures=currentInput.edgeFeatures.map((row,i)=>i===0?[Number(ui.strength.value),row[1]]:row);
  applyInput({...currentInput,edgeFeatures});
});
ui.reduction.addEventListener("change",()=>runGraph(graph));
ui.kernel.addEventListener("change",()=>{
  const url=new URL(location.href);
  if(ui.kernel.value==="dense") url.searchParams.set("kernel","dense");
  else url.searchParams.delete("kernel");
  history.replaceState(null,"",url);
  runGraph(graph);
});
ui.multiplier.addEventListener("input",()=>runGraph(graph));
[ui.token,ui.key,ui.head,ui.channel,ui["hidden-channel"]].forEach(control=>control.addEventListener("change",render));
function fromHash() {
  const requested = Number(location.hash.match(/^#step-(\d+)$/)?.[1]);
  select(requested >= 1 && requested <= steps.length ? requested - 1 : 0);
}
if(new URLSearchParams(location.search).get("kernel")==="dense") ui.kernel.value="dense";
preset();
fromHash();
window.addEventListener("hashchange", fromHash);
