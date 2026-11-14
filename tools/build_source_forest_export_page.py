"""Offline source-prefix inspector and matching vector figure from verified exports.

Run only with an existing complete export directory. No models are constructed,
no source programs run, and no training or publication is performed.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import html
import json
from mora3.data.forest_source_exports import ForestSourceExport

ROOT=Path(__file__).resolve().parents[1]


def wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def build(directory):
    directory=Path(directory).absolute()
    report=json.loads((directory/'verification.json').read_bytes())
    if (report['encoder_trained']is not False or report['optimizer_updates']!=0
            or len(report['rows'])!=3 or any(x['status']!='exported-evaluation-only'for x in report['rows'])):
        raise ValueError('Three complete real evaluation-only exports required')
    examples=[]
    for row in report['rows']:
        path=Path(row['path'])
        if path.is_absolute()or '..'in path.parts:raise ValueError('Relative exact export path required')
        source=ForestSourceExport.load(directory/path,expected_sha256=row['sha256'])
        d=source.document()
        if d['split']!='test' or d['encoder']!=report['encoder']:raise ValueError('Exact source role and encoder required')
        native_payloads=source.payloads();payloads=[json.loads(p)for p in native_payloads];values=source.latents().tolist()
        nodes=[dict(id=name,parent=d['parents'][i],direction=d['directions'][i],status=d['statuses'][i],
            latent=values[i],visible_facts=[f['name']for f in payloads[i]['facts']],
            payload_bytes=len(native_payloads[i]),payload_sha256=sha256(native_payloads[i]).hexdigest())
            for i,name in enumerate(d['node_ids'])]
        examples.append(dict(trace_id=row['trace_id'],export_sha256=source.sha256,source_sha256=row['original_sha256'],
            source_revision=row['source_revision'],original_dataflow_nodes=row['original_dataflow_nodes'],
            states=row['reasoning_states'],facts=row['source_facts'],prefix_tokens=row['total_native_prefix_tokens'],
            split='test',evidence_kind='computed',nodes=nodes))
    data=dict(schema='Mora3.SourceForestExportInspector/v1',trained=False,training_admission=False,
        optimizer_updates=0,encoder=report['encoder'],examples=examples,
        source_report_sha256=sha256((directory/'verification.json').read_bytes()).hexdigest(),
        visual_scope='two-coordinate projection of frozen untrained eight-dimensional prefix exports')
    destination=ROOT/'docs/project';destination.mkdir(parents=True,exist_ok=True)
    (destination/'source-forest-exports-data.json').write_text(wire(data)+'\n')
    figure=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560" role="img" aria-labelledby="title desc">',
        '<title id="title">Complete source-prefix exports</title><desc id="desc">Three retained numerical references preserve all source nodes and produce four, twenty-eight and thirty-four prefix states, each with eight untrained diagnostic coordinates. They remain evaluation-only.</desc>',
        '<rect width="1200" height="560" fill="#f7f8fa"/>',
        '<g font-family="Arial,sans-serif" fill="#172536"><text x="52" y="55" font-size="30" font-weight="700">Complete source-prefix exports</text>',
        '<text x="52" y="88" font-size="17">Original computations → complete execution prefixes → fixed native coordinates</text>',
        '<g font-size="15" fill="#526173"><text x="52" y="143">RETAINED REFERENCE</text><text x="400" y="143">SOURCE NODES</text><text x="640" y="143">PREFIX STATES</text><text x="890" y="143">EXPORTED AXES</text></g>']
    for i,row in enumerate(examples):
        y=195+i*96
        figure.extend([f'<path d="M52 {y+38} H1145" stroke="#dce2e7"/>',
            f'<text x="52" y="{y}" font-size="23">{html.escape(row["trace_id"])}</text>',
            f'<text x="435" y="{y}" font-size="29" font-weight="700">{row["original_dataflow_nodes"]}</text>',
            f'<path d="M515 {y-10} H600 l-10 -5 m10 5 l-10 5" fill="none" stroke="#6b7a8c" stroke-width="2"/>',
            f'<text x="684" y="{y}" font-size="29" font-weight="700">{row["states"]}</text>',
            f'<path d="M765 {y-10} H850 l-10 -5 m10 5 l-10 5" fill="none" stroke="#6b7a8c" stroke-width="2"/>',
            f'<text x="895" y="{y}" font-size="29" fill="#185b81">{row["states"]} × 8</text>'])
    figure.extend(['<text x="52" y="497" font-size="17">No source truncation · exact export-byte readback · original split: test</text>',
        '<text x="52" y="528" font-size="17" fill="#8a451d">Untrained diagnostic encoder. No biological validation or training admission.</text></g></svg>'])
    svg=''.join(figure)
    for folder in(ROOT/'docs/project/figures',ROOT/'paper/figures'):
        folder.mkdir(parents=True,exist_ok=True);(folder/'source-forest-exports.svg').write_text(svg)
    counts='<table class="compact-counts"><caption>Original nodes → prefix states × coordinates</caption><thead><tr><th>Reference</th><th>Nodes</th><th>State axes</th></tr></thead><tbody>'+''.join(
        '<tr><td>'+html.escape(e['trace_id'])+'</td><td>'+str(e['original_dataflow_nodes'])+'</td><td>'+str(e['states'])+' × 8</td></tr>'for e in examples)+'</tbody></table>'
    text=PAGE.replace('__DATA__',wire(data).replace('<','\\u003c')).replace('__COUNTS__',counts)
    (destination/'source-forest-exports.html').write_text(text)
    return dict(examples=len(examples),prefix_states=sum(e['states']for e in examples),html_bytes=len(text.encode()),
        data_sha256=sha256(wire(data).encode()).hexdigest(),trained=False)


PAGE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mora-3 · Source-prefix exports</title>
<style>
:root{color-scheme:light;--ink:#172536;--muted:#526173;--accent:#185b81;--line:#dce2e7;--focus:#a84914}*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:28px 24px 64px}a{color:var(--accent)}h1{font-size:clamp(30px,4vw,48px);line-height:1.12;letter-spacing:-.035em;margin:22px 0 14px}h2{font-size:22px;margin:0 0 12px}p{max-width:850px}.eyebrow{font-size:13px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}.notice{border-left:4px solid var(--focus);padding:12px 18px;background:#fff5ec;font-size:14px}.controls{display:grid;grid-template-columns:2fr 2fr 1fr 1fr;gap:14px;margin:24px 0}label{display:grid;gap:6px;font-size:13px;font-weight:600}select,button{min-height:44px;font:inherit;color:var(--ink);background:white;border:1px solid #a9b6c5;border-radius:6px;padding:8px;width:100%}button{cursor:pointer;color:white;background:var(--accent);border-color:var(--accent);font-size:14px}select:focus-visible,button:focus-visible,a:focus-visible{outline:3px solid var(--focus);outline-offset:3px}.panel{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;min-width:0}.grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:20px}.meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}#projection{display:block;width:100%;height:360px;overflow:visible}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px 4px;overflow-wrap:anywhere}td:last-child{font-variant-numeric:tabular-nums}caption{text-align:left;font-weight:650;margin:0 0 8px}#facts{padding-left:20px;font-size:13px;max-height:300px;overflow:auto;overflow-wrap:anywhere}code{font-size:12px;overflow-wrap:anywhere}.actions{display:flex;gap:10px;margin:16px 0}.actions button{width:auto}.mark{cursor:pointer;outline:none}.mark:focus{stroke:var(--focus);stroke-width:3px}.summary{margin-top:24px}.summary img{width:100%;height:auto}.compact-counts{display:none}.note{font-size:13px;color:var(--muted)}.hash{padding:10px;background:#f2f5f8;border-radius:5px}.metrics{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}.metrics strong{font-size:20px}.metrics span{display:block;font-size:12px;color:var(--muted)}@media(max-width:700px){.summary img{display:none}.compact-counts{display:table}main{padding:18px 16px 40px}.controls{grid-template-columns:1fr 1fr;gap:12px}.grid{grid-template-columns:1fr}.panel{padding:14px}#projection{height:280px}.actions{flex-direction:column}.actions button{width:100%}h2{font-size:20px}}@media print{.controls,.actions,nav{display:none}.grid{display:block}.panel{break-inside:avoid}#facts{max-height:none}}
</style></head><body><main>
<nav aria-label="Breadcrumb"><a href="../index.html">Mora-3</a> / <a href="grh-research-bridge.html">GRH bridge</a> / Source exports</nav>
<p class="eyebrow">Implementation evidence · 13 November 2026</p><h1>Every supported source prefix.<br>No truncated trajectory.</h1>
<p>Inspect complete execution-prefix embeddings from three retained numerical references. Source identity, direction, original split and exact exported coordinates remain inspectable.</p>
<div class="notice"><strong>Untrained diagnostic. Evaluation-only.</strong> These are fixed encodings of given computations, not learned high-quality reasoning, sampled FoT search, biological causality, or a completed training campaign. The two-axis plot omits six coordinates; the table retains all eight. Original source payloads remain in the checked export artifacts.</div>
<div class="controls"><label>Retained reference<select id="reference"></select></label><label>Available prefix<select id="prefix"></select></label><label>Horizontal coordinate<select id="x-axis"></select></label><label>Vertical coordinate<select id="y-axis"></select></label></div>
<div class="grid"><section class="panel" aria-labelledby="plot-heading"><h2 id="plot-heading">Frozen prefix coordinates</h2><div id="summary" class="metrics"></div><svg id="projection" role="img" aria-labelledby="svg-title svg-desc"></svg><p class="note">Arrows connect parent execution prefixes. Coordinate values are native embedding units, with an <strong>untrained</strong> frozen encoder here—not physical distances. Select a prefix using the menu, point, or keyboard.</p><div class="actions"><button id="export">Export selected evidence JSON</button><button id="export-svg">Export projection SVG</button></div><p class="meta">Exact complete source-export SHA256</p><div class="hash"><code id="export-sha"></code></div></section>
<section class="panel" aria-labelledby="selected-title"><p class="eyebrow">Selected prefix</p><h2 id="selected-title" aria-live="polite"></h2><p id="node-description" class="meta"></p><table id="coordinates"><caption>All eight exported coordinates</caption><thead><tr><th>Axis</th><th>Exact decimal readback</th></tr></thead><tbody></tbody></table><h3>Facts available at this prefix</h3><p class="note">No sibling or later fact is shown in this list. This viewer is reference inspection, not a policy input.</p><ul id="facts"></ul></section></div>
<section class="panel summary"><h2>Complete source accounting</h2>__COUNTS__<img src="figures/source-forest-exports.svg" alt="Three references: six source nodes produce four prefix states; fifty-nine produce twenty-eight; sixty-six produce thirty-four. Each prefix has eight diagnostic coordinates."><p class="note">Raw JSON payload capacity and aggregate prefix work are not full-attention context lengths. The source encoder preserves all supported tree branches; merged DAG histories and censored branches still require separate representations.</p><p><a href="../source_forest_exports_20261114.md">Implementation methods and remaining requirements</a> · <a href="source-forest-exports-data.json">Inspectable figure data</a></p></section>
<noscript><p>JavaScript is disabled. The source-accounting figure and linked numeric data remain available.</p></noscript>
<script id="source-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('source-data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const controls=['reference','prefix','x-axis','y-axis'];let source=data.examples[0],selected=0,xaxis=0,yaxis=1;
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
for(const row of data.examples)$('reference').append(option(row.trace_id,row.trace_id));
for(let i=0;i<8;i++){ $('x-axis').append(option(i,'Coordinate '+i));$('y-axis').append(option(i,'Coordinate '+i)); }
function number(value,fallback,max){return /^\d+$/.test(value||'')&&Number(value)<=max?Number(value):fallback;}
function readState(){const q=new URLSearchParams(location.hash.slice(1));source=data.examples.find(e=>e.trace_id===q.get('source'))||data.examples[0];selected=number(q.get('node'),0,source.nodes.length-1);xaxis=number(q.get('x'),0,7);yaxis=number(q.get('y'),1,7);if(xaxis===yaxis)yaxis=(xaxis+1)%8;}
function saveState(){const h=new URLSearchParams({source:source.trace_id,node:String(selected),x:String(xaxis),y:String(yaxis)}).toString();if(location.hash.slice(1)!==h)history.pushState(null,'','#'+h);}
function node(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs||{}))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function decimal(v){return Object.is(v,-0)?'-0.0':String(v);}
function render(){
 $('reference').value=source.trace_id;$('prefix').replaceChildren(...source.nodes.map((n,i)=>option(i,n.id+' · '+n.status)));$('prefix').value=String(selected);$('x-axis').value=String(xaxis);$('y-axis').value=String(yaxis);
 const current=source.nodes[selected];$('selected-title').textContent=current.id;$('node-description').textContent=current.status+' · '+current.direction+' · '+current.payload_bytes.toLocaleString()+' retained payload bytes';$('export-sha').textContent=source.export_sha256;
 $('facts').replaceChildren(...current.visible_facts.map(f=>{const li=document.createElement('li');li.textContent=f;return li;}));
 $('coordinates').querySelector('tbody').replaceChildren(...current.latent.map((v,i)=>{const tr=document.createElement('tr');for(const text of['c'+i,decimal(v)]){const td=document.createElement('td');td.textContent=text;tr.append(td);}return tr;}));
 $('summary').replaceChildren(...[[source.original_dataflow_nodes,'original dataflow nodes'],[source.states,'complete prefix states'],['test','unchanged split']].map(([value,label])=>{const div=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;div.append(strong,span);return div;}));
 const svg=$('projection'),W=Math.max(280,Math.round(svg.getBoundingClientRect().width)),H=W<500?280:360,left=58,right=22,top=28,bottom=48;svg.style.height=H+'px';svg.setAttribute('font-family','system-ui,sans-serif');svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.replaceChildren(node('title',{id:'svg-title'},source.trace_id+' untrained diagnostic prefix projection'),node('desc',{id:'svg-desc'},'Coordinates '+xaxis+' and '+yaxis+'. All '+source.nodes.length+' retained prefixes. Select a prefix from the accessible menu for complete values.'));
 const defs=node('defs'),marker=node('marker',{id:'arrow',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});marker.append(node('path',{d:'M0 0 L10 5 L0 10 Z',fill:'#8295a5'}));defs.append(marker);svg.append(defs);
 function domain(axis){let a=Math.min(...source.nodes.map(n=>n.latent[axis])),b=Math.max(...source.nodes.map(n=>n.latent[axis]));let p=(b-a)*.12||.1;return[a-p,b+p];}
 const dx=domain(xaxis),dy=domain(yaxis),X=v=>left+(v-dx[0])/(dx[1]-dx[0])*(W-left-right),Y=v=>H-bottom-(v-dy[0])/(dy[1]-dy[0])*(H-top-bottom);
 for(let i=0;i<4;i++){let t=i/3,x=left+t*(W-left-right),y=H-bottom-t*(H-top-bottom);svg.append(node('line',{x1:x,y1:top,x2:x,y2:H-bottom,stroke:'#e4e9ed','stroke-width':1}),node('line',{x1:left,y1:y,x2:W-right,y2:y,stroke:'#e4e9ed','stroke-width':1}),node('text',{x,y:H-bottom+19,'text-anchor':'middle','font-size':11,fill:'#526173'},(dx[0]+t*(dx[1]-dx[0])).toPrecision(3)),node('text',{x:left-8,y:y+4,'text-anchor':'end','font-size':11,fill:'#526173'},(dy[0]+t*(dy[1]-dy[0])).toPrecision(3)));}
 svg.append(node('text',{x:(left+W-right)/2,y:H-5,'text-anchor':'middle','font-size':12,fill:'#172536'},'Coordinate '+xaxis),node('text',{x:12,y:top-8,'font-size':12,fill:'#172536'},'c'+yaxis));
 for(const n of source.nodes){if(n.parent<0)continue;const p=source.nodes[n.parent];svg.append(node('line',{x1:X(p.latent[xaxis]),y1:Y(p.latent[yaxis]),x2:X(n.latent[xaxis]),y2:Y(n.latent[yaxis]),stroke:'#8295a5','stroke-width':1.3,'marker-end':'url(#arrow)'}));}
 source.nodes.forEach((n,i)=>{const dot=node('circle',{cx:X(n.latent[xaxis]),cy:Y(n.latent[yaxis]),r:i===selected?7:4,fill:i===selected?'#a84914':'#185b81',stroke:'white','stroke-width':1.5,class:'mark',tabindex:0,role:'button','aria-label':'Select '+n.id,'aria-pressed':i===selected});dot.append(node('title',{},n.id));const choose=()=>{selected=i;saveState();render();};dot.addEventListener('click',choose);dot.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});svg.append(dot);});
 svg.append(node('text',{x:W-right,y:16,'text-anchor':'end','font-size':12,fill:'#a84914'},current.id+' selected'));document.documentElement.dataset.ready='true';
}
for(const id of controls)$(id).addEventListener('change',()=>{const previous=source.trace_id;source=data.examples.find(e=>e.trace_id===$('reference').value);selected=previous!==source.trace_id?0:Number($('prefix').value);xaxis=Number($('x-axis').value);yaxis=Number($('y-axis').value);if(xaxis===yaxis)yaxis=(xaxis+1)%8;saveState();render();});
function download(text,type,name){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export').addEventListener('click',()=>download(JSON.stringify({schema:'Mora3.SelectedSourcePrefixEvidence/v1',trained:false,training_admission:false,source_revision:source.source_revision,source_sha256:source.source_sha256,export_sha256:source.export_sha256,encoder:data.encoder,node:source.nodes[selected]},null,2),'application/json',source.trace_id+'-prefix-'+selected+'.json'));
$('export-svg').addEventListener('click',()=>download(new XMLSerializer().serializeToString($('projection')),'image/svg+xml',source.trace_id+'-projection.svg'));
window.addEventListener('popstate',()=>{readState();render();});window.addEventListener('hashchange',()=>{readState();render();});window.addEventListener('resize',render);readState();render();
</script></main></body></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-dir',required=True)
    args=parser.parse_args();print(json.dumps(build(args.source_dir),indent=2))
