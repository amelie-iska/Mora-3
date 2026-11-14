"""Build the complete source-backed methods paper with deliberately empty results.

No data downloads, optimizer calls, training launches, external math CDN, or push.
Math is typeset by the existing local TeX installation with shell escape disabled.
Diagrams are specified here as architecture/mathematical illustrations, not results.
"""
from pathlib import Path
from hashlib import sha256
from dataclasses import asdict
import argparse
import html
import json
import re
import subprocess
import shutil
import textwrap
import xml.etree.ElementTree as ET
import markdown
from mora3.research.mup_sweep_selection import planned_proxy_trials

ROOT=Path(__file__).resolve().parents[1]
NS='http://www.w3.org/2000/svg'
INK='#182637';MUTED='#536477';BLUE='#175c7b';AMBER='#94531e';PALE='#f3f6f8';LINE='#c7d4df'


def wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def escape(value):return html.escape(str(value),quote=True)


def svg_start(key,title,description,width=1080,height=540):
    return [f'<svg xmlns="{NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{key}-title {key}-desc">',
        f'<title id="{key}-title">{escape(title)}</title><desc id="{key}-desc">{escape(description)}</desc>',
        f'<defs><marker id="{key}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="{BLUE}"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g font-family="Arial,sans-serif" fill="{INK}">',
        f'<text x="30" y="40" font-size="25" font-weight="700">{escape(title)}</text>',
        f'<text x="30" y="68" font-size="14" fill="{MUTED}">Architecture / mathematical illustration — not an empirical result</text>']


def card(key,x,y,title,lines,detail,width=300,height=125):
    result=[f'<g class="diagram-node" data-detail="{escape(detail)}" role="button" tabindex="0" aria-label="{escape(title)}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="9" fill="{PALE}" stroke="{LINE}"/>',
        f'<text x="{x+18}" y="{y+28}" font-size="19" font-weight="700">{escape(title)}</text>']
    for i,line in enumerate(lines):result.append(f'<text x="{x+18}" y="{y+55+21*i}" font-size="15" fill="{MUTED}">{escape(line)}</text>')
    return ''.join(result)+ '</g>'


def arrow(key,x1,y1,x2,y2,label=None,dashed=False):
    s=f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{BLUE}" stroke-width="2" marker-end="url(#{key}-arrow)"'+(' stroke-dasharray="6 5"'if dashed else'')+'/>'
    if label:s+=f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-9}" text-anchor="middle" font-size="13" fill="{MUTED}">{escape(label)}</text>'
    return s


def grid_figure(key,title,items,edges,footer):
    out=svg_start(key,title,footer,height=510)
    positions=[(30,110),(390,110),(750,110),(30,310),(390,310),(750,310)]
    for i,(heading,lines,detail)in enumerate(items):out.append(card(key,*positions[i],heading,lines,detail))
    for start,end,label in edges:
        x,y=positions[start];u,v=positions[end]
        if y==v:out.append(arrow(key,x+300 if u>x else x,y+62,u if u>x else u+300,v+62,label))
        elif x==u:out.append(arrow(key,x+150,y+125 if v>y else y,u+150,v if v>y else v+125,label))
    out.append(f'<text x="30" y="480" font-size="14" fill="{AMBER}">{escape(footer)}</text></g></svg>')
    return ''.join(out)


def diagrams():
    architecture=grid_figure('architecture','Source-bound graph reasoning',[
        ('Native records',['Values · axes · masks','Units · provenance · rights'],'Native molecular, protein, cell, image and bioelectric payloads keep original units, masks, axes and evidence classes. No fabricated joins.'),
        ('Shared native owner',['Exact parameter aliases','Known context only'],'All registered heads share the exact designated native geometry/TokenGT objects. Unique parameters and all aliases must be accounted for.'),
        ('Topology → latent flow',['Given roots; generate branches','Then jointly denoise the forest'],'The normalized prefix policy samples an ordered variable-size topology. The continuous field then denoises all free latent coordinates conditional on that topology.'),
        ('Supervision roles',['Original train records','Detached targets and masks'],'Flow targets, token reconstructions, quality candidates, analogy correspondences and topology labels have distinct source roles. Missing labels stay unavailable.'),
        ('Geometry + analogy',['Whole-trajectory retrieval','Directed relation witnesses'],'Retrieval proposes train-only group-disjoint memories. Directed relations, costs, native geometry and persistent features constrain but do not prove biological analogy.'),
        ('Decode + independently check',['Finite token refinement','Several outputs or shortfall'],'The complete latent forest is refined into permitted token embeddings, then independently checked. A learned score is not a biological validity certificate.')],
        [(0,1,None),(1,2,None),(2,5,None),(0,3,'role split'),(3,4,None),(4,5,None)],
        'Construction, empirical quality and full training admission are separate requirements.')
    refinement=grid_figure('refinement','Propose, retain, edit, decode, verify',[
        ('Several complete proposals',['Explicit topology / flow RNG','No guaranteed initial quality'],'Generate multiple full candidates using a bound policy and field. Independent noise draws do not guarantee semantic diversity or good answers.'),
        ('Score the entire forest',['Frozen whole/node criteria','Keep earlier candidates'],'Evaluate complete candidate traces and supported nodes. Retain prior candidates and record immutable assessments.'),
        ('Bounded branch edits',['Correction or improvement','Record parent and operation'],'Edit only within declared constraints. Record lineage, proposal budgets and failures. Edited samples are not unchanged on-policy GFlowNet samples.'),
        ('Independent output checks',['Grammar and native criteria','Repairable failure is explicit'],'After decoding, assess actual output structure and native domain constraints. Missing evidence stays unknown, not pass.'),
        ('Token refinement',['Nearest permitted codebook','Complete stop / capacity check'],'Decode only complete latent forests under the exact codebook axis. Refuse overflow instead of silently truncating outputs.'),
        ('Retained candidate pool',['Elites and distinct candidates','Budgeted additional edits'],'Retain valid or repairable candidates across rounds. Repeated proposals still consume the explicit proposal budget.')],
        [(0,1,None),(1,2,None),(2,5,None),(5,4,None),(4,3,None)],
        'Return K distinct qualified outputs only when present; otherwise report the shortfall.')
    campaign=grid_figure('campaign','Frozen REAL24 → transfer → two targets',[
        ('Qualified frozen snapshot',['Full sources and all heads','Recovery · resources · ownership'],'A real release requires source rights/coverage, matching construction/model evidence, complete operators and all enabled parameter axes. No manuscript creates authority.'),
        ('24 existing proxy slots',['3 rates × 2 × 2 × 2 seeds','Failures keep their slots'],'Use the immutable native grid and durable counted reservations. Do not add retries, intermediate runs or training ablations.'),
        ('Validation-only selector',['Matched reference and exposure','Both paired seeds eligible'],'Wait for all 24 distinct terminal records, then apply the frozen paired-seed macro-normalized validation selector. Test data and target tuning never choose the point.'),
        ('Checkpoint-linked evals',['Frozen native task metrics','Populate empty result artifacts'],'Evaluate the designated real target checkpoints and report denominators, uncertainty and failure rates. No test result is invented.'),
        ('At most two target runs',['Ordinary full-model training','Retain μP parameterization'],'Construct and train the actual approximately 80M full model at predeclared target structure and budget using transferred settings, not target tuning.'),
        ('Verify actual transfer',['Read back native owner / optimizer','Generated configs are insufficient'],'Check the selected values in the qualified native target factory and actual optimizer. Transfer completion and target training completion are distinct.')],
        [(0,1,None),(1,2,None),(2,5,None),(5,4,None),(4,3,None)],
        'Protocol only, not a completed timeline. Source publication does not confer training admission.')
    metrics=grid_figure('metrics','Geometry, persistence and directed structure',[
        ('Native geometry',['Coordinates · units · signs','Do not erase chirality'],'Fix source coordinates, units, periodicity and relation signs. Pairwise Euclidean distances cannot distinguish mirror chirality by themselves.'),
        ('Complete persistence',['Declared filtration / field','Essential and censored bars kept'],'Compute the bounded supported homology and retain source diagrams. Incomplete filtrations and essential features require explicit treatment.'),
        ('Exact finite landscapes',['All layers + common knots','Two-point Gauss → Euclidean L2'],'A shared collection-specific knot basis and all landscape layers make squared differences piecewise quadratic; two-point Gaussian integration is exact in real arithmetic.'),
        ('Directed presentation',['Paths · squares · chain maps','Evidence-backed directions'],'A finite directed presentation preserves typed path relations. Its quotient functor witnesses and integer chain identities remain separate from ordinary persistence.'),
        ('Sampled / image features',['Fixed grids or smoothing','Explicit approximation'],'Sampled landscapes and persistence images are not universally lossless. Report domain, resolution, layers and precision; never compare incompatible bases.'),
        ('Qualified comparison',['Compare compatible axes','No universal equivalence claim'],'Use approximate features for explicitly approximate retrieval and exact finite distances for a complete bounded rerank where qualified. None proves biological equivalence.')],
        [(0,1,None),(1,2,None),(0,3,None),(1,4,None),(2,5,None),(4,5,None)],
        'Complete diagram information, latent geometry and biological semantics are not interchangeable.')
    key='topology';out=svg_start(key,'One ordered forest, one depth-first construction','Given roots and ADD_CHILD/CLOSE give one trace for each supported ordered forest.',height=440)
    xy=[(135,170),(385,170),(635,170),(385,295)]
    for u,v in((0,1),(1,2),(0,3)):
        x,y=xy[u];a,b=xy[v];out.append(f'<g class="topology-edge" data-target="{v}">'+arrow(key,x+26,y,a-26,b)+'</g>')
    for i,(x,y)in enumerate(xy):
        out.append(f'<g class="topology-node" data-index="{i}"><circle cx="{x}" cy="{y}" r="29" fill="{PALE if i else BLUE}" stroke="{BLUE}" stroke-width="2"/><text x="{x}" y="{y+7}" text-anchor="middle" font-size="21" fill="{INK if i else "white"}">{("r","a","b","c")[i]}</text></g>')
    out.extend([card(key,760,120,'Finite grammar',['ADD_CHILD or CLOSE','Given root r is clamped','Bounded support only'],'An ADD creates one new child at the stack top; CLOSE finishes the current node. Children and roots retain a canonical order.',280,145),
        '<text x="30" y="365" font-size="16">Trace: ADD → ADD → CLOSE → CLOSE → ADD → CLOSE → CLOSE</text>',
        f'<text x="30" y="404" font-size="14" fill="{AMBER}">N = 4, R = 1: exactly 2N − R = 7 decisions. Illustration, not a sampled model result.</text></g></svg>'])
    topology=''.join(out)
    key='directed';out=svg_start(key,'Directed squares preserve more than reachability','Two typed parallel routes become equivalent only under an explicit declared square relation.',height=450)
    xy=[(100,150),(410,150),(100,315),(410,315)]
    for u,v,label in((0,1,'f'),(0,2,'g'),(1,3,'h'),(2,3,'k')):
        x,y=xy[u];a,b=xy[v]
        out.append(arrow(key,x+28 if a>x else x,y+28 if b>y else y,a-28 if a>x else a,b-28 if b>y else b,label))
    for (x,y),label in zip(xy,('A','B','C','D')):
        out.append(f'<circle cx="{x}" cy="{y}" r="26" fill="{PALE}" stroke="{BLUE}"/><text x="{x}" y="{y+7}" text-anchor="middle" font-size="22">{label}</text>')
    out.extend([f'<rect x="175" y="192" width="175" height="80" fill="#edf3f6" stroke="{LINE}" stroke-dasharray="6 5"/><text class="square-label" x="262" y="225" text-anchor="middle" font-size="17">declared square</text><text class="square-equivalence" x="262" y="250" text-anchor="middle" font-size="17">h ∘ f ~ k ∘ g</text>',
        card(key,565,125,'Quotient functor',['Objects → objects','Generators → directed paths','Relations → checked witnesses'],'Endpoint-compatible generator maps extend to a quotient functor exactly when every defining relation maps to an equivalent target pair.',470,135),
        f'<text x="565" y="300" font-size="16">No reverse path is inserted automatically.</text><text x="565" y="329" font-size="16">No witness / exhausted search budget → unknown.</text>',
        f'<text x="30" y="410" font-size="14" fill="{AMBER}">A finite relation certificate is not a proof of continuous realization or biological causality.</text></g></svg>'])
    return {'architecture':architecture,'topology':topology,'refinement':refinement,
            'directed':''.join(out),'topology-metrics':metrics,'campaign':campaign}


def results_schema():
    proxy=[]
    for trial in planned_proxy_trials():
        proxy.append(dict(trial_id=trial.trial_id,hp_id=trial.hp_id,learning_rate=str(trial.learning_rate),
            readout_multiplier=str(trial.readout_multiplier),attention_key_multiplier=str(trial.attention_key_multiplier),
            seed=trial.seed,terminal_status=None,updates=None,validation_score=None,gpu_seconds=None,artifact_sha256=None))
    return dict(schema='Mora3.UnpopulatedEmpiricalPaperResults/v1',populated=False,training_started=False,
        meaning_of_null='Unpopulated, not zero and not a terminal disposition',proxy=proxy,
        targets=[dict(seed=s,width=None,unique_parameters=None,transferred_hp=None,status=None,checkpoint=None)for s in(2026111001,2026111002)],
        modalities=[dict(task=t,metric=m,eligible_groups=None,score=None,uncertainty=None,checkpoint=None)for t,m in(
            ('Molecular ADMET','Task-specific error / AUROC / calibration'),('Protein design','Fold/sequence consistency and native criteria'),
            ('Reaction routes','Complete route validity / source-supported outcome'),('Single-cell perturbation','Native gene-axis effect / control-aware error'),
            ('Image / tissue morphology','Axis-preserving structural outcome'),('Bioelectric / SCRI','Time/cell-native masked error and constraints'))],
        reasoning=[dict(task=t,metric=m,value=None,uncertainty=None,checkpoint=None)for t,m in(
            ('Topology generation','Complete source-topology NLL'),('Whole-trajectory analogy','Supported correspondence and downstream outcome'),
            ('Multiple outputs','Distinct qualified outputs / requested K'),('Flow refinement','Native decoded quality at matched compute'),
            ('Calibration / abstention','Reliability and unsupported-output rate'),('Directed relations','Checked path/square preservation'))],
        resources=[dict(scope=t,seconds=None,peak_memory=None,updates=None,evidence=None)for t in(
            'Complete proxy update','Matched validation exposure','Full target update','Completed-update recovery','Prefix/topology inference','Exact persistence-distance rerank')],
        charts=[dict(id=k,title=t,x=x,y=y,series=[])for k,t,x,y in(
            ('learning','Proxy learning curves','Completed native update','Frozen validation score'),
            ('transfer','Proxy-to-target transfer','Frozen hyperparameter point','Validation-selected comparison'),
            ('quality','Quality and inference cost','Total model evaluations / time','Qualified native output metric'),
            ('resources','Resource profile','Declared model width / task','Peak memory / elapsed time'))])


def result_table(rows,columns,label):
    parts=[f'<div class="result-table" role="region" aria-label="{escape(label)}" tabindex="0"><table><caption>{escape(label)} — result cells intentionally empty</caption><thead><tr>']
    parts.extend('<th scope="col">'+escape(title)+'</th>'for _,title in columns);parts.append('</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>')
        for key,_ in columns:
            value=row[key]
            parts.append('<td class="unpopulated" data-result="null" aria-label="Unpopulated"></td>'if value is None else'<td>'+escape(value)+'</td>')
        parts.append('</tr>')
    return ''.join(parts)+'</tbody></table></div>'


def blank_chart(chart):
    key='empty-'+chart['id'];out=svg_start(key,chart['title'],'Empty empirical chart. No observations, curves, bars or confidence intervals have been supplied.',width=960,height=350)
    out.extend([f'<path d="M90 100 V275 H900" fill="none" stroke="{MUTED}" stroke-width="1.5"/>',
        f'<text x="495" y="319" text-anchor="middle" font-size="16">{escape(chart["x"])}</text>',
        f'<text x="28" y="188" transform="rotate(-90 28 188)" text-anchor="middle" font-size="15">{escape(chart["y"])}</text>',
        f'<text x="495" y="180" text-anchor="middle" font-size="22" fill="{MUTED}">Awaiting qualified empirical evidence</text>',
        f'<text x="495" y="210" text-anchor="middle" font-size="14" fill="{MUTED}">No data marks or inferred zero values</text></g></svg>'])
    return ''.join(out).replace('Architecture / mathematical illustration — not an empirical result','Reserved empirical panel — no measurements populated')


def render_math(expressions,destination,work):
    destination.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for i,expr in enumerate(expressions,1):
        if re.search(r'\\(?:input|include|write|openout|read|catcode|csname|usepackage|documentclass|special|immediate)\b',expr):
            raise ValueError('Unsupported command in source math block')
        pin=sha256(expr.encode()).hexdigest();item=work/('equation-'+str(i));item.mkdir(exist_ok=True)
        tex=(r'\documentclass[10pt]{article}'+'\n'+r'\usepackage{amsmath,amssymb,lmodern}'+'\n'+r'\pagestyle{empty}\begin{document}'+'\n'
             +r'\setbox0=\hbox{$\displaystyle '+expr+r'$}'+'\n'
             +r'\pdfpagewidth=\dimexpr\wd0+12pt\relax'+'\n'+r'\pdfpageheight=\dimexpr\ht0+\dp0+12pt\relax'+'\n'
             +r'\hoffset=-1in\voffset=-1in'+'\n'+r'\shipout\vbox{\kern6pt\hbox{\kern6pt\box0}\kern6pt}'+'\n'+r'\end{document}'+'\n')
        (item/'equation.tex').write_text(tex)
        run=subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','equation.tex'],cwd=item,capture_output=True,text=True,timeout=30)
        (item/'build.stdout').write_text(run.stdout);(item/'build.stderr').write_text(run.stderr)
        if run.returncode:raise RuntimeError('Math compilation failed at block '+str(i)+': '+run.stdout[-1800:])
        target=destination/f'equation-{i:02d}.svg'
        subprocess.run(['pdftocairo','-svg',str(item/'equation.pdf'),str(target)],check=True,capture_output=True,text=True,timeout=20)
        tree=ET.fromstring(target.read_text());width=float(tree.attrib['width'].removesuffix('pt'));height=float(tree.attrib['height'].removesuffix('pt'))
        outputs.append(dict(number=i,source=expr,source_sha256=pin,path=target.name,width_pt=width,height_pt=height))
    return outputs


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Continuation-owned output directory required')
    source=ROOT/'paper/mora3_complete_methods.md';text=source.read_text()
    if len(re.findall(r'^### (?:Theorem|Proposition) \d+\.',text,re.M))!=12:raise ValueError('All twelve scoped claims required')
    expressions=re.findall(r'```math\n(.*?)\n```',text,re.S)
    figures=diagrams();data=results_schema();assert len(data['proxy'])==24
    for root in(ROOT/'paper/figures',ROOT/'docs/project/figures'):
        root.mkdir(parents=True,exist_ok=True)
        for key,svg in figures.items():(root/f'methods-{key}.svg').write_text(svg)
        for chart in data['charts']:(root/('methods-empty-'+chart['id']+'.svg')).write_text(blank_chart(chart))
    (ROOT/'paper/empirical_results_template.json').write_text(wire(data)+'\n')
    (ROOT/'docs/project/empirical_results_template.json').write_text(wire(data)+'\n')
    equations=render_math(expressions,ROOT/'docs/project/equations',output/'equations')
    for row in equations:
        path=ROOT/'docs/project/equations'/row['path'];dest=ROOT/'paper/equations'/row['path'];dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
    def math_replace(match):
        index=expressions.index(match.group(1));eq=equations[index]
        return '<div class="equation" role="group" aria-label="Equation '+str(eq['number'])+'"><img style="width:'+str(eq['width_pt']*1.5)+'px" src="project/equations/'+eq['path']+'" alt="'+escape(eq['source'])+'"><span class="equation-number">('+str(eq['number'])+')</span><details class="math-source"><summary>Equation source</summary><pre>'+escape(eq['source'])+'</pre></details></div>'
    text=re.sub(r'```math\n(.*?)\n```',math_replace,text,flags=re.S)
    replacements={
        'PROXY':result_table(data['proxy'],(('trial_id','Trial'),('learning_rate','Rate'),('readout_multiplier','Readout'),('attention_key_multiplier','Attention'),('terminal_status','Disposition'),('updates','Updates'),('validation_score','Validation'),('gpu_seconds','GPU s')),'All 24 original proxy slots'),
        'TARGET':result_table(data['targets'],(('seed','Target seed'),('width','Width'),('unique_parameters','Parameters'),('transferred_hp','Transferred HP'),('status','Status'),('checkpoint','Checkpoint')),'Two target runs and actual transfer'),
        'MODALITY':result_table(data['modalities'],(('task','Task family'),('metric','Predeclared metric'),('eligible_groups','Groups'),('score','Score'),('uncertainty','Uncertainty'),('checkpoint','Checkpoint')),'Native multimodal evaluation'),
        'REASONING':result_table(data['reasoning'],(('task','Reasoning task'),('metric','Metric'),('value','Value'),('uncertainty','Uncertainty'),('checkpoint','Checkpoint')),'Reasoning and structural evaluation'),
        'RESOURCE':result_table(data['resources'],(('scope','Measured scope'),('seconds','Seconds'),('peak_memory','Peak memory'),('updates','Updates'),('evidence','Evidence')),'Resources and exact recovery')}
    for key,value in replacements.items():
        marker='<!-- MORA_EMPTY_'+key+'_TABLE -->';assert text.count(marker)==1;text=text.replace(marker,value)
    text=text.replace('<!-- MORA_EMPTY_CHARTS -->','\n'.join('<figure class="empty-chart" data-series-count="0"><img src="project/figures/methods-empty-'+c['id']+'.svg" alt="'+escape(c['title']+': intentionally empty; no measured values')+'"><figcaption>Reserved panel. No observations are plotted.</figcaption></figure>'for c in data['charts']))
    def fig_replace(match):
        caption,key=match.group(1),match.group(2)
        if key not in figures:raise ValueError('Unknown manuscript diagram '+key)
        return '<figure class="method-diagram" id="diagram-'+key+'"><div class="diagram-viewport">'+figures[key]+'</div><figcaption>'+escape(caption)+'</figcaption><div class="diagram-controls">'+('<button type="button" class="step-topology">Next construction prefix</button><button type="button" class="reset-topology">Reset</button>'if key=='topology'else'<button type="button" class="toggle-square" aria-pressed="true">Toggle declared square relation</button>'if key=='directed'else'')+'<button type="button" class="export-diagram">Export SVG</button></div><p class="diagram-detail" aria-live="polite">Select a diagram component for its exact scope. This is not empirical evidence.</p></figure>'
    text=re.sub(r'!\[([^\]]+)\]\(figures/methods-([a-z-]+)\.svg\)',fig_replace,text)
    # Visible primary-source links remain links, without fetching network assets.
    text=re.sub(r'(?m)(https://\S+)$',r'<\1>',text)
    body=markdown.markdown(text,extensions=['tables','fenced_code','toc','sane_lists'],extension_configs={'toc':{'toc_depth':'2-3'}})
    body=re.sub(r'<h3 id="([^"]+)">((?:Theorem|Proposition) (\d+)\.[^<]*)</h3>',r'<h3 class="formal-claim" id="\1" data-claim="\3">\2</h3>',body)
    body=re.sub(r'<p><strong>\[R(\d+)\]</strong>',r'<p class="reference" id="ref-R\1"><strong>[R\1]</strong>',body)
    body=re.sub(r'(?<!>)\[R(\d+)([^\]]*)\]',lambda m:'<a class="citation" href="#ref-R'+m.group(1)+'">[R'+m.group(1)+escape(m.group(2))+']</a>',body)
    aliases={'methods':'3. Native records','flow-forest-methods':'5. Continuous forest flow',
        'persistence-vectors':'9. Geometry, persistent','native-forest-heads':'7. Finite token',
        'consolidation-20261113':'Appendix B.', 'grh-research-integration':'3.3 Complete source-prefix',
        'evidence':'12. Empirical reporting','flow-fot-refinement':'6.2 Candidate generation',
        'forest-training':'7.2 Masked joint','limits':'13. Limitations','refs':'References',
        'source-forest-exports':'3.3 Complete source-prefix'}
    headings=re.findall(r'<h[23] id="([^"]+)">([^<]+)</h[23]>',body)
    for alias,prefix in aliases.items():
        matches=[identity for identity,title in headings if title.startswith(prefix)]
        if len(matches)!=1:raise ValueError('Legacy paper anchor has no unique destination: '+alias)
        target=matches[0]
        body=re.sub(r'(<h[23] id="'+re.escape(target)+r'">)',
                    '<span class="legacy-anchor" id="'+alias+'"></span>'+r'\1',body,count=1)
    toc=markdown.Markdown(extensions=['toc']);toc.convert(source.read_text());contents=toc.toc
    page=HTML.replace('__STYLE__',CSS).replace('__TOC__',contents).replace('__BODY__',body).replace('__DATA__',wire(data).replace('<','\\u003c')).replace('__SCRIPT__',SCRIPT)
    old=ROOT/'docs/paper.html';backup=output/'before/docs/paper.html';backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_bytes(old.read_bytes())
    old.write_text(page)
    (ROOT/'paper/mora3_preprint.md').write_text(source.read_text())
    tracked=['src/mora3/model/forest_topology.py','src/mora3/reasoning/forest_topology.py',
        'src/mora3/model/forest_training.py','src/mora3/model/full_owner.py','src/mora3/model/full_owner_mup.py',
        'src/mora3/research/forest_training_routes.py','src/mora3/research/primary_training.py',
        'src/mora3/research/mup_sweep_selection.py','src/mora3/reasoning/persistence_vectors.py',
        'src/mora3/reasoning/directed_topology.py','src/mora3/data/forest_source_exports.py',
        'paper/mora3_complete_methods.md','tools/build_complete_methods_paper.py']
    manifest=dict(schema='Mora3.CompleteMethodsPaperBuild/v1',source_sha256={n:sha256((ROOT/n).read_bytes()).hexdigest()for n in tracked},
        figures=list(figures),math_equations=equations,formal_claims=12,empirical_tables=5,empty_charts=4,
        empirical_values_populated=False,training_started=False,public_push_performed=False,
        source_word_count=len(source.read_text().split()),primary_references=18,
        publication_condition='reviewed-tested-source-checkpoints-authorized-independently-of-training',book_redistributed=False)
    (ROOT/'paper/methods_build_manifest.json').write_text(wire(manifest)+'\n')
    (ROOT/'docs/project/methods_build_manifest.json').write_text(wire(manifest)+'\n')
    print(json.dumps({k:v for k,v in manifest.items()if k not in('source_sha256','math_equations')},indent=2),flush=True)
    return manifest


CSS='''
:root{color-scheme:light;--ink:#182637;--muted:#536477;--blue:#175c7b;--line:#d5dee6;--paper:#fff;--soft:#f3f6f8;--amber:#94531e}*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;color:var(--ink);background:#f4f6f8;font:17px/1.7 Georgia,'Times New Roman',serif}a{color:var(--blue);text-underline-offset:3px}.topbar{font:14px/1.4 system-ui,sans-serif;background:var(--ink);color:#fff;padding:18px 28px;display:flex;gap:22px;flex-wrap:wrap}.topbar a{color:#fff}.layout{max-width:1420px;margin:auto;display:grid;grid-template-columns:265px minmax(0,1fr);gap:38px;padding:32px 28px 80px}aside{font:13px/1.5 system-ui,sans-serif;position:sticky;top:18px;align-self:start;max-height:90vh;overflow:auto;padding-right:14px}aside ul{list-style:none;padding-left:10px}aside>details>summary{font-weight:700}aside a{color:var(--muted);text-decoration:none}aside li{margin:7px 0}article{max-width:100%;min-width:0;background:var(--paper);padding:45px 48px;box-shadow:0 1px 7px #18263712}h1,h2,h3,h4{font-family:system-ui,sans-serif;line-height:1.25;scroll-margin-top:22px}h1{font-size:37px;letter-spacing:-.035em;margin:0 0 25px}h2{font-size:27px;border-top:1px solid var(--line);padding-top:27px;margin-top:48px}h3{font-size:20px;margin-top:31px}p{margin:14px 0}code,pre{font:13px/1.5 ui-monospace,monospace}code{background:var(--soft);padding:1px 4px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}a,td,th{overflow-wrap:anywhere}.formal-claim{border-left:4px solid var(--blue);padding:12px 16px;background:var(--soft);font-size:19px}.citation{font-size:.9em;white-space:nowrap}.reference{font-size:14px}.equation{margin:25px 0;position:relative;padding:8px 32px 8px 4px;text-align:center;min-width:0;overflow:auto}.equation img{display:inline-block;max-width:100%;height:auto;vertical-align:middle}.equation-number{position:absolute;right:0;top:16px;font-size:13px;color:var(--muted)}.math-source{font:12px/1.5 system-ui,sans-serif;text-align:left;margin-top:7px;color:var(--muted)}figure{margin:30px 0;padding:15px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);break-inside:avoid}figcaption{font:13px/1.55 system-ui,sans-serif;color:var(--muted);margin:12px 0}figure img{max-width:100%;height:auto}.diagram-viewport{overflow:auto;width:100%}.diagram-viewport svg{display:block;width:100%;height:auto}.diagram-node{cursor:pointer}.diagram-node:focus{outline:none}.diagram-node:focus rect,.diagram-node.active rect{stroke:var(--amber);stroke-width:3}.diagram-controls{display:flex;gap:10px;flex-wrap:wrap;font-family:system-ui,sans-serif}.diagram-controls button{font:13px system-ui,sans-serif;min-height:42px;border:1px solid #b8c6d0;border-radius:5px;background:white;padding:9px 12px;cursor:pointer;color:var(--blue)}.diagram-detail{font:13px/1.55 system-ui,sans-serif;background:var(--soft);padding:12px 15px;min-height:62px}.result-table{overflow:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font:12px/1.45 system-ui,sans-serif;min-width:620px}caption{text-align:left;font-weight:700;padding:10px 0}thead{display:table-header-group}th{padding:10px 7px;background:#edf2f5;color:var(--ink);text-align:left;border-top:1px solid #a7b8c7;border-bottom:1px solid #a7b8c7}td{padding:9px 7px;min-height:28px;border-bottom:1px solid #dbe3e9}.unpopulated{min-width:64px;height:33px;background:repeating-linear-gradient(135deg,#fff,#fff 8px,#fafbfc 8px,#fafbfc 9px)}.empty-chart img{width:100%;height:auto}.paper-tools{font:13px system-ui,sans-serif;padding:14px 18px;background:#edf2f5;border-left:4px solid var(--blue);margin:0 0 27px}.paper-tools a{margin-right:17px}.footer{font:12px system-ui,sans-serif;color:var(--muted);margin-top:40px}button:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid var(--amber);outline-offset:3px}@media(max-width:1050px){.layout{grid-template-columns:1fr;padding:18px 15px;gap:15px}aside{position:static;max-height:250px}aside details:not([open]){height:auto}article{padding:30px 25px}.diagram-viewport svg{min-width:750px}.result-table{max-width:100%}}@media(max-width:600px){body{font-size:16px}.topbar{padding:16px;gap:15px}article{padding:25px 18px}h1{font-size:29px}h2{font-size:24px}h3{font-size:19px}.diagram-controls button{min-height:44px}.equation{padding-right:24px}.formal-claim{padding:10px 12px}.diagram-detail{min-height:0}.result-table table{font-size:11px}.equation img{min-width:0}.layout{padding:12px 8px}}
@page{size:A4;margin:18mm 17mm 20mm} @media print{.topology-node,.topology-edge{visibility:visible!important}.live-prefix{display:none!important}body{background:#fff;font-size:10.5pt;line-height:1.5;color:#182637}.topbar,aside,.paper-tools,.diagram-controls,.diagram-detail,.math-source{display:none!important}.layout{display:block;padding:0;max-width:none}article{padding:0;box-shadow:none;width:auto;max-width:none}h1{font-size:24pt;line-height:1.18}h2{font-size:16pt;margin-top:24pt;padding-top:14pt;break-after:avoid}h3{font-size:12.5pt;margin-top:18pt;break-after:avoid}p{margin:8pt 0;orphans:3;widows:3}figure{margin:17pt 0;padding:9pt 0;break-inside:avoid}.diagram-viewport{overflow:visible}.diagram-viewport svg{min-width:0;width:100%;height:auto}figcaption{font-size:8.5pt}table{font-size:8pt;min-width:0;width:100%}th,td{padding:5pt;min-width:0!important}.unpopulated{background:#fff;height:23pt}.result-table{overflow:visible;break-inside:auto}tr{break-inside:avoid}.equation{overflow:visible;padding:5pt 20pt 5pt 0;break-inside:avoid;margin:12pt 0}.equation img{max-width:100%;width:88%!important;max-height:52pt;object-fit:contain}.equation-number{font-size:8pt;top:10pt}.reference{font-size:9pt;line-height:1.35;margin:6pt 0}.formal-claim{font-size:12pt;border-left:2pt solid #175c7b;padding:8pt 10pt}.empty-chart{max-height:220pt}.empty-chart img{max-height:190pt;object-fit:contain}a{color:inherit;text-decoration:none}code{font-size:8pt}.footer{font-size:8pt}}
'''

HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Mora-3 complete methods and scoped theory, with interactive diagrams and intentionally empty empirical results."><title>Mora-3 · Complete methods and theory</title><style>__STYLE__</style></head><body>
<nav class="topbar" aria-label="Project navigation"><a href="index.html">Mora-3 project</a><a href="mora3_methods.pdf">Print paper PDF</a><a href="project/source-forest-exports.html">Source-prefix inspector</a><a href="project/native-forest-heads.html">Native forest heads</a><a href="project/persistence-vectors.html">Persistence L2 lab</a></nav>
<div class="layout"><aside><details open><summary>Manuscript contents</summary>__TOC__</details></aside><article id="main"><div class="paper-tools"><strong>Methods / theory manuscript. Empirical results are unpopulated.</strong><p><a href="mora3_methods.pdf">PDF with static vector figures</a><a href="project/empirical_results_template.json">Empty result schema</a><a href="project/methods_build_manifest.json">Construction/source manifest</a></p></div>__BODY__<p class="footer">Offline-first manuscript. Diagram interactions are explanatory; they are not training or evaluation runs. Reviewed source publication is authorized independently of training; empirical claims require qualified evidence.</p></article></div>
<script id="paper-results" type="application/json">__DATA__</script><script>__SCRIPT__</script></body></html>'''

SCRIPT=r'''
'use strict';const NS='http://www.w3.org/2000/svg';
function saveDiagram(figure){let svg=figure.querySelector('svg').cloneNode(true);svg.querySelectorAll('[tabindex]').forEach(x=>{x.removeAttribute('tabindex');x.removeAttribute('role');});const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}));const a=document.createElement('a');a.href=url;a.download=figure.id+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const figure of document.querySelectorAll('.method-diagram')){
 const detail=figure.querySelector('.diagram-detail');
 for(const node of figure.querySelectorAll('.diagram-node')){const select=()=>{figure.querySelectorAll('.diagram-node').forEach(n=>n.classList.remove('active'));node.classList.add('active');detail.textContent=node.dataset.detail;};node.addEventListener('click',select);node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});}
 figure.querySelector('.export-diagram').addEventListener('click',()=>saveDiagram(figure));
}
const topo=document.getElementById('diagram-topology'),actions=[1,1,0,0,1,0,0];let position=0;
function topologyState(n){let parents=[-1],stack=[0],closed=[];for(let i=0;i<n;i++){if(actions[i]===1){parents.push(stack.at(-1));stack.push(parents.length-1);}else{closed.push(stack.pop());}}return{parents,stack,closed};}
function showTopology(){const s=topologyState(position);topo.querySelectorAll('.topology-node').forEach(n=>n.style.visibility=Number(n.dataset.index)<s.parents.length?'visible':'hidden');topo.querySelectorAll('.topology-edge').forEach(n=>n.style.visibility=Number(n.dataset.target)<s.parents.length?'visible':'hidden');topo.dataset.step=String(position);topo.querySelector('.diagram-detail').textContent='Mathematical example — prefix '+position+'/7. Present parents: ['+s.parents.join(', ')+']. Stack: ['+s.stack.join(', ')+']. Closed: ['+s.closed.join(', ')+']. '+(position===7?'Complete ordered tree; no further action.':'Only constructed nodes appear below; future children are not policy inputs.');let old=topo.querySelector('.live-prefix');if(old)old.remove();const line=document.createElementNS(NS,'text');line.classList.add('live-prefix');line.setAttribute('x','30');line.setAttribute('y','100');line.setAttribute('font-family','Arial,sans-serif');line.setAttribute('font-size','16');line.setAttribute('fill','#175c7b');line.textContent='Current prefix '+position+': parents ['+s.parents.join(', ')+'] | stack ['+s.stack.join(', ')+']';topo.querySelector('svg').append(line);topo.querySelector('.step-topology').disabled=position===7;}
topo.querySelector('.step-topology').addEventListener('click',()=>{position=Math.min(7,position+1);showTopology();});topo.querySelector('.reset-topology').addEventListener('click',()=>{position=0;showTopology();});showTopology();
const directed=document.getElementById('diagram-directed'),toggle=directed.querySelector('.toggle-square');
toggle.addEventListener('click',()=>{const enabled=toggle.getAttribute('aria-pressed')!=='true';toggle.setAttribute('aria-pressed',String(enabled));directed.dataset.square=String(enabled);directed.querySelector('.diagram-detail').textContent=enabled?'Declared relation enabled: h∘f and k∘g are identified by the supplied square. This is a finite-presentation example, not empirical biological evidence.':'No square relation declared: the two routes have the same endpoints but are not identified by this presentation. Equal endpoints alone do not prove a trajectory analogy.';directed.querySelector('.square-label').textContent=enabled?'declared square':'no square declared';directed.querySelector('.square-equivalence').textContent=enabled?'h ∘ f ~ k ∘ g':'distinct path words';directed.querySelectorAll('rect[stroke-dasharray]').forEach(x=>x.setAttribute('opacity',enabled?'1':'.15'));});
const results=JSON.parse(document.getElementById('paper-results').textContent);
if(results.populated!==false||results.charts.some(c=>c.series.length))throw new Error('The unmeasured paper cannot contain empirical series');
document.documentElement.dataset.paperReady='true';
'''


def print_paper(output):
    from playwright.sync_api import sync_playwright
    output=Path(output).resolve();folder=output/'print';folder.mkdir(parents=True,exist_ok=True)
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Continuation-owned paper output required')
    manifest=json.loads((ROOT/'paper/methods_build_manifest.json').read_text())
    if any(sha256((ROOT/name).read_bytes()).hexdigest()!=pin for name,pin in manifest['source_sha256'].items()):
        raise ValueError('Methods source changed after generation')
    with sync_playwright()as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,
            args=['--no-sandbox','--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((ROOT/'docs/paper.html').as_uri())
        page.wait_for_function("document.documentElement.dataset.paperReady==='true'")
        if not page.evaluate('[...document.images].every(i=>i.complete&&i.naturalWidth>0)'):
            raise ValueError('Every mathematical vector asset must render before printing')
        page.emulate_media(media='print')
        import inspect
        options={k:True for k in ('tagged','outline')if k in inspect.signature(page.pdf).parameters}
        page.pdf(path=str(folder/'mora3_methods.pdf'),format='A4',print_background=True,**options,
            prefer_css_page_size=True,display_header_footer=True,
            header_template='<div style="font-size:8px;width:100%;padding:0 65px;color:#536477;">MORA-3 · METHODS AND SCOPED THEORY · EMPIRICAL RESULTS UNPOPULATED</div>',
            footer_template='<div style="font-size:9px;width:100%;padding:0 65px;color:#536477;display:flex;justify-content:space-between;"><span>Working manuscript · 14 November 2026</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
        browser.close()
    if errors:raise ValueError('Paper has browser errors: '+str(errors))
    raw=(folder/'mora3_methods.pdf').read_bytes()
    for path in(ROOT/'paper/mora3_methods.pdf',ROOT/'docs/mora3_methods.pdf'):path.write_bytes(raw)
    info=subprocess.run(['pdfinfo',str(folder/'mora3_methods.pdf')],capture_output=True,text=True,check=True,timeout=15).stdout
    subprocess.run(['pdftotext','-layout',str(folder/'mora3_methods.pdf'),str(folder/'mora3_methods.txt')],check=True,timeout=15)
    result=dict(schema='Mora3.MethodsPaperPrint/v1',pdf_sha256=sha256(raw).hexdigest(),pdf_bytes=len(raw),
        pdfinfo=info,empirical_values_populated=False,training_started=False,public_push_performed=False)
    (folder/'print-manifest.json').write_text(wire(result)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args();build(args.output)
    if args.pdf:print_paper(args.output)
