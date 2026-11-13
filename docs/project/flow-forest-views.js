/* Source-backed SVG architecture/math views. All numeric illustrations are fixtures. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory(require('./flow-forest-math.js'));
  else root.MoraViews=factory(root.MoraMath);
})(typeof globalThis!=='undefined'?globalThis:this,function(M){
 'use strict';
 const C={ink:'#193747',muted:'#526672',line:'#c5d0d2',teal:'#0b756b',blue:'#206a95',amber:'#a15b15',red:'#a03743',paper:'#ffffff',pale:'#f0f6f4',sand:'#fbf4e8',purple:'#635394'};
 const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
 const text=(x,y,s,size=15,fill=C.ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
 const line=(x1,y1,x2,y2,color=C.line,width=2,dash='',arrow=false)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dash?`stroke-dasharray="${dash}"`:''} ${arrow?'marker-end="url(#arrow)"':''}/>`;
 const box=(x,y,w,h,title,sub,kind='planned')=>{
   const col=kind==='existing'?C.teal:kind==='blocked'?C.amber:C.blue;
   return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${kind==='blocked'?C.sand:C.paper}" stroke="${col}" stroke-width="1.6" ${kind==='planned'?'stroke-dasharray="5 4"':''}/>`+
    text(x+15,y+25,title,16,C.ink,650)+text(x+15,y+48,sub,12.5,C.muted)+`<circle cx="${x+w-17}" cy="${y+18}" r="4" fill="${col}"/>`;
 };
 const chip=(x,y,w,label,color=C.teal)=>`<rect x="${x}" y="${y}" width="${w}" height="25" rx="5" fill="${color}" fill-opacity=".09"/>`+text(x+9,y+17,label,12,color,600);
 function shell(title,desc,body,height=500){const uid="figure-"+title.toLowerCase().replace(/[^a-z0-9]+/g,"-");return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="${uid}-title ${uid}-desc" viewBox="0 0 960 ${height}" width="960" height="${height}"><title id="${uid}-title">${esc(title)}</title><desc id="${uid}-desc">${esc(desc)}</desc><defs><marker id="arrow" markerWidth="9" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M1,1 L7,4 L1,7" fill="none" stroke="${C.muted}" stroke-width="1.3"/></marker></defs><rect width="960" height="${height}" fill="white"/><g font-family="Arial, Helvetica, sans-serif">${text(25,31,title,20,C.ink,650)}${body}</g></svg>`;}
 // SVG coordinates are presentation values, not analysis results. Nine decimal
 // places make exports reproducible across JS runtimes whose Math.exp may differ
 // at the last binary bit; error is at most 5e-10 viewBox units. Numeric science
 // outputs remain untouched. Limit normalization to geometry attributes.
 function canonicalSvgCoordinates(svg){
   return svg.replace(/\b(x|y|x1|y1|x2|y2|cx|cy|r|rx|ry|width|height|points|d)="([^"]*)"/g,
     (match,name,value)=>name+'="'+value.replace(/[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?/g,
       token=>String(Number(Number(token).toFixed(9))))+'"');
 }
 const META={
  architecture:{name:'Architecture',eyebrow:'01 · SYSTEM DESIGN',title:'Generate a forest. Preserve its meaning.',
   takeaway:'A topology policy, continuous flow, retrieval, and final decoder have distinct responsibilities. The new system must preserve the existing shared owner.',
   equation:'F = (T, Z, A)    ·    π*(F | c) ∝ R(F, c)',
   caveat:'Dashed boxes are proposed components, not implemented neural capabilities. Existing component tests do not establish full-model readiness.',
   detail:['Topology is discrete. A fixed-dimensional ODE does not choose arbitrary branches or length.','Condition-only retrieval proposes complete exemplars; directed and domain checks may abstain.','GFlowNet transition accounting and flow transport are separate mathematical objects.','All new axes, parameters, token rows and objectives must be frozen before the applicable μP campaign.'],
   source:'Plan §§2–5; Flow Matching; continuous GFN theory.'},
  forest:{name:'Forest flow',eyebrow:'02 · TWO TIMES, ONE OBJECT',title:'Transport the entire latent forest—not one token at a time.',
   takeaway:'Move solver time while all eight root-to-terminal paths remain present. Two conditioning roots are clamped; twelve non-root states are transported.',
   equation:'Zₛ = (1 − s) ε + s Z₁    ·    d = reasoning depth, s = solver time',
   caveat:'This is target-aware analytic interpolation of a fixed 2D fixture. It is not a learned sampler, a molecular geometry, or a speed benchmark.',
   detail:['Two trees share no root with each other; shared prefixes within each tree are stored once.','The reference path exposes known targets for teaching. A deployed learned velocity field cannot read clean target forests.','The probability bars normalize four finite forest rewards with an explicit nonuniform base mass.','A proper forest policy must also account for topology and construction-history multiplicity.'],
   source:'Plan §§2–4; existing gaussian_path reference; exact fixture calculations.'},
  topology:{name:'Analogy & topology',eyebrow:'03 · ORDER IS INFORMATION',title:'Equal endpoints do not make two arguments equivalent.',
   takeaway:'Toggle the target relation: a declared square identifies two directed paths; without it, the same mapping fails. Metric persistence is a separate descriptor.',
   equation:'F(b) ∘ F(a) = F(d) ∘ F(c)    only when a defining relation is supported',
   caveat:'The square is supplied evidence in a finite presentation. Neither it nor equal persistence barcodes establishes semantic or biological causal equivalence.',
   detail:['Reverse traversal is not permitted merely because an underlying undirected route exists.','The left example checks one relation; production analogy checks every required intermediate step and complete branch.','The right example is exact F₂ Rips H₀/H₁ of a unit square. At √2, diagonals and triangles kill its H₁ class.','Physical orientation, chirality and computational direction are distinct; distance-only topology cannot distinguish mirror images.'],
   source:'Grandis, printed pp.2–3 and §§1.4,2.1,3.2; Topological Autoencoders.'},
  decode:{name:'Decode & tokens',eyebrow:'04 · CONTINUOUS TO DISCRETE',title:'Nearest is an initializer. Valid is a stronger requirement.',
   takeaway:'Move the proposed latent token. Unrestricted nearest projection may select an inadmissible evidence label; the typed hypothesis grammar restricts the final choice.',
   equation:'token = argminₖ∈allowed(role) ‖z − eₖ‖²    ·    mass(token) = ∫cell(token) p(z) dz',
   caveat:'The six displayed vectors are a schematic codebook, not trained embeddings. A generated SOURCE_OBSERVED token does not confer source authority.',
   detail:['The current payload pooling is lossy. A learned token-slot decoder and aligned codebook are additional requirements.','After projection, discrete probability mass is integrated over decoding regions; latent reward proportionality is not automatically retained.','Forty proposed controls are reserved outside ConvexTok’s learned byte-token budget. IDs are not allocated here.','Appending controls changes metadata offsets and checkpoint/token-axis bindings. No current vocabulary is modified.'],
   source:'Plan §6; existing PayloadEncoder/ConvexTok contracts; Jev typed-output interface, not a claimed Jev token list.'},
  fewstep:{name:'Few-step maps',eyebrow:'05 · REDUCE SOLVER WORK, NOT REASONING',title:'One solver step can still represent a deep forest.',
   takeaway:'The analytic ODE below shows why coarse Euler updates lose accuracy and why an interval flow map can skip intermediate solver points. Learned maps need real validation.',
   equation:'M(x,s,t) ≈ solution at t    ·    M(M(x,s,t),t,u) ≈ M(x,s,u)',
   caveat:'Exact x(s)=exp(s) is an oracle for this scalar fixture—not a trained one-step Mora-3 result. No runtime or biological-quality improvement is measured.',
   detail:['The selected investigation is DMF-informed conversion of a trained native flow, not changing Euler steps and calling it distillation.','Initial native teacher-pair regression avoids presuming JVP support in first-order tropical kernels.','Evaluate 1,2,4 solver steps with the same full forest and held-out problems; record all guidance, retrieval and decoding work.','An additional distillation training run exceeds the current 24+2 allocation and remains budget-blocked.'],
   source:'Plan §7; MeanFlow; Decoupled MeanFlow; Align Your Flow; MFD screened alternative.'},
  quality:{name:'Quality & adapters',eyebrow:'06 · CONTEXTUAL PREFERENCE, NOT BLIND SUPPRESSION',title:'A steering direction is not globally good or bad.',
   takeaway:'Compare matched contexts, retain uncertainty and exclude unsupported labels. DoRA is a qualification candidate; quantization is a measured trade-off, not a synonym for quality.',
   equation:'z′ = z + α dₖ    ·    same context + independent evidence → eligible preference pair',
   caveat:'These are planned evaluation and adapter contracts. No useful GraphCG factors, protein stability, causal cell response, or preference-training benefit is established.',
   detail:['Use signed amplitudes and an unedited control. Do not delete a direction just because one context produced a poor outcome.','Unknown, neutral, conflicting and infeasible are distinct states. A missing assay is not a negative training label.','Quality-first candidate: BF16 DoRA. Quantized candidate: QDoRA. Reference comparison: QLoRA. No universal winner is asserted.','The current environment lacks PEFT/bitsandbytes. No installation or new fine-tuning run is authorized by this figure.'],
   source:'Plan §8; GraphCG; DoRA/QLoRA; PEFT0.21.0; protein-quality and refolding limitations.'},
  campaign:{name:'Campaign & evidence',eyebrow:'07 · REAL RUNS, FIXED BOUNDARIES',title:'Twenty-four proxies. Two targets. No hidden extra runs.',
   takeaway:'The original campaign cap remains 26. Distillation, preference adaptation and a trained baseline comparison cannot be hidden inside the completed campaign.',
   equation:'3 learning rates × 2 readout scales × 2 attention scales × 2 seeds = 24',
   caveat:'This is an evidence snapshot, not live telemetry. Real campaign results remain unavailable; no training receipt is fabricated from a test or schematic.',
   detail:['Live inspection: 11 November 2026,12:08 UTC; full_dataset_ready=false and both owner attestations absent.','Actual source-exposure/group qualifications have progressed; they do not establish optimizer training or a full model release.','New head/vocabulary/family settings must be frozen before applicable proxies, and cannot be inserted into an already running campaign.','Post-training stages are specified but cannot launch without a revised explicit run allocation and all native qualifications.'],
   source:'Current campaign26 directive; source/model ledgers; commit98c7521 snapshot.'}
 };
 const QUALITY={
  protein:[['Sequence ↔ structure','Exact residues, requested backbone and independent compatibility'],['Stability & assembly','Relevant assay or calibrated evidence; preserve legitimate disorder'],['Function & context','Task function, oligomerization and environment are distinct from confidence']],
  molecule:[['Identity & chemistry','Valence, stereochemistry, protonation and atom mapping'],['Geometry & assay','Strain, pose context, concentration, units and uncertainty'],['Developability','Supported solubility/ADMET evidence, not a valid-string proxy']],
  cell:[['Cell-state context','Donor, type, mixture, baseline, dose and sampling time'],['Communication','Supported intercellular effects; co-expression is not causation'],['Confounding & coverage','Composition shifts, batch/doublets, unknown targets and rare types']],
  structure:[['Frame & identity','Exact atom/residue maps, transforms and source roles'],['Physical constraints','Conservation, clashes and separately defined pose/template criteria'],['Mechanistic evidence','A plausible shape is not evidence of a transition state']],
  image:[['Channel & scale','Native channel identity, physical or pixel units and registration'],['Objects & masks','Instance validity, compartments, morphology and uncertainty'],['Evidence class','Generated image plus its mask is internal consistency, not experiment']],
  reasoning:[['Prerequisites','No future, sibling or hidden-target facts'],['Relational transfer','Complete directed paths, grounded mappings and uncovered branches'],['Truth & uncertainty','Citations, units, abstention and independent outcome evidence']]
 };
 function architecture(){
   let b=text(25,59,'Existing source components are not a fully qualified trained system.',13,C.muted);
   b+=box(25,91,250,73,'Native multimodal context','TokenGT · ConvexTok · typed masks','existing');
   b+=box(355,91,250,73,'Analogy retrieval','Train-only problems + whole forests');
   b+=box(685,91,250,73,'Topology / action policy','Discrete branches · lengths · stops');
   b+=line(275,127,345,127,C.muted,2,'',true)+line(605,127,675,127,C.muted,2,'',true);
   b+=box(355,222,250,80,'Conditional forest flow','Continuous node / token-slot transport');
   b+=line(480,164,480,212,C.muted,2,'',true)+line(810,164,810,261,C.muted)+line(810,261,615,261,C.muted,2,'',true);
   b+=box(25,222,250,80,'Native GFlowNet teacher','Qualified stochastic mass accounting');
   b+=line(275,260,345,260,C.teal,2,'',true);
   b+=text(284,246,'distill',11,C.teal);
   b+=box(355,361,250,75,'Final typed decoder','Projection + constrained refinement');
   b+=line(480,302,480,351,C.muted,2,'',true);
   b+=box(685,361,250,75,'Quality / uncertainty','Independent evidence · full coverage');
   b+=line(605,399,675,399,C.muted,2,'',true);
   b+=box(25,361,250,75,'Flow-map / DoRA tuning','Additional runs need allocation','blocked');
   b+=`<path d="M275,399 H325 V280 H345" fill="none" stroke="${C.amber}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arrow)"/>`;
   b+=`<path d="M935,399 H947 V193 H150 V212" fill="none" stroke="${C.muted}" stroke-width="1.3" stroke-dasharray="4 4" marker-end="url(#arrow)"/>`+text(25,185,'Independently qualified reward feedback',11,C.muted);
   b+=text(25,478,'Solid border: existing foundation',12,C.teal)+text(330,478,'Dashed: proposed extension',12,C.blue)+text(635,478,'Amber: post-training budget blocked',12,C.amber);
   return shell('A hybrid, complete-forest architecture','Components and mathematical boundaries of the proposed Mora-3 flow-forest extension.',b);
 }
 function forest(s){
   const f=M.forest(),z=M.interpolateForest(f,s.s),look=new Map(f.nodes.map(n=>[n.id,n]));let b='';
   b+=text(25,65,'Fixed reasoning topology',16,C.ink,600)+text(492,65,`Continuous latent fixture · s = ${s.s.toFixed(2)}`,16,C.ink,600);
   b+=line(463,75,463,390,C.line,1);
   const treePos=n=>{const i=Number(n.id.slice(1));const xx=[210,120,300,70,165,260,355][i];return [xx,105+n.depth*60+(n.tree==='B'?171:0)];};
   for(const e of f.edges){const [x,y]=treePos(look.get(e.source)),[a,c]=treePos(look.get(e.target));b+=line(x,y+11,a,c-13,C.line,2,'',true);}
   for(const n of f.nodes){const[x,y]=treePos(n);b+=`<circle cx="${x}" cy="${y}" r="16" fill="${n.fixed?C.ink:n.tree==='A'?C.teal:C.blue}"/>`+text(x,y+4,n.id,11,'#fff',600,'middle');}
   b+=text(25,430,'2 trees · 14 states · 12 edges · 8 complete paths',13,C.muted);
   const xp=v=>701+v*99,yp=v=>238-v*83;
   b+=line(491,381,936,381,C.line,1)+line(501,94,501,383,C.line,1);
   b+=text(911,400,'z₁',14,C.muted)+text(483,90,'z₂',14,C.muted);
   for(const n of f.nodes){b+=`<circle cx="${xp(n.target[0])}" cy="${yp(n.target[1])}" r="5" fill="none" stroke="${C.line}"/>`;}
   const zl=new Map(z.map(n=>[n.id,n]));
   for(const e of f.edges){const a=zl.get(e.source),c=zl.get(e.target);b+=line(xp(a.z[0]),yp(a.z[1]),xp(c.z[0]),yp(c.z[1]),a.tree==='A'?C.teal:C.blue,1.3,'3 3',false);}
   for(const n of z){b+=`<circle cx="${xp(n.z[0])}" cy="${yp(n.z[1])}" r="${n.fixed?8:6}" fill="${n.fixed?C.ink:n.tree==='A'?C.teal:C.blue}"/>`+text(xp(n.z[0])+9,yp(n.z[1])-6,n.id,11,C.muted);}
   b+=text(492,430,'Filled: current state · rings: target · dark: clamped root',12,C.muted);
   b+=chip(25,460,322,'Analytic interpolation; clean target known',C.amber)+chip(492,460,415,'No learned field, optimizer or biological coordinates',C.amber);
   return shell('Solver time moves all free states; reasoning edges remain','Two complete trees retain all eight paths as solver time moves between fixed noise and a target fixture.',b);
 }
 function topology(s){
   let b=text(25,65,'Directed target presentation',16,C.ink,600)+text(535,65,'Separate metric Rips filtration',16,C.ink,600);
   const P=[[210,110],[90,230],[330,230],[210,350]];
   if(s.square)b+=`<polygon points="210,110 90,230 210,350 330,230" fill="${C.teal}" fill-opacity=".1"/>`;
   for(const [i,j,l] of [[0,1,'a'],[1,3,'b'],[0,2,'c'],[2,3,'d']]){const dx=P[j][0]-P[i][0],dy=P[j][1]-P[i][1],len=Math.hypot(dx,dy);b+=line(P[i][0],P[i][1],P[j][0]-14*dx/len,P[j][1]-14*dy/len,C.teal,2,'',true)+text((P[i][0]+P[j][0])/2+(i<2?-15:8),(P[i][1]+P[j][1])/2,l,17,C.teal,600);}
   for(let i=0;i<4;i++)b+=`<circle cx="${P[i][0]}" cy="${P[i][1]}" r="10" fill="${C.ink}"/>`+text(P[i][0]+15,P[i][1]-10,i===0?'start':i===3?'finish':`state ${i}`,12,C.muted);
   b+=text(210,224,s.square?'ab ∼ cd':'ab ≁ cd',22,s.square?C.teal:C.red,600,'middle')+text(210,252,s.square?'supplied 2-cell':'no supplied 2-cell',13,C.muted,400,'middle');
   const d=M.directedDiamond(s.square);
   b+=chip(32,399,391,`${d.pathClasses} path class${d.pathClasses===1?'':'es'} · mapping ${s.square?'preserved':'refused'}`,s.square?C.teal:C.red);
   b+=text(32,450,'Reverse finish → start is not reachable.',13,C.muted);
   const ph=M.squarePersistence(s.epsilon),Q=[[615,110],[825,110],[825,320],[615,320]];
   if(ph.triangles)b+=`<rect x="615" y="110" width="210" height="210" fill="${C.blue}" fill-opacity=".08"/>`;
   for(const [i,j]of ph.edges)b+=line(...Q[i],...Q[j],C.blue,2);
   for(let i=0;i<4;i++)b+=`<circle cx="${Q[i][0]}" cy="${Q[i][1]}" r="8" fill="${C.blue}"/>`;
   b+=text(720,217,`H₀=${ph.h0}    H₁=${ph.h1}`,20,C.ink,650,'middle');
   b+=text(535,350,`ε = ${s.epsilon.toFixed(2)} · complete H₁ interval [1, √2)`,13,C.muted);
   const barx=v=>550+v*222;
   b+=line(barx(0),385,barx(1.6),385,C.line,2)+line(barx(1),385,barx(Math.SQRT2),385,C.blue,7);
   b+=`<circle cx="${barx(1)}" cy="385" r="5" fill="${C.blue}"/><circle cx="${barx(Math.SQRT2)}" cy="385" r="5" fill="white" stroke="${C.blue}" stroke-width="2"/>`;
   b+=line(barx(s.epsilon),365,barx(s.epsilon),408,C.amber,2)+text(barx(1),426,'1',12,C.muted,400,'middle')+text(barx(Math.SQRT2),426,'√2',12,C.muted,400,'middle');
   b+=text(535,463,'Barcode similarity cannot certify a directed analogy.',12,C.amber);
   return shell('Directed composition and metric persistence answer different questions','Two directed paths remain different unless a defining square is supplied. A separate unit-square metric example has one H1 interval from one to square root two.',b);
 }
 function decode(s){
   let b=text(25,65,'Schematic token embedding codebook',16,C.ink,600);
   const px=x=>65+x*220,py=y=>398-y*150;
   b+=line(55,405,510,405,C.line,1)+line(55,85,55,405,C.line,1);
   const allowed=['hypothesis','unknown'],raw=M.nearestToken([s.x,s.y],null),typed=M.nearestToken([s.x,s.y],allowed);
   b+=line(px(s.x),py(s.y),px(raw.z[0]),py(raw.z[1]),C.amber,2,'4 3');
   b+=line(px(s.x),py(s.y),px(typed.z[0]),py(typed.z[1]),C.teal,2);
   for(const t of M.CODEBOOK){const ok=allowed.includes(t.type);b+=`<circle cx="${px(t.z[0])}" cy="${py(t.z[1])}" r="${ok?8:6}" fill="${ok?C.teal:'#a7b2b7'}"/>`+text(px(t.z[0])+9,py(t.z[1])-10,t.name,11,ok?C.teal:C.muted,ok?650:400);}
   b+=`<path d="M ${px(s.x)} ${py(s.y)-10} l 10 10 l -10 10 l -10 -10 Z" fill="${C.ink}"/>`+text(px(s.x)+14,py(s.y)+15,'proposed z',11,C.ink,650);
   b+=text(25,451,'Green candidates satisfy the declared hypothesis-role grammar.',12,C.muted);
   b+=box(560,99,365,84,'Unrestricted nearest',raw.name+' · distance '+raw.distance.toFixed(3),'planned');
   b+=line(740,183,740,225,C.muted,2,'',true);
   b+=box(560,235,365,84,'Typed refinement',typed.name+' · distance '+typed.distance.toFixed(3),'existing');
   b+=text(565,360,'40 proposed controls. No token IDs allocated.',14,C.ink,650);
   b+=text(565,389,'Byte coverage remains; checkpoints are versioned.',13,C.muted);
   b+=text(565,427,'Token labels are not provenance evidence.',13,C.amber,650);
   b+=text(565,452,'Discrete output mass changes after projection.',13,C.amber);
   return shell('Final decoding must preserve grammar, roles and evidence boundaries','A proposed latent point can have a nearest token that is disallowed by the current grammar. Constrained refinement chooses among valid token types.',b);
 }
 function fewstep(s){
   let b=text(25,65,'Exact analytic fixture: dx/ds = x, x(0)=1',16,C.ink,600);
   const px=t=>70+550*t,py=x=>391-(x-1)*158; b+=line(80,89,110,89,C.ink,2.5)+text(118,94,'Exact exp(s)',12,C.ink)+line(280,89,310,89,C.teal,2.5)+text(318,94,'Euler approximation',12,C.teal);
   b+=line(70,391,640,391,C.line,1)+line(70,90,70,400,C.line,1);
   for(const v of [1,1.5,2,2.5])b+=line(70,py(v),630,py(v),'#e7ecee',1)+text(54,py(v)+4,v.toFixed(1),12,C.muted,400,'end');
   const exact=Array.from({length:101},(_,i)=>[i/100,Math.exp(i/100)]),r=M.eulerExp(s.steps);
   b+=`<polyline points="${exact.map(p=>px(p[0])+','+py(p[1])).join(' ')}" fill="none" stroke="${C.ink}" stroke-width="2.5"/>`;
   b+=`<polyline points="${r.points.map(p=>px(p[0])+','+py(p[1])).join(' ')}" fill="none" stroke="${C.teal}" stroke-width="2.5"/>`;
   for(const [t,x]of r.points)b+=`<circle cx="${px(t)}" cy="${py(x)}" r="4" fill="${C.teal}"/>`;
   b+=text(70,420,'0',12,C.muted)+text(610,420,'1',12,C.muted)+text(310,447,'solver time s',14,C.muted);
   b+=text(686,114,`${s.steps}`,46,C.teal,650)+text(685,143,'Euler field evaluations',13,C.muted);
   b+=text(685,198,r.final.toFixed(5),25,C.ink,650)+text(685,225,'Euler x(1)',13,C.muted);
   b+=text(685,273,r.error.toFixed(5),25,C.amber,650)+text(685,301,'absolute endpoint error',13,C.muted);
   b+=text(685,350,'Exact flow map',15,C.ink,650)+text(685,377,'M(x,s,t)=e^(t−s) x',15,C.ink);
   b+=text(685,410,'Analytic oracle; not a trained head.',12,C.amber);
   b+=chip(25,466,430,'1–4-step neural quality requires a trained student',C.amber);
   return shell('A flow map is not a coarse Euler step','The exact exponential solution is compared with Euler integration at the selected number of steps. Endpoint errors are mathematical fixture errors, not learned model measurements.',b);
 }
 function quality(s){
   let b=text(25,65,'Steering is conditional; labels require evidence',16,C.ink,600);
   b+=line(40,245,390,245,C.line,1)+line(210,100,210,375,C.line,1);
   b+=`<circle cx="210" cy="245" r="8" fill="${C.ink}"/>`+text(222,269,'unedited z',12,C.muted);
   b+=line(210,245,345,140,C.teal,3,'',true)+line(210,245,75,350,C.purple,3,'',true);
   b+=text(310,118,'+ αdₖ',18,C.teal,650)+text(60,378,'− αdₖ',18,C.purple,650);
   b+=text(45,406,'Same direction + different context',15,C.ink,650)+text(45,432,'can yield a different quality disposition.',13,C.muted);
   const items=QUALITY[s.modality];b+=text(460,65,`Applicable evidence: ${s.modality}`,16,C.ink,600);
   items.forEach(([title,desc],i)=>{
     const y=92+i*77;b+=`<rect x="460" y="${y}" width="470" height="66" rx="7" fill="${i%2?C.pale:'#f6f7f8'}"/>`+text(474,y+23,title,15,C.ink,650);
     // Short domains are wrapped in two accessible, legible lines if necessary.
     const words=desc.split(' ');let a='',parts=[];for(const word of words){if((a+' '+word).trim().length>58){parts.push(a);a=word;}else a=(a+' '+word).trim();}parts.push(a);
     parts.slice(0,2).forEach((t,j)=>b+=text(474,y+43+j*15,t,11.6,C.muted));
   });
   b+=chip(460,338,142,'DoRA: candidate',C.teal)+chip(612,338,153,'QDoRA: candidate',C.blue)+chip(775,338,153,'QLoRA: baseline',C.muted);
   b+=text(460,390,'64 × 64 fixture, rank 8:',13,C.muted)+text(460,418,`LoRA ${M.adapterParameterCount(64,64,8,'lora')} · DoRA ${M.adapterParameterCount(64,64,8,'dora')} trainable parameters`,15,C.ink,650);
   b+=text(460,444,'Row-norm convention. Not the native model parameter count.',11.8,C.muted);
   b+=chip(25,467,905,'Unknown ≠ failed. Predicted confidence ≠ experimental stability. Co-expression ≠ causation.',C.amber);
   return shell('Contextual quality evidence precedes preference training','Matched-context GraphCG exploration is assessed through modality-appropriate evidence. DoRA and QDoRA are candidates, QLoRA is the reference baseline, and no outcome is guaranteed.',b);
 }
 function campaign(){
   let b=text(25,65,'Predeclared native proxy grid',16,C.ink,600)+text(651,65,'Downstream target / later stages',16,C.ink,600);
   const lrs=['1e−4','3e−4','1e−3'];
   for(let lr=0;lr<3;lr++){
     b+=text(25,112+lr*95,`LR ${lrs[lr]}`,14,C.ink,650);
     for(let j=0;j<8;j++){
       const x=130+j*59,y=87+lr*95;const id=lr*8+j+1;
       b+=`<rect x="${x}" y="${y}" width="48" height="50" rx="6" fill="#f3f5f6" stroke="${C.line}"/>`+text(x+24,y+24,String(id).padStart(2,'0'),14,C.ink,650,'middle')+text(x+24,y+41,'—',11,C.muted,400,'middle');
     }
   }
   b+=text(130,379,'Each row: 2 readout × 2 attention × 2 seeds',12.5,C.muted);
   b+=text(25,426,'— = no verified actual outcome in this snapshot',13,C.muted);
   b+=box(650,88,280,71,'Target run 25','Selected HPs · native μP retained','blocked');
   b+=box(650,183,280,71,'Target run 26','Independent predeclared seed','blocked');
   b+=line(790,159,790,174,C.muted,1.6,'',true);
   b+=box(650,298,280,99,'Later optimization stages','Distillation · preference · baseline','blocked');
   b+=text(665,374,'No allocated extra training slots',12,C.amber,650);
   b+=chip(25,463,905,'Snapshot: full_dataset_ready=false · owner attestations absent · no readiness override',C.amber);
   return shell('The 26-run cap is not silently expanded','Twenty-four proxy slots and two target slots remain the full authorized training campaign. Additional fine-tuning stages are planned but budget-blocked.',b);
 }
 function renderSvg(tab,state){const s={...M.parseState(''),...state};const f={architecture,forest,topology,decode,fewstep,quality,campaign}[tab];if(!f)throw new Error('unknown view');return canonicalSvgCoordinates(f(s));}
 function rewardSvg(temperature){
   const e=[.2,.8,1.4,2.1],q=[.4,.3,.2,.1],p=M.rewardProbabilities(e,temperature,q);let b=text(25,60,'Finite illustration: Rᵢ = q₀ᵢ exp(−Eᵢ / temperature)',14,C.muted);
   for(let i=0;i<4;i++){const y=94+i*58;b+=text(25,y+18,`Forest ${i+1}`,14,C.ink,600)+`<rect x="130" y="${y}" width="${p[i]*650}" height="30" rx="3" fill="${C.teal}"/>`+text(140+p[i]*650,y+21,(100*p[i]).toFixed(1)+'%',13,C.ink,600)+text(800,y+21,`E=${e[i]} · q₀=${q[i]}`,12,C.muted);}
   b+=text(25,354,'Normalized exactly on four fixture objects. This is not a trained GFlowNet distribution.',13,C.amber);
   return canonicalSvgCoordinates(shell('Reward-proportional mass is a distribution, not a top-score choice','A four-object finite example uses an explicit base mass and nonnegative energies. Probabilities sum to one.',b,375));
 }
 return Object.freeze({META,QUALITY,C,renderSvg,rewardSvg});
});
