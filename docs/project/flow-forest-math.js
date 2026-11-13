/* Mora-3 FRF-00: deterministic mathematical illustrations, not a trained model.
 * Original local reference code. No network, optimizer, model weights or training.
 * All plotted quantities are computed by the same exported functions tested in Node.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MoraMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const finite = x => typeof x === 'number' && Number.isFinite(x);
  const requireValue = (ok,msg) => { if (!ok) throw new Error(msg); };
  const CONTROLS = Object.freeze({
    Structure: Object.freeze(['FOREST_BEGIN','FOREST_END','TREE_BEGIN','TREE_END','NODE_BEGIN','NODE_END','EDGE_BEGIN','EDGE_END'].map(x=>`<MORA_${x}>`)),
    Actions: Object.freeze(['PARENT','BRANCH','JOIN','SUBGOAL','PROPOSE','VERIFY','REVISE','FINISH'].map(x=>`<MORA_${x}>`)),
    Evidence: Object.freeze(['CONDITION','DESIRED','PREDICTED','SOURCE_OBSERVED','SOURCE_COMPUTED','SIMULATED','HYPOTHESIS','UNKNOWN'].map(x=>`<MORA_${x}>`)),
    Relations: Object.freeze(['ANALOGY','MAPPING','RELATION','CONSTRAINT','UNIT','FRAME','MISSING','ABSTAIN'].map(x=>`<MORA_${x}>`)),
    Decisions: Object.freeze(['SUPPORTED','UNSUPPORTED','PASS','FAIL','UNCERTAIN','CONFLICT','INFEASIBLE','BUDGET_EXHAUSTED'].map(x=>`<MORA_${x}>`))
  });
  const CODEBOOK=Object.freeze([
    {id:0,name:'SOURCE_OBSERVED',type:'source_observed',z:[.20,.25]},
    {id:1,name:'HYPOTHESIS',type:'hypothesis',z:[.95,.45]},
    {id:2,name:'UNKNOWN',type:'unknown',z:[.10,1.35]},
    {id:3,name:'VERIFY',type:'verify',z:[1.05,1.50]},
    {id:4,name:'FINISH',type:'finish',z:[1.65,.70]},
    {id:5,name:'FAIL',type:'fail',z:[1.70,1.70]}
  ].map(x=>Object.freeze({...x,z:Object.freeze(x.z)})));
  const TABS=Object.freeze(['architecture','forest','topology','decode','fewstep','quality','campaign']);
  function forest() {
    const nodes=[],edges=[];
    for(let tree=0;tree<2;tree++){
      const key=tree?'B':'A';
      const coords=tree?[[.1,-.65],[.55,-.25],[.8,-1.05],[1.4,.05],[1.45,-.5],[1.55,-.8],[1.8,-1.3]]:
        [[-1.8,.55],[-1.2,1.05],[-1.15,.1],[-.45,1.45],[-.35,.85],[-.35,.35],[-.2,-.15]];
      for(let i=0;i<7;i++){
        const id=key+i,depth=i===0?0:i<3?1:2,seed=i+tree*7;
        nodes.push({id,tree:key,depth,fixed:i===0,parent:i?key+Math.floor((i-1)/2):null,
          target:coords[i].slice(),noise:i?[1.85*Math.sin(1.73*(seed+1)),1.35*Math.cos(2.11*(seed+1))]:coords[i].slice()});
        if(i)edges.push({source:key+Math.floor((i-1)/2),target:id,kind:'computational-dependency'});
      }
    }
    return {nodes,edges,kind:'fixed-topology-mathematical-fixture',learned:false};
  }
  function paths(f) {
    requireValue(f && Array.isArray(f.nodes) && Array.isArray(f.edges),'finite forest required');
    const ids=new Set(f.nodes.map(n=>n.id));requireValue(ids.size===f.nodes.length,'duplicate node');
    const children=new Map(f.nodes.map(n=>[n.id,[]])),parents=new Map();
    for(const e of f.edges){requireValue(ids.has(e.source)&&ids.has(e.target)&&!parents.has(e.target),'known one-parent forest required');
      parents.set(e.target,e.source);children.get(e.source).push(e.target);}
    const out=[],seen=new Set();
    const visit=(id,path)=>{requireValue(!path.includes(id),'cycle');seen.add(id);const p=[...path,id];
      if(!children.get(id).length)out.push(p);else children.get(id).forEach(c=>visit(c,p));};
    f.nodes.filter(n=>!parents.has(n.id)).forEach(n=>visit(n.id,[]));
    requireValue(seen.size===ids.size,'unreachable cycle');return out;
  }
  function interpolateForest(f,s){
    requireValue(finite(s)&&s>=0&&s<=1,'flow time in [0,1] required');
    return f.nodes.map(n=>({...n,z:n.fixed?n.target.slice():n.target.map((v,k)=>(1-s)*n.noise[k]+s*v)}));
  }
  function rewardProbabilities(energy,temperature=1,base=null){
    requireValue(Array.isArray(energy)&&energy.length>0&&energy.every(x=>finite(x)&&x>=0),'finite nonnegative energies required');
    requireValue(finite(temperature)&&temperature>0,'positive temperature required');
    const q=base===null?energy.map(()=>1/energy.length):base;
    requireValue(Array.isArray(q)&&q.length===energy.length&&q.every(x=>finite(x)&&x>0),'positive complete base measure required');
    requireValue(Math.abs(q.reduce((a,b)=>a+b,0)-1)<1e-10,'base mass must sum to one');
    const log=energy.map((e,i)=>Math.log(q[i])-e/temperature),max=Math.max(...log);
    const weights=log.map(x=>Math.exp(x-max)),total=weights.reduce((a,b)=>a+b,0);return weights.map(x=>x/total);
  }
  function nearestToken(point,types=null){
    requireValue(Array.isArray(point)&&point.length===2&&point.every(finite),'finite two-dimensional point required');
    requireValue(types===null||(Array.isArray(types)&&types.every(x=>typeof x==='string')),'explicit type allowlist required');
    const candidates=CODEBOOK.filter(t=>types===null||types.includes(t.type));if(!candidates.length)return null;
    return candidates.map(t=>({...t,distance:Math.hypot(point[0]-t.z[0],point[1]-t.z[1])}))
      .sort((a,b)=>a.distance-b.distance||a.id-b.id)[0];
  }
  function squarePersistence(radius){
    requireValue(finite(radius)&&radius>=0,'nonnegative finite filtration radius required');
    const points=[[0,0],[1,0],[1,1],[0,1]],edges=[];
    for(let i=0;i<4;i++)for(let j=i+1;j<4;j++)if(Math.hypot(points[i][0]-points[j][0],points[i][1]-points[j][1])<=radius)edges.push([i,j]);
    return {points,edges,h0:radius<1?4:1,h1:radius>=1&&radius<Math.SQRT2?1:0,
      h1Interval:[1,Math.SQRT2],triangles:radius>=Math.SQRT2?4:0,
      scope:'exact-unit-square-Rips-fixture-F2-H0-H1-not-general-persistence-solver'};
  }
  function directedDiamond(square){
    requireValue(typeof square==='boolean','explicit supplied square flag required');
    return {paths:[['a','b'],['c','d']],pathClasses:square?1:2,
      mappingStatus:square?'proved-in-presentation':'refused',reverseReachable:false,biologicalCausality:false};
  }
  function eulerExp(steps){
    requireValue(Number.isInteger(steps)&&steps>=1&&steps<=128,'bounded positive integer steps required');
    let x=1;const points=[[0,x]];for(let i=0;i<steps;i++){x+=x/steps;points.push([(i+1)/steps,x]);}
    return {points,final:x,exact:Math.E,error:Math.abs(Math.E-x),nfe:steps,learned:false};
  }
  function exactFlowMap(x,s,t){
    requireValue([x,s,t].every(finite)&&s>=0&&t>=s&&t<=1,'finite forward interval required');
    return x*Math.exp(t-s);
  }
  function adjudicate(values){
    requireValue(Array.isArray(values)&&values.length>0&&values.every(x=>['pass','fail','unknown'].includes(x)),'explicit criterion dispositions required');
    return {status:values.includes('fail')?'review-required':values.includes('unknown')?'unevaluated':'passes-specified-checks',
      observed:values.filter(x=>x!=='unknown').length,total:values.length,biologicalValidation:false};
  }
  function adapterParameterCount(output,input,rank,method){
    requireValue([output,input,rank].every(x=>Number.isInteger(x)&&x>0)&&rank<=Math.min(output,input),'positive bounded matrix/rank required');
    requireValue(['lora','dora'].includes(method),'method required');return rank*(output+input)+(method==='dora'?output:0);
  }
  function parseState(hash){
    const q=new URLSearchParams(String(hash).replace(/^#/,''));
    const bounded=(key,fallback,min,max)=>{const v=q.get(key);const n=v===null?fallback:Number(v);return finite(n)?Math.min(max,Math.max(min,n)):fallback;};
    return {tab:TABS.includes(q.get('tab'))?q.get('tab'):'architecture',s:bounded('s',.6,0,1),
      steps:[1,2,4,8,16,32].includes(Number(q.get('steps')))?Number(q.get('steps')):4,
      epsilon:bounded('epsilon',1.2,0,1.6),square:q.get('square')!=='0',
      x:bounded('x',.32,-.1,1.9),y:bounded('y',.34,-.1,1.9),temperature:bounded('temperature',1,.2,3),
      modality:['protein','molecule','cell','structure','image','reasoning'].includes(q.get('modality'))?q.get('modality'):'protein'};
  }
  function stateHash(state){
    const q=new URLSearchParams({tab:state.tab,s:state.s,steps:state.steps,epsilon:state.epsilon,
      square:state.square?'1':'0',x:state.x,y:state.y,temperature:state.temperature,modality:state.modality});return '#'+q;
  }
  return Object.freeze({CONTROLS,CODEBOOK,TABS,forest,paths,interpolateForest,rewardProbabilities,nearestToken,
    squarePersistence,directedDiamond,eulerExp,exactFlowMap,adjudicate,adapterParameterCount,parseState,stateHash});
});
