'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../../docs/project/flow-forest-math.js');
const close = (a,b,e=1e-12) => assert.ok(Math.abs(a-b)<=e, `${a} != ${b}`);

test('two complete trees retain all eight three-state paths', () => {
  const f=M.forest(); assert.equal(f.nodes.length,14); assert.equal(f.edges.length,12);
  const paths=M.paths(f); assert.equal(paths.length,8);
  assert.ok(paths.every(p=>p.length===3)); assert.equal(new Set(paths.flat()).size,14);
});
test('shared ancestors are represented once rather than duplicated by leaf', () => {
  const ps=M.paths(M.forest()); assert.equal(ps.filter(p=>p[0]==='A0').length,4);
  assert.equal(M.forest().nodes.filter(n=>n.id==='A0').length,1);
});
test('flow endpoint and clamped conditioning roots are exact', () => {
  for (const s of [0,.25,.5,1]) {
    const f=M.forest(); const z=M.interpolateForest(f,s);
    for (let i=0;i<f.nodes.length;i++) {
      const n=f.nodes[i];const want=n.fixed?n.target:n.target.map((v,k)=>(1-s)*n.noise[k]+s*v);
      z[i].z.forEach((v,k)=>close(v,want[k]));
    }
  }
});
test('solver steps do not change topology or reasoning depth', () => {
  for (const steps of [1,2,4,16]) {
    assert.equal(M.paths(M.forest()).length,8); assert.equal(M.eulerExp(steps).points.length,steps+1);
  }
});
test('reward-proportional fixture includes its stated base measure', () => {
  const e=[.2,.8,1.4,2.1], q=[.4,.3,.2,.1];
  const p=M.rewardProbabilities(e,1,q);close(p.reduce((a,b)=>a+b,0),1);
  close(p[0]/p[1],q[0]/q[1]*Math.exp(e[1]-e[0]));
});
test('large energies normalize stably and no arbitrary score is called probability', () => {
  const p=M.rewardProbabilities([1000,1001],1,[.5,.5]); assert.ok(p.every(Number.isFinite));
  close(p[0],1/(1+Math.exp(-1)));
});
test('typed decoding may differ from unrestricted nearest codebook point', () => {
  const x=[.32,.34]; assert.equal(M.nearestToken(x,null).id,0);
  const grammar=M.nearestToken(x,['hypothesis','unknown']); assert.notEqual(grammar.id,0);
  assert.ok(['hypothesis','unknown'].includes(grammar.type));
});
test('empty grammar abstains; nearest ties are stable', () => {
  assert.equal(M.nearestToken([0,0],[]),null);
  const a=M.nearestToken([1.25,1.1],['finish','verify']);
  assert.ok(a && Number.isFinite(a.distance));
});
test('the proposal has exactly forty distinct controls, no allocated token IDs', () => {
  const t=Object.values(M.CONTROLS).flat();assert.equal(t.length,40);assert.equal(new Set(t).size,40);
  assert.ok(t.every(x=>/^<MORA_[A-Z_]+>$/.test(x)));
});
test('unit-square Rips H1 is born at one and killed at square-root two', () => {
  assert.equal(M.squarePersistence(.99).h1,0);assert.equal(M.squarePersistence(1).h1,1);
  assert.equal(M.squarePersistence(1.3).h1,1);assert.equal(M.squarePersistence(Math.SQRT2).h1,0);
  assert.equal(M.squarePersistence(.99).h0,4);assert.equal(M.squarePersistence(1).h0,1);
});
test('directed path equivalence depends on a declared two-cell, not endpoints', () => {
  assert.equal(M.directedDiamond(false).pathClasses,2);assert.equal(M.directedDiamond(true).pathClasses,1);
  assert.equal(M.directedDiamond(false).mappingStatus,'refused');
  assert.equal(M.directedDiamond(true).mappingStatus,'proved-in-presentation');
  assert.equal(M.directedDiamond(true).reverseReachable,false);
});
test('Euler errors are real analytic fixture errors, not trained model results', () => {
  let previous=Infinity;
  for (const n of [1,2,4,16]) { const r=M.eulerExp(n); close(r.final,(1+1/n)**n);
    assert.ok(r.error<previous);previous=r.error;assert.equal(r.nfe,n); }
  close(M.exactFlowMap(1,0,1),Math.E);
});
test('flow-map composition and identity are exact for the declared ODE', () => {
  close(M.exactFlowMap(2,.3,.3),2);
  close(M.exactFlowMap(M.exactFlowMap(1,0,.4),.4,1),M.exactFlowMap(1,0,1));
});
test('unknown evidence is neither a negative label nor a passed criterion', () => {
  assert.equal(M.adjudicate(['pass','unknown']).status,'unevaluated');
  assert.equal(M.adjudicate(['pass','fail']).status,'review-required');
  assert.equal(M.adjudicate(['pass','pass']).status,'passes-specified-checks');
  assert.equal(M.adjudicate(['pass']).biologicalValidation,false);
});
test('DoRA parameter illustration uses a stated row-norm convention', () => {
  assert.equal(M.adapterParameterCount(64,64,8,'lora'),1024);
  assert.equal(M.adapterParameterCount(64,64,8,'dora'),1088);
});
test('URL state rejects unsupported modes and clamps bounded numeric controls', () => {
  const x=M.parseState('#tab=oops&s=999&steps=3&epsilon=-1');
  assert.equal(x.tab,'architecture');assert.equal(x.s,1);assert.equal(x.steps,4);assert.equal(x.epsilon,0);
});
test('canonical URL roundtrip preserves public controls', () => {
  const s={...M.parseState(''),tab:'topology',s:.37,epsilon:1.2,square:false,steps:2};
  const t=M.parseState(M.stateHash(s));assert.equal(t.tab,'topology');close(t.s,.37);close(t.epsilon,1.2);
  assert.equal(t.square,false);assert.equal(t.steps,2);
});
test('invalid mathematical inputs fail rather than silently generating evidence', () => {
  for (const f of [()=>M.rewardProbabilities([1],0),()=>M.rewardProbabilities([NaN],1),
      ()=>M.rewardProbabilities([1,2],1,[1,0]),()=>M.eulerExp(0),()=>M.eulerExp(1.5),
      ()=>M.squarePersistence(-1),()=>M.nearestToken([NaN,0],null),
      ()=>M.interpolateForest(M.forest(),1.1),()=>M.adjudicate(['maybe'])]) assert.throws(f);
});
