import {AtlasScene, toyModel} from './viewer-scene.js';

const root = document.querySelector('[data-atlas]');
const stages = JSON.parse(document.querySelector('#atlas-data').textContent);
const canvas = root.querySelector('canvas');
const stageButtons = [...root.querySelectorAll('[data-stage]')];
const labelsRoot = root.querySelector('.atlas-labels');
const slider = root.querySelector('#atlas-slider');
const play = root.querySelector('[data-play]');
const noMotion = matchMedia('(prefers-reduced-motion: reduce)');
const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
const lerp = (a,b,t) => a+(b-a)*t;
const pad = n => String(n).padStart(2,'0');
const fixed = (v,n=2) => (Math.abs(v)<1e-12?0:v).toFixed(n);
const percent = v => `${(v*100).toFixed(1)}%`;
const vector = items => `(${items.map(v=>fixed(v)).join(', ')})`;
const text = (name, value) => { root.querySelector(`#atlas-${name}`).textContent = value; };
const groupColor = ['#8de3ba','#edc579','#c8b5ef','#9bdaef'];
let scene, selected=0, camera, animation, frame=0, timer=0, playing=false, focused=false;
let values = stages.map(stage=>stage.control.value);

const labels = stages.map((stage,i)=>{
  const label=document.createElement('div');
  const number=document.createElement('strong');
  const name=document.createElement('span');
  label.className='atlas-label';
  number.textContent=pad(i+1);
  number.style.setProperty('--label-color',groupColor[stage.group]);
  name.className='atlas-label-text';
  name.textContent=stage.short;
  label.append(number,name);
  labelsRoot.append(label);
  return label;
});

function example(index,value,model) {
  switch (index) {
    case 0: return {rows:[['Observed',`${value} / 6`],['Unavailable',6-value]],math:'Only observed and permitted inputs enter the toy condition.'};
    case 1: return {rows:[['Node tokens',4],['Directed edge tokens',value],['Learned graph token',1],['Rows before padding',5+value]],math:'Feature + ordered endpoint IDs + role form each entity row. The graph token is prepended separately.'};
    case 2: return {rows:[['Input bytes',model.input.length],['Shortest path tokens',model.pieces.length]],math:`Fixed vocabulary: A, C, AC, ACAC. Encoding: ${model.pieces.join(' · ')}`};
    case 3: return {rows:[['Winning keys by channel',model.winners.map(keys=>keys.map(k=>`K${k+1}`).join('/')).join(' · ')],['Output vector',vector(model.result)]],math:'score(q,k) = min(q−k) − max(q−k). Each output takes maxₖ(score + Vₖ).'};
    case 4: return {rows:[['Edit Δz',vector(model.direction.map(v=>v*value))],['Edit length',fixed(Math.hypot(...model.direction)*Math.abs(value))]],math:'z′ = z + αd, with unit d = '+vector(model.direction)+'.'};
    case 5: return {rows:[['Exit at steps 1 / 2 / 3',model.masses.map(percent).join(' / ')],['Total exit mass',percent(model.masses.reduce((a,b)=>a+b,0))]],math:'Stop masses are p, (1−p)p, and (1−p)². The last stop is forced. Continuous move densities are separate.'};
    case 6: return {rows:model.selected.map((ids,i)=>['Tree '+(i+1)+' selected scores',ids.map(id=>fixed(model.scores[i][id])).join(' · ')]),math:'Each toy tree has three eligible branches. Keep the highest-scoring K per tree.'};
    case 7: return {rows:[['4D query',vector(model.query)],
      ['Nearest in the 2D view',model.projected],['Nearest in the full space',model.nearest],
      ['Selected eligible candidate',model.selected],['No-transfer baseline',model.baseline]],
      math:'Candidates rank by 4D Euclidean distance. The first two coordinates give the projected distance. Role, directed edges, cell state, intervention, and endpoint gates apply before selection.'};
    case 8: {
      const mean=model.points.reduce((sum,point,i)=>sum+point.reduce((s,v,j)=>s+(v-model.reference[i][j])**2,0),0)/model.points.length;
      return {rows:[['Mapped atoms',model.points.length],['Unaligned RMS from A',`${fixed(Math.sqrt(mean),3)} Å`]],math:'Linear interpolation in a fixed frame. RMS displacement is from endpoint A without alignment.'};
    }
    case 9: return {rows:[['Stage temperature',model.temperature+' K'],
      ['Cooling from prior stage',model.gap===null?'Initial stage':model.gap+' K'],
      ['Lower-energy state',percent(1-model.population)],['Higher-energy state',percent(model.population)]],
      math:'Illustrative schedule 800 → 550 → 400 → 325 → 290 K. Equal base weights, ΔE = 0.06 eV. p(high) = 1 / (1 + exp(ΔE / kᴮT)).'};
    case 10: return {rows:[['Toy energy',`${fixed(model.energy,3)} eV`],['Toy force',`${fixed(model.force,3)} eV/Å`]],math:'E = ½kx², F = −kx, with k = 0.4 eV/Å². The browser does not run either teacher.'};
    case 11: return {rows:[['Illustrative grid',value*8],['Recorded primary starts','0 / 24']],math:'Toy grid = LR choices × 2 readout × 2 key × 2 seeds. Moving the slider does not change the project plan.'};
    default: throw new Error('Unrecognized atlas stage');
  }
}

function renderAttentionTerms(model) {
  const container=root.querySelector('#atlas-terms');
  if (!container) return;
  container.replaceChildren();
  container.hidden=selected!==3;
  if (container.hidden) return;
  const table=document.createElement('table');
  table.createCaption().textContent='Score + value by key and channel';
  const header=table.createTHead().insertRow();
  for (const label of ['Key','Score','C1','C2','C3']) {
    const cell=document.createElement('th');
    cell.scope='col'; cell.textContent=label; header.append(cell);
  }
  const body=table.createTBody();
  model.scores.forEach((score,key)=>{
    const row=body.insertRow();
    const label=document.createElement('th');
    label.scope='row'; label.textContent=`K${key+1}`; row.append(label);
    row.insertCell().textContent=fixed(score);
    model.terms.forEach((channel,i)=>{
      const cell=row.insertCell();
      const winning=model.winners[i].includes(key);
      cell.textContent=fixed(channel[key])+(winning?' ★':'');
      if (winning) cell.className='is-winner';
    });
  });
  container.append(table);
}

function setNumbers() {
  const value=values[selected], stage=stages[selected];
  const model=toyModel(selected,value);
  const sample=example(selected,value,model);
  text('slider-value',`${Number(value.toFixed(3))}${stage.control.unit}`);
  text('math',sample.math);
  renderAttentionTerms(model);
  const candidates=root.querySelector('#atlas-candidates');
  candidates.replaceChildren(); candidates.hidden=selected!==7;
  if(selected===7) {
    const heading=document.createElement('h3');heading.textContent='Candidate audit · toy values';candidates.append(heading);
    model.candidates.forEach(item=>{
      const row=document.createElement('div');row.className='atlas-candidate';row.dataset.refused=String(!!item.refusal);
      const label=document.createElement('strong');label.textContent=`${item.id} · 4D ${fixed(item.full,3)} · 2D ${fixed(item.projected,3)}${item.id===model.selected?' · selected':''}`;
      const reason=document.createElement('span');reason.textContent=item.refusal?`REFUSED · ${item.refusal}`:'ELIGIBLE · outcome masked';
      row.append(label,reason);candidates.append(row);
    });
  }
  const container=root.querySelector('#atlas-numerics');
  container.replaceChildren(...sample.rows.map(([name,value])=>{
    const wrapper=document.createElement('div');
    const label=document.createElement('span');
    const number=document.createElement('strong');
    wrapper.className='atlas-number';
    label.textContent=name;
    number.textContent=value;
    wrapper.append(label,number);
    return wrapper;
  }));
}

function overviewCamera() {
  const aspect=canvas.clientWidth/Math.max(1,canvas.clientHeight);
  return {yaw:-.25,pitch:.83,distance:Math.max(24.5,29/Math.max(.80,aspect)),target:[0,.18,0]};
}
function focusCamera() {
  const [x,,z]=scene.centers[selected];
  return {yaw:-.34,pitch:.59,distance:7.2,target:[x,.66,z]};
}
function updateLabels() {
  if(!scene) return;
  const width=canvas.clientWidth, height=canvas.clientHeight;
  scene.labelPoints().forEach((point,i)=>{
    const label=labels[i];
    const active=i===selected;
    label.hidden=!point || (focused&&!active) || point[0]<10 || point[0]>width-10 || point[1]<29 || point[1]>height-12;
    label.classList.toggle('is-active',active);
    if(point) label.style.transform=`translate(${point[0]}px,${point[1]}px) translate(-50%,-100%)`;
  });
}
function redraw() {
  if(frame || !scene) return;
  frame=requestAnimationFrame(tick);
}
function tick(time) {
  frame=0;
  if(!scene) return;
  if(animation) {
    const {start,from,to}=animation;
    const t=clamp((time-start)/520,0,1);
    const eased=1-(1-t)**3;
    camera={yaw:lerp(from.yaw,to.yaw,eased),pitch:lerp(from.pitch,to.pitch,eased),distance:lerp(from.distance,to.distance,eased),target:from.target.map((v,i)=>lerp(v,to.target[i],eased))};
    if(t===1) animation=null;
  }
  scene.render(camera);
  updateLabels();
  if(animation) redraw();
}
function moveCamera(to, animate=true) {
  if(!scene) return;
  if(animate && !noMotion.matches) animation={start:performance.now(),from:camera,to};
  else {animation=null;camera=to;}
  redraw();
}
function rebuild() {
  if(!scene) return;
  scene.build(selected,values[selected],focused);
  redraw();
}
function cameraMode(nextFocus, animate=true) {
  if(!scene) return;
  focused=nextFocus;
  root.querySelector('[data-camera=focus]').setAttribute('aria-pressed',String(focused));
  root.querySelector('[data-camera=overview]').setAttribute('aria-pressed',String(!focused));
  text('mode',focused?`STAGE ${pad(selected+1)} / ${stages[selected].short.toUpperCase()}`:'ARCHITECTURE / OVERVIEW');
  rebuild();
  moveCamera(focused?focusCamera():overviewCamera(),animate);
}
function updateInspector(index, focus=true, updateURL=true) {
  selected=clamp(index,0,stages.length-1);
  const stage=stages[selected];
  for(const [field,value] of [['stage-number',`${pad(selected+1)} / ${stages.length}`],['status',stage.status],['stage-title',stage.title],['description',stage.description],['slider-label',stage.control.label]]) text(field,value);
  root.querySelector('#atlas-status').dataset.status=stage.status==='Scoped implementation'?'implemented':'plan';
  const {min,max,step}=stage.control;
  slider.min=min;slider.max=max;slider.step=step;slider.value=values[selected];
  for(const [key,value] of Object.entries(stage.detail)) text(`detail-${key}`,value);
  const evidence=root.querySelector('#atlas-evidence');
  const caption=document.createElement('strong');
  caption.textContent='Project evidence';
  evidence.replaceChildren(caption,stage.evidence);
  const refs=root.querySelector('#atlas-refs');
  refs.replaceChildren(...stage.refs.map(([name,href])=>{
    const link=document.createElement('a');
    link.textContent=name;
    link.href=href;
    return link;
  }));
  refs.hidden=!stage.refs.length;
  stageButtons.forEach((button,i)=>{
    if(i===selected) button.setAttribute('aria-current','step');
    else button.removeAttribute('aria-current');
  });
  root.querySelector('[data-step="-1"]').disabled=selected===0;
  root.querySelector('[data-step="1"]').disabled=selected===stages.length-1;
  setNumbers();
  if(focus) cameraMode(true);
  else rebuild();
  if(updateURL) history.replaceState(null,'',`#${stage.id}`);
}
function setPlay(on) {
  if(timer) {clearTimeout(timer);timer=0;}
  playing=on;
  play.setAttribute('aria-pressed',String(on));
  play.textContent=on?'Pause tour':'Play tour';
  if(!on) return;
  if(selected===stages.length-1) updateInspector(0);
  else cameraMode(true);
  timer=setTimeout(advance,4300);
}
function advance() {
  if(!playing) return;
  updateInspector(selected+1);
  if(selected===stages.length-1) setPlay(false);
  else timer=setTimeout(advance,4300);
}
function choose(index) {setPlay(false);updateInspector(index);}

slider.addEventListener('input',()=>{
  setPlay(false);
  values[selected]=Number(slider.value);
  setNumbers();
  rebuild();
});
play.addEventListener('click',()=>setPlay(!playing));
stageButtons.forEach((button,i)=>button.addEventListener('click',()=>choose(i)));
root.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>choose(selected+Number(button.dataset.step))));
window.addEventListener('hashchange',()=>{
  const target=stages.findIndex(stage=>`#${stage.id}`===location.hash);
  if(target>=0) {setPlay(false);updateInspector(target,true,false);}
});

function disableCanvas() {
  scene=null;
  canvas.tabIndex=-1;
  canvas.setAttribute('aria-hidden','true');
  root.querySelector('.atlas-fallback').hidden=false;
  root.querySelectorAll('[data-camera]').forEach(button=>{button.disabled=true;});
  labels.forEach(label=>{label.hidden=true;});
}
function initCanvas() {
  try {scene=new AtlasScene(canvas,stages);}
  catch {disableCanvas();return;}
  camera=overviewCamera();
  canvas.tabIndex=0;
  canvas.removeAttribute('aria-hidden');
  root.querySelector('.atlas-fallback').hidden=true;
  root.querySelectorAll('[data-camera]').forEach(button=>{button.disabled=false;});
  cameraMode(focused,false);
}
initCanvas();
const first=stages.findIndex(stage=>`#${stage.id}`===location.hash);
updateInspector(Math.max(first,0),first>=0,false);
canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();disableCanvas();});
canvas.addEventListener('webglcontextrestored',initCanvas);
new ResizeObserver(redraw).observe(canvas);

function zoom(factor) {
  if(!scene) return;
  animation=null;
  camera.distance=clamp(camera.distance*factor,2.8,75);
  redraw();
}
function orbit(dx,dy) {
  animation=null;
  camera.yaw-=dx*.007;
  camera.pitch=clamp(camera.pitch+dy*.006,.10,1.46);
  redraw();
}
function pan(dx,dy) {
  if(!scene.view) return;
  animation=null;
  camera.target=scene.pan(camera,dx,dy);
  redraw();
}
root.querySelectorAll('[data-camera]').forEach(button=>button.addEventListener('click',()=>{
  if(!scene) return;
  const command=button.dataset.camera;
  if(command==='focus' || command==='overview') cameraMode(command==='focus');
  if(command==='reset') cameraMode(focused);
  if(command==='in' || command==='out') zoom(command==='in'?.80:1.25);
}));
canvas.addEventListener('wheel',event=>{
  if(!scene || event.ctrlKey) return;
  event.preventDefault();
  const units=event.deltaMode===1?16:1;
  zoom(Math.exp(clamp(event.deltaY*units,-500,500)*.0015));
},{passive:false});
canvas.addEventListener('keydown',event=>{
  if(!scene) return;
  const offsets={ArrowLeft:[-21,0],ArrowRight:[21,0],ArrowUp:[0,-21],ArrowDown:[0,21]};
  if(event.key in offsets) (event.shiftKey?pan:orbit)(...offsets[event.key]);
  else if(event.key==='+' || event.key==='=') zoom(.82);
  else if(event.key==='-') zoom(1.22);
  else if(event.key==='Home') cameraMode(false);
  else if(event.key==='Enter') cameraMode(true);
  else if(event.key===' ') setPlay(!playing);
  else return;
  event.preventDefault();
});

const pointers=new Map();
let moved=false;
const point=event=>[event.clientX,event.clientY];
function pinchGeometry() {
  const [a,b]=[...pointers.values()];
  return {x:(a[0]+b[0])/2,y:(a[1]+b[1])/2,d:Math.hypot(a[0]-b[0],a[1]-b[1])};
}
canvas.addEventListener('pointerdown',event=>{
  if(!scene || (event.pointerType==='mouse' && event.button!==0 && event.button!==2)) return;
  if(pointers.size===0) moved=false;
  else moved=true;
  animation=null;
  pointers.set(event.pointerId,point(event));
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove',event=>{
  if(!scene || !pointers.has(event.pointerId)) return;
  const old=pointers.get(event.pointerId);
  const position=point(event);
  const dx=position[0]-old[0],dy=position[1]-old[1];
  if(Math.abs(dx)+Math.abs(dy)>2) moved=true;
  if(pointers.size>1) {
    const before=pinchGeometry();
    pointers.set(event.pointerId,position);
    const after=pinchGeometry();
    pan(after.x-before.x,after.y-before.y);
    if(before.d>3 && after.d>3) zoom(before.d/after.d);
  } else {
    pointers.set(event.pointerId,position);
    if(event.shiftKey || event.buttons===2) pan(dx,dy);
    else orbit(dx,dy);
  }
});
canvas.addEventListener('pointerup',event=>{
  const wasTracking=pointers.delete(event.pointerId);
  if(!wasTracking || moved || !scene?.view || event.button===2) return;
  const rect=canvas.getBoundingClientRect();
  const index=scene.pick(event.clientX-rect.left,event.clientY-rect.top,focused,selected);
  if(index>=0) choose(index);
});
for(const eventName of ['pointercancel','lostpointercapture']) canvas.addEventListener(eventName,event=>pointers.delete(event.pointerId));
canvas.addEventListener('contextmenu',event=>event.preventDefault());
