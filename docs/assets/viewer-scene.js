const palette = ['#8de3ba', '#edc579', '#c8b5ef', '#9bdaef'];
const ink = '#102d2b';
const mint = '#dceaa0';
const shade = (color, brightness) => color.map(channel => Math.min(1, channel * brightness));
const rgb = color => [1, 3, 5].map(i => parseInt(color.slice(i, i + 2), 16) / 255);
const mix = (a, b, ratio) => a.map((channel, i) => channel * (1 - ratio) + b[i] * ratio);
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, n) => a.map(v => v * n);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = ([x, y, z], [a, b, c]) => [y*c-z*b, z*a-x*c, x*b-y*a];
const norm = a => mul(a, 1 / Math.hypot(...a));
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
const ZERO = [0, 0, 0];
const FOV = 1 / Math.tan(Math.PI / 9);
const faces = [
  [[0, 1, 2, 3], .53], [[4, 7, 6, 5], .73], [[0, 4, 5, 1], .51],
  [[3, 2, 6, 7], 1.04], [[0, 3, 7, 4], .67], [[1, 5, 6, 2], .82],
];
const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const vertexSource = `attribute vec3 a_position;
attribute vec3 a_color;
uniform vec3 u_eye;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_forward;
uniform vec2 u_scale;
varying vec3 v_color;
void main() {
  vec3 delta = a_position - u_eye;
  float depth = dot(delta, u_forward);
  gl_Position = vec4(dot(delta, u_right) * u_scale.x, dot(delta, u_up) * u_scale.y,
                     depth * 1.0013 - 0.20, depth);
  v_color = a_color;
}`;
const fragmentSource = `precision mediump float;
varying vec3 v_color;
void main() { gl_FragColor = vec4(v_color, 1.0); }`;

function buildProgram(gl) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

class Geometry {
  constructor() {
    this.solids = [];
    this.lines = [];
    this.origin = ZERO;
    this.strength = 1;
  }
  vertex(target, point, color) { target.push(...add(point, this.origin), ...color); }
  color(color) { return mix(rgb(ink), rgb(color), this.strength); }
  line(a, b, color) {
    const tint = this.color(color);
    this.vertex(this.lines, a, tint);
    this.vertex(this.lines, b, tint);
  }
  path(points, color) {
    for (let i = 1; i < points.length; i++) this.line(points[i-1], points[i], color);
  }
  triangle(a, b, c, tint) {
    for (const point of [a,b,c]) this.vertex(this.solids, point, tint);
  }
  box([x,y,z], [w,h,d], color, outlined = true) {
    const points = [
      [x-w/2,y-h/2,z-d/2],[x+w/2,y-h/2,z-d/2],[x+w/2,y+h/2,z-d/2],[x-w/2,y+h/2,z-d/2],
      [x-w/2,y-h/2,z+d/2],[x+w/2,y-h/2,z+d/2],[x+w/2,y+h/2,z+d/2],[x-w/2,y+h/2,z+d/2],
    ];
    const tint = this.color(color);
    for (const [[a,b,c,d], light] of faces) {
      this.triangle(points[a], points[b], points[c], shade(tint,light));
      this.triangle(points[a], points[c], points[d], shade(tint,light));
    }
    if (outlined) for (const [a,b] of edges) this.line(points[a],points[b],color);
  }
  gem(center, radius, color) {
    const rim = [[1,0,0],[0,0,1],[-1,0,0],[0,0,-1]].map(p => add(center,mul(p,radius)));
    const top = add(center,[0,radius,0]);
    const low = add(center,[0,-radius,0]);
    const tint = this.color(color);
    rim.forEach((point, i) => {
      this.triangle(point, rim[(i+1)%4], top, shade(tint,[.97,1.15,.77,.84][i]));
      this.triangle(point, rim[(i+1)%4], low, shade(tint,[.66,.8,.53,.65][i]));
    });
  }
  circle(center, radius, color, segments = 28) {
    const points = Array.from({length:segments+1}, (_, i) => {
      const angle = i * Math.PI * 2 / segments;
      return add(center,[Math.cos(angle)*radius,0,Math.sin(angle)*radius]);
    });
    this.path(points,color);
  }
  arrow(a, b, color) {
    this.line(a,b,color);
    const d = norm(sub(b,a));
    const p = norm(cross(d, Math.abs(d[1]) > .9 ? [1,0,0] : [0,1,0]));
    const root = add(b,mul(d,-.23));
    this.line(b,add(root,mul(p,.11)),color);
    this.line(b,add(root,mul(p,-.11)),color);
  }
  axes(center, length = 1.2) {
    for (const [v,color] of [[[length,0,0],'#a9d5bd'],[[0,length,0],'#e0eaaa'],[[0,0,length],'#91c1c8']]) {
      this.arrow(center,add(center,v),color);
    }
  }
}

const nodes = [[-1.18,.66,.50],[-.46,1.25,-.50],[.39,.80,.46],[1.18,1.32,-.50]];
const links = [[0,1],[1,2],[2,3],[0,2],[1,3]];
const structureA = [[-1.1,.7,.6],[-.5,1.1,.25],[.1,.95,-.3],[.75,.68,.25],[1.15,1.0,-.4]];
const structureB = [[-1.15,1.2,.24],[-.45,.8,-.3],[.16,1.17,.2],[.73,1.2,-.35],[1.2,.76,.2]];

function drawInputs(g, c, n) {
  for (let i = 0; i < 6; i++) {
    const x = (i % 3 - 1) * .94;
    const z = i < 3 ? -.42 : .57;
    const tint = i < n ? [c,'#91bed1','#b5d584','#cfbaed','#ebc485','#90dcc6'][i] : '#33534a';
    const height = i < n ? .47 + .16 * (i % 3) : .12;
    g.box([x,.22+height/2,z],[.58,height,.65],tint);
    if (i < n) for (let j=0;j<3;j++) g.line([x-.21,.26+height,z-.21+j*.19],[x+.21,.26+height,z-.21+j*.19],mint);
    g.line([x,.24,z],[0,.24,0],tint);
  }
  g.gem([0,1.25,0],.19,mint);
}
function drawTokenGT(g, c, n) {
  nodes.forEach((p,i) => {
    g.gem(p,.18,c);
    g.box([(i-1.5)*.64,.26,.87],[.43,.16,.28],c);
  });
  links.slice(0,n).forEach(([a,b],i) => {
    g.arrow(nodes[a],nodes[b],mint);
    g.box([(i-2)*.50,.26,1.27],[.35,.16,.24],'#e5c982');
  });
}
function drawConvex(g, c, n) {
  for (let i=0;i<n;i++) {
    const x = (i - (n-1)/2) * .36;
    g.box([x,.40,.60],[.28,.34,.42],c);
    g.box([x,.79,.22],[.28,.20,.25],i%2?'#aecbdd':'#e5c982');
  }
  const path = Array.from({length:n+1},(_,i) => [(i-n/2)*.36,1.20,-.44]);
  path.forEach(point => g.gem(point,.065,mint));
  g.path(path,mint);
  g.line([-1.46,.21,-.70],[1.46,.21,-.70],c);
}
function drawAttention(g, c, v, model) {
  const side = 3;
  for (let x=0;x<side;x++) for (let z=0;z<side;z++) {
    const win = model.winners[x] === z;
    const h = .27 + (win ? .58 : .12) + ((x+z)%3)*.08;
    g.box([(x-1)*.62,.22+h/2,(z-1)*.66],[.48,h,.50],win?mint:c);
  }
  g.gem([v*.55-.18,1.45,-.82],.19,'#e8d293');
  for (let x=0;x<side;x++) g.line([v*.55-.18,1.45,-.82],[(x-1)*.62,1.02,(model.winners[x]-1)*.66],mint);
}
function drawConcepts(g, c, v, model) {
  const origin = [-.55,.40,.35];
  g.axes(origin,1.3);
  for (let i=0;i<27;i++) {
    const point = [Math.sin(i*2.11)*1.03,.48+((i*17)%23)/32,Math.cos(i*3.23)*.9];
    g.gem(point,.05,i%4===0?c:'#7e996b');
  }
  g.arrow(origin,add(origin,[1.16,.62,-.7]),c);
  g.arrow(origin,add(origin,[.31,.9,.69]),'#b8d6e6');
  const dest = add(origin,mul(model.direction,v*.52));
  if (v !== 0) g.arrow(origin,dest,mint);
  g.gem(dest,.19,mint);
}
function drawGFN(g, c, v, model) {
  const origin=[-1.37,.59,.03];
  g.gem(origin,.16,mint);
  for (let i=0;i<3;i++) {
    const s=i-1;
    const path=[origin,[-.67,.83+s*.15,s*.21],[.14,1.02+s*.25,s*.50],[1.25,.66+s*.37,s*.77]];
    g.path(path,c);
    for (let j=1;j<3;j++) g.gem(path[j],.077,c);
    g.gem(path[3],.10+model.masses[i]*.30,mint);
  }
}
function drawForest(g, c, n) {
  for (let t=0;t<2;t++) {
    const root=[-1.12,.44,t*.93-.48];
    g.gem(root,.13,mint);
    for(let i=0;i<3;i++) {
      const mid=[-.30,.75+i*.19,root[2]+(i-1)*.24];
      const end=[1.16,1.14+i*.26,root[2]+(i-1)*.38];
      const tint=i<n?c:'#51605c';
      g.path([root,mid,end],tint);
      g.gem(mid,.087,tint);
      g.gem(end,i<n?.14:.08,tint);
    }
  }
}
function drawAnalogy(g, c, v, model) {
  // Two typed motifs; same ordered incidence, distinct source/target context.
  const colors=['#8de3ba','#edc579','#c8b5ef'];
  for(let row=0;row<2;row++) {
    const motif=[[-1.15,1.25,-.68+row*.86],[-.45,1.25,-.68+row*.86],[.25,1.25,-.68+row*.86]];
    motif.forEach((point,i)=>g.gem(point,.14,colors[i]));
    g.arrow(motif[0],motif[1],colors[0]);g.arrow(motif[1],motif[2],colors[1]);
    if(row===0) motif.forEach(point=>g.line(point,add(point,[0,0,.86]),'#668d84'));
    g.arrow(motif[2],[.8,.65,-.68+row*.86],c);
  }
  g.box([.85,.62,-.18],[.34,.34,.9],mint); // shared embedding route
  model.candidates.forEach((item,i)=>{
    const x=-1.15+i*.76, color=item.refusal?'#da8e88':item.id===model.selected?'#dceaa0':c;
    const h=.12+.48/(1+item.distance);
    g.box([x,.24+h/2,1.02],[.46,h,.35],color);
    g.line([.85,.43,.2],[x,.26,1.02],color);
    if(item.refusal) {g.line([x-.20,.85,.99],[x+.20,1.12,.99],color);g.line([x+.20,.85,.99],[x-.20,1.12,.99],color);}
    else if(item.id===model.selected)g.gem([x,.98,1.02],.09,mint);
  });
}
function drawStructure(g, c, v, model) {
  g.path(structureA,'#567c77');
  g.path(structureB,'#71959b');
  g.path(model.points,mint);
  model.points.forEach((point,i) => g.gem(point,i===2?.23:.15,i%2?c:mint));
  for(let i=0;i<3;i++) g.box([-.68+i*.68,.26,1.12],[.49,.14,.28],c);
}
function drawDynamics(g, c, v, model) {
  for(let k=0;k<2;k++) {
    const center=[k*1.72-.86,.36,0];
    const prob=k ? model.population : 1-model.population;
    for(let i=0;i<4;i++) g.circle(add(center,[0,i*.14,0]),.26+i*.15,c);
    const top=add(center,[0,.28+prob*.88,0]);
    g.line(center,top,mint);
    g.gem(top,.14+prob*.12,mint);
  }
  g.line([-.86,.34,0],[.86,.34,0],c);
}
function drawTeachers(g, c, v, model) {
  for(let i=0;i<2;i++) {
    const x=i*2-1;
    g.box([x,.73,-.42],[.95,1.02,.79],c);
    g.box([x,1.29,-.42],[.97,.12,.83],i?c:mint);
    g.path([[x,.36,0],[x,.42,.52],[0,.42,.76]],c);
  }
  const points=Array.from({length:17},(_,i) => {const x=(i-8)*.18;return [x,.37+x*x*.36,.85];});
  g.path(points,mint);
  g.gem([v*1.35,.37+model.energy*2.6,.85],.14,mint);
}
function drawTraining(g, c, n) {
  for(let i=0;i<n;i++) for(let j=0;j<4;j++) {
    const h=.17+(i+j)%4*.065;
    g.box([(i-(n-1)/2)*.38,.22+h/2,(j-1.5)*.42],[.26,h,.29],c);
  }
  for(let i=0;i<3;i++) g.box([(i-1)*.68,.43+i*.19,-1.18],[.46,.44+i*.38,.35],i===2?mint:c);
}
const drawers=[drawInputs,drawTokenGT,drawConvex,drawAttention,drawConcepts,drawGFN,drawForest,drawAnalogy,drawStructure,drawDynamics,drawTeachers,drawTraining];

export function toyModel(index, value) {
  if (index===3) {
    const query=[value,.3,-.25];
    const keys=[[0,.1,.4],[.7,.4,-.1],[1.3,-.25,.35]];
    const values=[[.9,.1,.2],[.3,.5,.1],[.05,.15,1.1]];
    const scores=keys.map(key => {
      const difference=sub(query,key);
      return Math.min(...difference)-Math.max(...difference);
    });
    const terms=query.map((_,i)=>keys.map((_,j)=>scores[j]+values[j][i]));
    const result=terms.map(row=>Math.max(...row));
    return {result,winners:terms.map((row,i)=>row.indexOf(result[i]))};
  }
  if(index===4) return {direction:[.8,.5,-.3]};
  if(index===5) return {masses:[value,(1-value)*value,(1-value)**2]};
  if(index===7) {
    const displacement=[.8,.5,-.3],target=[.2,-.1,.4];
    const query=add(target,mul(displacement,value));
    const candidates=[
      {id:'A',vector:[1.05,.45,.08],refusal:null,label:'Matched motif / endpoint'},
      {id:'B',vector:[.8,.3,.16],refusal:'Reversed receptor → ligand edge',label:'Nearest, wrong topology'},
      {id:'C',vector:[1,.4,.1],refusal:'Affinity endpoint substituted for cell response',label:'Nearest, wrong endpoint'},
      {id:'D',vector:[.36,0,.34],refusal:null,label:'Matched motif / endpoint'}
    ].map(x=>({...x,distance:x.vector.reduce((sum,v,i)=>sum+(v-query[i])**2,0)}));
    const selected=[...candidates].filter(x=>!x.refusal).sort((a,b)=>a.distance-b.distance||a.id.localeCompare(b.id))[0].id;
    return {displacement,query,candidates,selected};
  }
  if(index===8) return {points:structureA.map((point,i)=>mix(point,structureB[i],value)),reference:structureA};
  if(index===9) return {population:1/(1+Math.exp(.06/(8.617333262e-5*value)))};
  if(index===10) return {energy:.5*.4*value**2,force:-.4*value};
  return {};
}

function stagePosition(index) {
  const row=Math.floor(index/4);
  const column=row%2===0?index%4:3-index%4;
  return [(column-1.5)*4.55,0,(row-1)*4.25];
}

export class AtlasScene {
  constructor(canvas, stages) {
    this.canvas = canvas;
    this.stages = stages;
    this.centers = stages.map((_,i)=>stagePosition(i));
    this.gl = canvas.getContext('webgl',{alpha:true, antialias:true, premultipliedAlpha:false});
    if (!this.gl) throw new Error('WebGL unavailable');
    const gl = this.gl;
    this.program = buildProgram(gl);
    this.position = gl.getAttribLocation(this.program,'a_position');
    this.color = gl.getAttribLocation(this.program,'a_color');
    this.uniforms = Object.fromEntries(['eye','right','up','forward','scale'].map(name => [name,gl.getUniformLocation(this.program,`u_${name}`)]));
    this.buffers = [gl.createBuffer(), gl.createBuffer()];
    gl.useProgram(this.program);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enableVertexAttribArray(this.position);
    gl.enableVertexAttribArray(this.color);
    gl.clearColor(0,0,0,0);
  }
  build(selected, value, focused) {
    const g = new Geometry();
    for(let x=-12;x<=12;x++) g.line([x,-.26,-9],[x,-.26,9],x%4===0?'#3e6757':'#2c4d42');
    for(let z=-9;z<=9;z++) g.line([-12,-.26,z],[12,-.26,z],z%4===0?'#3e6757':'#2c4d42');
    for (let i=1;i<this.centers.length;i++) {
      const a=add(this.centers[i-1],[0,.06,0]);
      const b=add(this.centers[i],[0,.06,0]);
      const middle=mix(a,b,.50);
      const end=mix(a,b,.59);
      g.path([a,b], i===selected || i===selected+1 ? mint : '#78ab93');
      g.arrow(middle,end,i===selected || i===selected+1 ? mint : '#78ab93');
    }
    this.stages.forEach((stage,i) => {
      g.origin = this.centers[i];
      g.strength = focused && i!==selected ? .36 : 1;
      const color = palette[stage.group];
      const active = i===selected;
      g.box([0,-.08,0],[3.55,.18,3.17],active?'#788b60':'#476b5d');
      g.box([0,.07,0],[3.33,.14,2.97],'#254b40',false);
      for(const x of [-1.59,1.59]) for(const z of [-1.40,1.40]) g.box([x,.17,z],[.13,.08,.13],active?mint:color,false);
      if (active) {
        g.path([[-1.67,.16,-1.49],[1.67,.16,-1.49],[1.67,.16,1.49],[-1.67,.16,1.49],[-1.67,.16,-1.49]],mint);
      }
      const v=active?value:stage.control.value;
      drawers[i](g,color,v,toyModel(i,v));
      for(let lane=0;lane<3;lane++) {
        const y=.32+lane*.24;
        g.box([-1.42,y,-1.18],[.18,.14,.34],lane===1?mint:color);
        g.line([-1.30,y,-1.18],[-.94,y,-1.18],color);
      }
    });
    const gl=this.gl;
    this.counts=[g.solids.length/6,g.lines.length/6];
    [g.solids,g.lines].forEach((vertices,i)=>{
      gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers[i]);
      gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.DYNAMIC_DRAW);
    });
  }
  render(camera) {
    const canvas=this.canvas, gl=this.gl;
    const rect=canvas.getBoundingClientRect();
    const dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(rect.width*dpr));
    const height=Math.max(1,Math.round(rect.height*dpr));
    if(canvas.width!==width || canvas.height!==height) {
      canvas.width=width;
      canvas.height=height;
      gl.viewport(0,0,width,height);
    }
    const {yaw,pitch,distance,target}=camera;
    const direction=[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)];
    const eye=add(target,mul(direction,distance));
    const forward=mul(direction,-1);
    const right=norm(cross(forward,[0,1,0]));
    const up=cross(right,forward);
    const scale=[FOV*height/width,FOV];
    this.view={eye,right,up,forward,scale,width:rect.width,height:rect.height};
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    for(const [key,value] of Object.entries({eye,right,up,forward})) gl.uniform3fv(this.uniforms[key],value);
    gl.uniform2fv(this.uniforms.scale,scale);
    this.buffers.forEach((buffer,i)=>{
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
      gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,24,0);
      gl.vertexAttribPointer(this.color,3,gl.FLOAT,false,24,12);
      gl.drawArrays(i===0?gl.TRIANGLES:gl.LINES,0,this.counts[i]);
    });
  }
  project(point) {
    const {eye,right,up,forward,scale,width,height}=this.view;
    const delta=sub(point,eye), depth=dot(delta,forward);
    if(depth<.16) return null;
    return [(1+dot(delta,right)*scale[0]/depth)*width/2,(1-dot(delta,up)*scale[1]/depth)*height/2];
  }
  labelPoints() { return this.centers.map(center=>this.project(add(center,[0,1.82,0]))); }
  pan(camera, dx, dy) {
    const {right,up,height}=this.view;
    const scale=camera.distance*2/FOV/height;
    return add(camera.target,add(mul(right,-dx*scale),mul(up,dy*scale))).map(v=>clamp(v,-22,22));
  }
  pick(x,y,focused,selected) {
    let best = focused ? 80 : 48;
    let index=-1;
    this.centers.forEach((center,i)=>{
      if(focused && i!==selected) return;
      for(const location of [add(center,[0,.75,0]),add(center,[0,1.82,0])]) {
        const point=this.project(location);
        if(!point) continue;
        const distance=Math.hypot(x-point[0],y-point[1]);
        if(distance<best) {best=distance;index=i;}
      }
    });
    return index;
  }
}
