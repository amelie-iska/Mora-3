// Orthographic projection of a small stack of tensor grids. Tables retain the exact values.
const namespace = "http://www.w3.org/2000/svg";
function svgElement(tag, attributes, text) {
  const element = document.createElementNS(namespace, tag);
  for (const [key,value] of Object.entries(attributes)) element.setAttribute(key,value);
  if (text !== undefined) element.textContent=text;
  return element;
}

export function tensorScene(svg, tensors, selected, angle, tilt, links = []) {
  const a=angle*Math.PI/180;
  const pitch=tilt*Math.PI/180;
  const project=([x,y,z]) => {
    const depth=x*Math.sin(a)+z*Math.cos(a);
    return [x*Math.cos(a)-z*Math.sin(a), -y*Math.cos(pitch)+depth*Math.sin(pitch), depth*Math.cos(pitch)+y*Math.sin(pitch)];
  };
  const facets=[];
  const labels=[];
  const centers=new Map();
  for (const [i,tensor] of tensors.entries()) {
    const {label,values,selection,mask}=tensor;
    const offset=(i-(tensors.length-1)/2)*3.7;
    const rows=values.length;
    const cols=values[0].length;
    const base=(r,c,y)=>{
      const [x,vertical,depth]=project([(c-cols/2)*2.55/cols,y,(r-rows/2)*2.65/rows]);
      return [x+offset,vertical,depth];
    };
    labels.push([base(0,0,.25),label]);
    values.forEach((row,r)=>row.forEach((value,c)=>{
      const masked=mask?.(r,c);
      const h=masked ? .035 : Math.min(.42,.08+Math.abs(value)*.12);
      const top=[base(r,c,h),base(r,c+.93,h),base(r+.93,c+.93,h),base(r+.93,c,h)];
      const low=[base(r,c,0),base(r,c+.93,0),base(r+.93,c+.93,0),base(r+.93,c,0)];
      const active=selection?.[0]===r&&(selection[1]===undefined||selection[1]===c);
      const style=masked?"masked":active?"active":value<0?"negative":selected===tensor.head?"head-selected":"value";
      const name=label+", row "+(r+1)+", channel "+(c+1)+": "
        +(masked?"masked":Number.isFinite(value)?value.toFixed(4):"−∞");
      const cell=[i,r,c];
      centers.set(cell.join(","),base(r+.465,c+.465,h));
      facets.push({points:top,style,name,cell});
      for (let f=0;f<4;f++) {
        const g=(f+1)%4;
        facets.push({points:[top[f],low[f],low[g],top[g]],style:style+" side"});
      }
    }));
  }
  const points=facets.flatMap(f=>f.points);
  const maxX=Math.max(...points.map(p=>Math.abs(p[0])));
  const minY=Math.min(...points.map(p=>p[1]));
  const maxY=Math.max(...points.map(p=>p[1]));
  const scale=Math.min(320/Math.max(maxX,1),126/Math.max(maxY-minY,.8));
  const position=([x,y])=>[370+x*scale,118+(y-(minY+maxY)/2)*scale];
  const depth=facet=>facet.points.reduce((sum,p)=>sum+p[2],0)/facet.points.length;
  const shapes=facets.sort((a,b)=>depth(b)-depth(a)).map(f=>{
    const polygon=svgElement("polygon",{points:f.points.map(p=>position(p).join(",")).join(" "),class:f.style});
    if(f.cell) {
      polygon.setAttribute("data-cell",f.cell.join(","));
      polygon.append(svgElement("title",{},f.name));
    }
    return polygon;
  });
  const definitions=svgElement("defs",{});
  const marker=svgElement("marker",{id:"trace-arrow",viewBox:"0 0 10 10",refX:9,refY:5,
    markerWidth:5,markerHeight:5,orient:"auto"});
  marker.append(svgElement("path",{d:"M 0 0 L 10 5 L 0 10 z"}));
  definitions.append(marker);
  const arrows=links.map(({from,to,kind})=>{
    const [x1,y1]=position(centers.get(from.join(",")));
    const [x2,y2]=position(centers.get(to.join(",")));
    const middle=(x1+x2)/2;
    return svgElement("path",{class:"trace-link "+kind,d:`M ${x1} ${y1} C ${middle} ${y1-17}, ${middle} ${y2-17}, ${x2} ${y2}`,
      "marker-end":"url(#trace-arrow)","data-dependency":kind});
  });
  svg.replaceChildren(definitions,...shapes,...arrows,...labels.map(([p,label])=>{
    const [x,y]=position(p);
    return svgElement("text",{x,y:y-15},label);
  }));
  svg.setAttribute("aria-label",tensors.map(t=>t.label+" "+t.values.length+" by "+t.values[0].length).join(". ")
    + ". Cell height shows activation magnitude. " + (links.length ? "Arrows link selected operands to their result. " : "")
    + "Exact values are in the tables.");
}

export function graphScene(svg, graph, selected) {
  const count=graph.identifiers.length;
  const positions=Array.from({length:count},(_,i)=>[310+(i-(count-1)/2)*176,75]);
  const names=graph.tokens.slice(1,count+1);
  const definitions=svgElement("defs",{});
  const marker=svgElement("marker",{id:"trace-graph-arrow",viewBox:"0 0 10 10",refX:9,refY:5,
    markerWidth:6,markerHeight:6,orient:"auto"});
  marker.append(svgElement("path",{d:"M 0 0 L 10 5 L 0 10 z"}));
  definitions.append(marker);
  const parallel=new Map();
  const edges=graph.endpoints.slice(count).map(([a,b],i)=>{
    const token=count+1+i;
    const x=positions[a][0], target=positions[b][0];
    const pair=[Math.min(a,b),Math.max(a,b)].join(":");
    const order=parallel.get(pair)??0;
    parallel.set(pair,order+1);
    const bend=(order%2===0 ? -1 : 1)*(24+10*Math.floor(order/2)+30*(Math.abs(a-b)>1));
    const direction=Math.sign(target-x);
    const start=x+direction*19, end=target-direction*19;
    const y=75+bend;
    const d=a===b ? `M ${x-12} 61 C ${x-45-order*7} ${30-order*5}, ${x+44+order*7} ${30-order*5}, ${x+12} 61`
      : `M ${start} 75 Q ${(x+target)/2} ${y} ${end} 75`;
    const path=svgElement("path",{d,"data-token":token,class:selected===token?"selected":"",
      "marker-end":"url(#trace-graph-arrow)"});
    path.append(svgElement("title",{},graph.tokens[token]+" · edge "+(i+1)));
    return path;
  });
  const nodes=positions.map(([x,y],i)=>{
    const group=svgElement("g",{"data-token":i+1,class:selected===i+1?"selected":""});
    const characters=Array.from(names[i]);
    const label=characters.length>17 ? characters.slice(0,16).join("")+"…" : names[i];
    group.append(svgElement("circle",{cx:x,cy:y,r:17}),svgElement("text",{x,y:y+35},label),
      svgElement("title",{},names[i]));
    return group;
  });
  svg.replaceChildren(definitions,...edges,...nodes);
  svg.setAttribute("aria-label",names.join(", ")+". "+edges.length+" directed edges. Node and edge selection is also available in the token menu.");
}
