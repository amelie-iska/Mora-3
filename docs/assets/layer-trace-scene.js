// Orthographic projection of a small stack of tensor grids. Tables retain the exact values.
const namespace = "http://www.w3.org/2000/svg";
function svgElement(tag, attributes, text) {
  const element = document.createElementNS(namespace, tag);
  for (const [key,value] of Object.entries(attributes)) element.setAttribute(key,value);
  if (text !== undefined) element.textContent=text;
  return element;
}

export function tensorScene(svg, tensors, selected, angle, tilt) {
  const a=angle*Math.PI/180;
  const pitch=tilt*Math.PI/180;
  const project=([x,y,z]) => {
    const depth=x*Math.sin(a)+z*Math.cos(a);
    return [x*Math.cos(a)-z*Math.sin(a), -y*Math.cos(pitch)+depth*Math.sin(pitch), depth*Math.cos(pitch)+y*Math.sin(pitch)];
  };
  const facets=[];
  const labels=[];
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
      facets.push({points:top,style,name,cell:[i,r,c]});
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
  svg.replaceChildren(...shapes,...labels.map(([p,label])=>{
    const [x,y]=position(p);
    return svgElement("text",{x,y:y-15},label);
  }));
  svg.setAttribute("aria-label",tensors.map(t=>t.label+" "+t.values.length+" by "+t.values[0].length).join(". ")
    + ". Cell height shows activation magnitude. Exact values are in the tables.");
}
