"""Version 1 local pixel-center raster convention; not COCO mask API parity.

Even-odd interior with boundary inclusion, union of components per instance.
Degenerate zero-area components contribute nothing. Float64 calculations on
CPU/CUDA, no resize/crop, no exclusivity assignment for overlapping cells.
Original polygons remain authoritative; this is derived supervision only.
"""
import torch
from .livecell import LiveCellExample
from .livecell_graph import LiveCellGraphTask

CONVENTION = 'pixel-center-even-odd-boundary-inclusive-f64-v1'


def rasterize_instances(example, *, device='cpu'):
    """Return bool [instances,height,width], retaining overlapping objects.

    Scan edges sequentially: workspace is O(HW), output O(instances*HW).
    Exact floating equality defines boundary membership; no epsilon dilation.
    """
    if not isinstance(example, LiveCellExample):
        raise ValueError('validated LIVECell example required')
    h,w=example.height,example.width
    yy,xx=torch.meshgrid(torch.arange(h,dtype=torch.float64,device=device)+.5,
                         torch.arange(w,dtype=torch.float64,device=device)+.5,indexing='ij')
    result=torch.zeros((len(example.polygons),h,w),dtype=torch.bool,device=device)
    for i,components in enumerate(example.polygons):
        for coords in components:
            points=list(zip(coords[::2],coords[1::2]))
            edges=list(zip(points,points[1:]+points[:1]))
            area=sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)
            if area==0: continue
            inside=torch.zeros((h,w),dtype=torch.bool,device=device)
            boundary=torch.zeros_like(inside)
            for (ax,ay),(bx,by) in edges:
                cross=(xx-ax)*(by-ay)-(yy-ay)*(bx-ax)
                boundary |= ((cross==0)&(xx>=min(ax,bx))&(xx<=max(ax,bx))
                             &(yy>=min(ay,by))&(yy<=max(ay,by)))
                if ay!=by:
                    inside ^= (((ay>yy)!=(by>yy)) & (xx < ax+(yy-ay)*(bx-ax)/(by-ay)))
            result[i] |= inside | boundary
    return result


def raster_task(task, *, device='cpu', for_training=False):
    """Project aligned derived masks into the existing two task directions.

    Graph target topology stays separate; source hashes/instance count are not
    image inputs. Training guards run before expensive projection.
    """
    if not isinstance(task,LiveCellGraphTask):
        raise ValueError('validated graph task required')
    if for_training: task.example.require_training()
    image=task.example.project(task.direction,device=device)
    masks=rasterize_instances(task.example,device=device)
    if task.direction=='image_to_mask':
        inputs=image['inputs']; targets={'instances':masks}
    else:
        inputs={'instances':masks}; targets=image['targets']
    return dict(inputs=inputs,targets=targets,input_graph=task.input_graph(),
                target_graph=task.target_graph(),raster_convention=CONVENTION)
