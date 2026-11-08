"""LIVECell morphology tasks bridging image/polygon sidecars and shared graphs.

Unknown target topology is kept in a separate supervision graph. This is not a
ConditionalTask with known target topology, a rasterizer or a neural image head.
Bounding boxes summarize all source vertices; authoritative polygons remain in
LiveCellExample. No adjacency/contact, physical scale or biological pairing is
inferred. Source identities and split keys are audit sidecars, never features.
"""
from dataclasses import dataclass
import json
from .livecell import LiveCellExample
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import Provenance, LeakageKey
from .splits import audit_splits


REVISION = 'LIVECell_dataset_2021;repo=1915b2511dcab0dd5dd71bb4204f91e502f8c66d'


def _measurement(name, value, units=None):
    return Measurement(name, value, units, 'missing' if value is None else 'observed', value is not None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class LiveCellGraphTask:
    example: LiveCellExample
    direction: str

    def __post_init__(self):
        if not isinstance(self.example, LiveCellExample):
            raise ValueError('validated LIVECell example required')
        self.example.input_metadata(self.direction)

    def _graph(self, include_cells):
        x=self.example
        context=x.context
        # Well/location/parent/crop are audit identities, not learned features.
        nodes=[Node('microscopy_image', tuple(_measurement(k,v,u) for k,v,u in (
            ('width',x.width,'pixel'), ('height',x.height,'pixel'),
            ('cell_line',context['cell_line'],None), ('microscopy',context['microscopy'],None),
            ('acquisition_elapsed_time',context['elapsed_time'],None),
            ('pixel_size_um',None,'um/pixel'), ('perturbation',None,None),
            ('expression',None,None), ('organism',None,None))))]
        edges=[]
        if include_cells:
            for obj in x.polygons:
                xs=[p[i] for p in obj for i in range(0,len(p),2)]
                ys=[p[i] for p in obj for i in range(1,len(p),2)]
                nodes.append(Node('observed_whole_cell',tuple(_measurement(k,v,u) for k,v,u in (
                    ('x_min',min(xs),'pixel'), ('y_min',min(ys),'pixel'),
                    ('x_max',max(xs),'pixel'), ('y_max',max(ys),'pixel'),
                    ('vertex_count',len(xs),'dimensionless'),
                    ('component_count',len(obj),'dimensionless')))))
                edges.append(Edge(0,len(nodes)-1,'contains_observed_cell',True))
        return GraphRecord('LIVECell2021:'+x.file_name,tuple(nodes),tuple(edges),
            (Provenance('LIVECell_dataset_2021',REVISION,x.file_name,x.source_sha256,'CC-BY-NC-4.0'),),
            (LeakageKey('LIVECell2021:well-v1',context['leakage_group']),
             LeakageKey('LIVECell2021:parent-v1',context['parent_image'])))

    def input_graph(self):
        return self._graph(self.direction=='mask_to_image')

    def target_graph(self):
        return self._graph(self.direction=='image_to_mask')

    def input_metadata(self):
        # No source/target hash, mask count, geometry or evaluation split here.
        return dict(version=1,adapter='livecell-graph-task-v1',direction=self.direction,
                    image_shape=[self.example.height,self.example.width],
                    units='pixel',geometry='whole-cell-manual-2D-polygons',
                    target_topology_known=False if self.direction=='image_to_mask' else True)

    def project(self, *, device='cpu', for_training=False):
        if for_training:
            self.example.require_training()
        result=self.example.project(self.direction,device=device)
        result.update(input_graph=self.input_graph(),target_graph=self.target_graph())
        return result

    def canonical_bytes(self):
        # Full snapshot is audit/resume-only, never an encoder input.
        return _canonical(dict(version=1,adapter='livecell-graph-task-v1',direction=self.direction,
                               example=json.loads(self.example.canonical_bytes())))

    @classmethod
    def from_bytes(cls, raw):
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate JSON key')
                d[k]=v
            return d
        try:
            d=json.loads(raw,object_pairs_hook=unique)
            if (set(d)!={'version','adapter','direction','example'} or type(d['version']) is not int
                    or d['version']!=1 or d['adapter']!='livecell-graph-task-v1'):
                raise ValueError('unsupported morphology task snapshot')
            return cls(LiveCellExample.from_bytes(_canonical(d['example'])),d['direction'])
        except (TypeError,KeyError,UnicodeError) as exc:
            raise ValueError('malformed morphology task') from exc

    def assert_resume_matches(self, other):
        if not isinstance(other,LiveCellGraphTask) or self.canonical_bytes()!=other.canonical_bytes():
            raise ValueError('resume source/task mismatch')


def audit_morphology_splits(tasks):
    """Report conservative source well/parent overlap; never change official splits.

    Duplicate task directions for an identical image collapse to one audit row.
    This bounded identity audit does not perform image similarity/donor matching.
    """
    records={}; assignments={}; snapshots={}
    for t in tasks:
        if not isinstance(t,LiveCellGraphTask): raise ValueError('morphology task required')
        g=t.input_graph(); identity=g.record_id
        snapshot=t.example.canonical_bytes()
        if identity in snapshots and snapshots[identity]!=snapshot:
            raise ValueError('conflicting source image snapshots')
        snapshots[identity]=snapshot
        records[identity]=g
        assignments[identity]='validation' if t.example.split=='val' else t.example.split
    return audit_splits(records.values(),assignments)
