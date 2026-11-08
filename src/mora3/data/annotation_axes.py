"""Source-bound complete annotation target registry, never an admission grant.

Inventory rows retain their source evidence/scope counts verbatim. They are target
schema metadata, not predictor features or per-example supervision. Raw source
assertions and task-specific admission remain with the native corpus reader.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

CATEGORIES = frozenset(('EC', 'GO', 'cofactor_source_ID', 'feature_ligand_source_ID',
    'feature_type', 'keyword_source_ID', 'subcellular_location',
    'subcellular_orientation', 'subcellular_topology',
    *('domain_family_reference:'+x for x in
      ('CDD','Gene3D','InterPro','PANTHER','PROSITE','Pfam','SUPFAM'))))


@dataclass(frozen=True)
class AnnotationAxes:
    """Immutable canonical serialization; callers receive detached metadata."""
    specification_json: str

    @property
    def categories(self):
        return tuple(json.loads(self.specification_json)['rows'])

    def rows(self, category):
        return json.loads(self.specification_json)['rows'][category]

    def axis(self, category):
        return tuple(row['axis_id'] for row in self.rows(category))

    def binding(self):
        return json.loads(self.specification_json)['binding']


def read_annotation_axes(manifest_path, *, expected_sha256=None):
    """Verify exact manifest/category bytes and complete ordered target axes.

    Paths follow corpus project-relative conventions (absolute fixture paths are
    also supported). No source scans, writes, vocabulary fitting or top-k filter.
    Source manifests are bound as references, not asserted reviewed/admitted here.
    """
    path=Path(manifest_path);raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError('annotation registry manifest hash mismatch')
    m=json.loads(raw)
    if m.get('schema')!='mora3.TrainSourceAnnotationAxes/v1':
        raise ValueError('unsupported annotation registry schema')
    entries=m.get('axis_files',[])
    names=[e['axis'] for e in entries]
    if len(names)!=len(CATEGORIES) or set(names)!=CATEGORIES:
        raise ValueError('complete unique annotation categories required')
    if set(m.get('labels_by_axis',{}))!=CATEGORIES:
        raise ValueError('complete category counts required')
    rows={};files=[]
    for e in entries:
        source=Path(e['path']);payload=source.read_bytes()
        if len(payload)!=e['bytes'] or hashlib.sha256(payload).hexdigest()!=e['sha256']:
            raise ValueError('annotation category file hash/bytes mismatch')
        # Read verified bytes rather than reopening a potentially changed file.
        import pyarrow as pa
        data=pq.read_table(pa.BufferReader(payload)).to_pylist()
        ids=[x['axis_id'] for x in data]
        if (len(ids)!=e['labels'] or len(ids)!=m['labels_by_axis'][e['axis']]
                or not ids or any(not isinstance(x,str) or not x for x in ids)
                or len(set(ids))!=len(ids) or ids!=sorted(ids)):
            raise ValueError('annotation category axis/count/order mismatch')
        if any(x.get('absence_is_negative') is not False
               or x.get('new_supervision_admitted') is not False for x in data):
            raise ValueError('registry must preserve unknown and admission policy')
        if sum(x['source_assertion_views'] for x in data)!=e['source_assertion_views']:
            raise ValueError('annotation source assertion count mismatch')
        rows[e['axis']]=data;files.append(dict(e))
    binding=dict(version=1,manifest_sha256=digest,axis_files=files,
                 sources={k:m[k] for k in ('master','fullKB_source','GOA_source')},
                 registry_is_admission=False,absence_is_negative=False)
    return AnnotationAxes(json.dumps(dict(rows=rows,binding=binding),sort_keys=True,
                                    separators=(',',':'),allow_nan=False))
