"""Quarantined sci-Plex2 CellTiter-Glo scalar projection, not transcriptomics.

Source doses have no established physical units in this table. Do not infer
exposure time, baseline matching, structures, or molecular identity from aliases.
No training is authorized by a public URL or this parser. No upstream code reused.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import Provenance, LeakageKey
from .splits import leakage_groups
from .tasks import ConditionalTask

COLUMNS = ['CTG', 'replicate', 'cell_type', 'treatment', 'dose', 'real_dose']
SOURCE = 'sci-Plex2-CTG'
STUDY = 'GSE139944:GSM4150377'


def convert(raw, *, revision):
    """Parse all 1..256 source rows; invalid/ambiguous tables fail as a unit."""
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Pinned repository revision required')
    if not isinstance(raw, bytes) or len(raw) > 64 * 2**10:
        raise ValueError('Require source bytes within 64 KiB budget')
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=None))
        if reader.fieldnames != COLUMNS:
            raise ValueError('Unexpected or duplicate CSV headers')
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise ValueError('Malformed CSV') from exc
    if not 1 <= len(rows) <= 256:
        raise ValueError('Require 1..256 rows')
    records, seen = [], set()
    sha = hashlib.sha256(raw).hexdigest()
    for row_number, row in enumerate(rows, start=2):
        try:
            if set(row) != set(COLUMNS) or any(v is None or not v.strip() for v in row.values()):
                raise ValueError('Missing/extra CSV values')
            if row['cell_type'] != 'A549' or row['treatment'] not in ('Dex','BMS','Nutlin','SAHA'):
                raise ValueError('Unsupported cell or treatment alias')
            if not re.fullmatch('[1-9][0-9]*', row['replicate']):
                raise ValueError('Positive integer replicate required')
            ctg, dose, real_dose = (float(row[k]) for k in ('CTG','dose','real_dose'))
            if not all(math.isfinite(x) and x >= 0 for x in (ctg,dose,real_dose)):
                raise ValueError('Nonfinite or negative readout/dose')
            if (dose == 0) != (real_dose == 0):
                raise ValueError('Inconsistent zero-dose control')
            identity = (row['cell_type'],row['treatment'],int(row['replicate']),dose)
            if identity in seen:
                raise ValueError('Duplicate assay identity')
            seen.add(identity)
        except (TypeError, OverflowError) as exc:
            raise ValueError('Malformed source row') from exc
        observed = lambda name,value,units=None: Measurement(name,value,units,'observed',True)
        missing = lambda name,units=None: Measurement(name,None,units,'missing',False)
        nodes = (
            Node('chemical_intervention', (observed('treatment',row['treatment']),
                 missing('structure'))),
            Node('assay_context', (observed('cell_type',row['cell_type']),
                 observed('study',STUDY),observed('assay','CellTiter-Glo'),
                 observed('replicate',int(row['replicate']),'source_replicate_index'),
                 observed('dose',dose,'source-unspecified'),
                 observed('real_dose',real_dose,'source-unspecified'),
                 missing('exposure_time','hour'))),
            Node('cellular_assay_readout', (observed('CTG',ctg,'relative_viability'),
                 observed('response_state','vehicle_control' if dose == 0 else 'treated'),
                 missing('baseline_CTG','relative_viability'),missing('expression'))))
        p = Provenance(SOURCE,revision,f'small_screen/sciPlex2_CTG_results.csv:row:{row_number}',
                       sha,'UNRESOLVED-NO-TRAINING')
        records.append(GraphRecord(f'sciplex2-ctg:{sha}:row:{row_number}', nodes,
            (Edge(0,1,'applied_in_assay',True),Edge(1,2,'has_assay_readout',True)),(p,),
            (LeakageKey('study:sciplex:v1',STUDY),
             LeakageKey('chemical:sciplex2:alias:v1',row['treatment']))))
    return tuple(records)


def tasks(record):
    """Two serialization-ready masks, never a license or training-readiness grant.

Inverse treatment prediction hides both dose columns to avoid drug-specific dose
scaling as a direct alias proxy. Missing and response-state fields remain unused.
"""
    if any(p.source != SOURCE for p in record.provenance):
        raise ValueError('Expected sci-Plex CTG record')
    fields = [(('node',i,m.name),m) for i,n in enumerate(record.nodes) for m in n.measurements]
    result = []
    for target in [('node',2,'CTG'),('node',0,'treatment')]:
        roles=[]
        for ref,m in fields:
            role='condition' if m.available else 'unused'
            if ref == ('node',2,'response_state') or (target[1]==0 and ref[2] in ('dose','real_dose')):
                role='unused'
            if ref == target:
                role='target'
            roles.append((ref,role))
        context = tuple((name,('node',1,name)) for name in
                        ('cell_type','study','assay','replicate') +
                        (('dose','real_dose') if target[1] == 2 else ()))
        task = ConditionalTask(tuple(roles),context,topology_known=True)
        task.input_fields(record)
        task.available_targets(record)
        result.append(task)
    return tuple(result)


def require_training_eligible(manifest):
    """Current source disposition always fails closed pending license resolution."""
    raise ValueError('Unresolved source license: this adapter emits quarantine artifacts only')


def curate(source, output, *, expected_sha256, revision):
    """Save a NEW non-training artifact, source checksums and explicit task gaps."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    if source.stat().st_size > 64*2**10:
        raise ValueError('Source exceeds budget')
    if shutil.disk_usage(output.parent if output.parent.exists() else source.parent).free < 200*2**30 + 2**20:
        raise ValueError('Must retain 200 GiB free')
    raw = source.read_bytes()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    if sha(raw) != expected_sha256:
        raise ValueError('Source checksum mismatch')
    records = convert(raw,revision=revision)
    packed = b''.join(r.canonical_bytes()+b'\n' for r in records)
    task_bytes = b''.join(json.dumps(dict(record_id=r.record_id,
        masks=[json.loads(t.canonical_bytes()) for t in tasks(r)]),sort_keys=True).encode()+b'\n'
        for r in records)
    manifest = dict(adapter='sciplex2-ctg-quarantine-v1', revision=revision,
        source_path=str(source.resolve()),source_sha256=sha(raw),accepted=len(records),rejected=[],
        seed=None,license_status='unresolved',quarantined=True,training_ready=False,
        split_assignments={r.record_id:'quarantine' for r in records},
        leakage_groups=leakage_groups(records),
        split_audit_scope='All rows share study group; no held-out split; chemical similarity unresolved',
        pairing_audit=dict(source_rows=len(records),cross_source_joins=0,
            rule='Each original row retains treatment, doses, cell type, replicate and CTG readout',
            baseline_joins=0,zero_dose_rows=sum(r.nodes[1].measurements[4].value==0 for r in records)),
        coverage=['source treatment alias','cell line','replicate','source doses','relative CTG viability'],
        missing=['resolved reuse license','physical dose units','assay exposure time',
                 'matched baseline','chemical structure/SELFIES','expression','morphology','GRN','held-out evaluation'],
        task_coverage=[dict(conditions='treatment + source dose + assay context',target='relative CTG',
                            status='mask supported; quarantined; no trained response head'),
                       dict(conditions='relative CTG + assay context; doses hidden',target='treatment alias',
                            status='mask supported; no molecular generation or causal validation')],
        attribution='Srivatsan et al., Science 367:45-51 (2020), doi:10.1126/science.aax6234; original authors CSV; scalar projection',
        checksum_files={'records.jsonl':sha(packed),'tasks.jsonl':sha(task_bytes)},
        derived_output_budget_bytes=2**20)
    if len(packed)+len(task_bytes)>2**20:
        raise ValueError('Derived output exceeds 1 MiB budget')
    if source.read_bytes()!=raw:
        raise RuntimeError('Source changed during curation')
    manifest['source_unchanged']=True
    output.mkdir(parents=True,exist_ok=False)
    (output/'records.jsonl').write_bytes(packed)
    (output/'tasks.jsonl').write_bytes(task_bytes)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
