"""Bounded GPerturb Figshare v3 sci-Plex2 CSV inspection reader.

All genes are preserved, with original selected-count units. Source-scaled dose
coordinates are not physical concentrations. Returns a paired inspection schema,
not GraphRecord tensors or training eligibility. No upstream code is copied.
"""
import csv
import io
import math


def _table(raw, *, max_bytes=4 * 2**20):
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise ValueError('Require bytes within the per-file inspection budget')
    try:
        return list(csv.reader(io.StringIO(raw.decode('utf-8')), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError('Malformed CSV') from exc


def _number(text):
    try:
        x = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid numeric value') from exc
    if not math.isfinite(x) or x < 0:
        raise ValueError('Require finite nonnegative value')
    return x


def _integer(text):
    x = _number(text)
    if not x.is_integer():
        raise ValueError('Require integer count')
    return int(x)


def parse_prefix(*, count_prefix, perturbation, cell_info, gene_names,
                 limit=32, expected_genes=5000, expected_cells=20643):
    """Read the first `limit` cells; reject ambiguous order or malformed fields.

The author exports count/perturbation rows with positional integer indices and
cell_info with original barcodes in the same AnnData order. No barcode matching
is possible from the count CSV alone. The join is version-specific positional
alignment, not an independent verification against raw GEO/H5AD. Incomplete
last prefix line is excluded; all requested rows must have complete newlines.
Count prefixes are bounded to16MiB/512 selected cells; metadata stays4MiB/file.
"""
    if (type(limit) is not int or not 1 <= limit <= 512 or
        type(expected_genes) is not int or not 1 <= expected_genes <= 5000 or
        type(expected_cells) is not int or not limit <= expected_cells <= 20643):
        raise ValueError('Invalid bounded dimensions')
    if not isinstance(count_prefix, bytes) or b'\n' not in count_prefix:
        raise ValueError('Require newline-terminated complete prefix rows')
    tail = not count_prefix.endswith(b'\n')
    complete = count_prefix[:count_prefix.rfind(b'\n')+1]
    counts = _table(complete, max_bytes=16 * 2**20)
    perts, cells, genes = map(_table, (perturbation, cell_info, gene_names))
    if (not counts or counts[0] != ['']+[str(i) for i in range(expected_genes)] or
        not perts or perts[0] != ['', 'Dex','Nutlin','BMS','Vehicle','SAHA'] or
        not cells or cells[0] != ['', 'n_genes','n_counts'] or
        not genes or genes[0] != ['0']):
        raise ValueError('Unexpected CSV headers or gene order')
    if (len(perts) != expected_cells+1 or len(cells) != expected_cells+1 or
        len(genes) != expected_genes+1 or len(counts)-1 < limit or
        len(counts)-1 > expected_cells or
        any(len(g)!=1 or not g[0].strip() for g in genes[1:])):
        raise ValueError('Dimension mismatch or empty gene name')
    seen=set()
    for i,(pert,cell) in enumerate(zip(perts[1:],cells[1:])):
        if len(pert)!=6 or pert[0]!=str(i) or len(cell)!=3 or not cell[0] or cell[0] in seen:
            raise ValueError('Ambiguous positional index or cell identity')
        seen.add(cell[0])
        values=[_number(v) for v in pert[1:]]
        if sum(v>0 for v in values)!=1:
            raise ValueError('Require one source intervention including vehicle')
        _integer(cell[1]); _number(cell[2])
    result=[]
    for i,row in enumerate(counts[1:]):
        if len(row)!=expected_genes+1 or row[0]!=str(i):
            raise ValueError('Incomplete or reordered count row')
        vector=[_integer(x) for x in row[1:]]
        if i<limit:
            pert=perts[i+1]; cell=cells[i+1]
            active=next(j for j,x in enumerate(pert[1:],1) if _number(x)>0)
            result.append(dict(source_index=i,cell_id=cell[0],intervention=perts[0][active],
                dose_coordinate=_number(pert[active]),dose_units='source_scaled_coordinate',
                counts=vector,n_genes=_integer(cell[1]),n_counts=_number(cell[2])))
    return tuple(result),tuple(g[0] for g in genes[1:]),dict(
        complete_prefix_rows=len(counts)-1,selected_rows=len(result),
        genes=expected_genes,partial_tail_discarded=tail,
        physical_dose_resolved=False,
        pairing_rule='Pinned author export positional indices; cell_info barcode by same row order',
        readout_units='source count layer; selected 5000 genes; no normalization applied',
        training_ready=False)


def array_task(row, genes, *, direction):
    """Versioned inspection sidecar; deliberately does not grant eligibility.

Two complementary masks on the SAME observation. Reverse prediction is not a
causal inverse or a desired-response design task. Gene symbols may repeat:
identity is pinned source column position, never an inferred stable gene ID.
Cell IDs and post-assay QC are evaluation-only. This is not yet a GraphRecord
array extension or neural encoder. Physical dose remains unavailable.
"""
    if direction not in ('response', 'intervention'):
        raise ValueError('Unknown array task direction')
    genes = tuple(genes)
    if not genes or any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError('Require ordered nonempty gene symbols')
    counts = row['counts']
    if len(counts) != len(genes):
        raise ValueError('Array axis length mismatch')
    values = [_integer(x) for x in counts]
    name = row['intervention']
    dose = _number(row['dose_coordinate'])
    if (name not in ('Dex', 'Nutlin', 'BMS', 'SAHA', 'Vehicle') or
            not 0 < dose <= 1 or row['dose_units'] != 'source_scaled_coordinate' or
            (name == 'Vehicle' and dose != 1)):
        raise ValueError('Unsupported source intervention coordinate')
    intervention = dict(identity=name, source_coordinate=dose,
                        units='source_scaled_coordinate', is_vehicle=name == 'Vehicle',
                        physical_concentration=None)
    expression = dict(values=values, axis_ids=list(genes),
                      axis_identity='source_column_position',
                      units='source selected-gene count layer', status='observed')
    context = dict(study='GSM4150377', cell_line='A549', exposure_hours=24,
                   baseline=None, desired_response=None)
    conditions = dict(context=context)
    if direction == 'response':
        conditions['intervention'] = intervention
        targets = dict(expression=expression)
    else:
        conditions['expression'] = expression
        targets = dict(intervention=intervention)
    return dict(version=1, task=direction, conditions=conditions, targets=targets,
                evaluation=dict(cell_id=row['cell_id'], source_index=row['source_index'],
                                n_genes=row['n_genes'], n_counts=row['n_counts']),
                training_ready=False,
                quarantine_reason='Physical dose conversion unresolved; tensor integration pending')
