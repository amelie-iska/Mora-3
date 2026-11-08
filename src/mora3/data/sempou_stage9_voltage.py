"""Native stage 9 plotted voltage, with independent original polarity verification."""
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from defusedxml.ElementTree import fromstring
from mora3.data.bioelectric_evidence import (
    BioelectricWorkbookNumericCell, NativeBioelectricBaseline,
    NativeBioelectricEndpoint, NativeBioelectricEvent, NativeBioelectricObservation,
)
from mora3.data.source_scalar_native_rows import ScalarTarget

ROOT = Path.cwd()
AXES = 'outputs/data-sources/readiness-20261107-sempou-native-voltage-source-axes-v2/manifest.json'
AXES_SHA = 'd0b501e9a242b736bd564363720b3203449406cc441ea7316b67b896204eba68'
SHEET = 'xl/worksheets/sheet24.xml'
WORK = 'PMC9637099.1'
X = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
GROUP_COUNTS = {'Control Normal K+':16, 'CholineCl Normal K+':16, 'CholineCl High K+':16, 'High K+':14}


def _read(root, path, digest, size):
    p = (root/path).resolve()
    p.relative_to(root)
    with p.open('rb') as f:
        raw = f.read(size+1)
    if len(raw)!=size or sha256(raw).hexdigest()!=digest:
        raise ValueError('immutable Sempou source artifact differs')
    return raw


def load_sempou_stage9_voltage(root):
    """Yield the original signed plot axis once per source coordinate."""
    root = Path(root).resolve()
    manifest = json.loads(_read(root,AXES,AXES_SHA,3299))
    rows_ref = manifest['exact_measurement_position_index']
    index = json.loads(_read(root,rows_ref['path'],rows_ref['sha256'],rows_ref['bytes']))
    workbook = manifest['original_workbook']
    raw = _read(root,workbook['path'],workbook['sha256'],workbook['bytes'])
    rows = [r for r in index if r.get('source_plot_signed_mV_cell')]
    if len(rows)!=62 or any(r['native_developmental_stage']!=9
            or r['independently_confirmed_source_plot_sign_negation'] is not True for r in rows):
        raise ValueError('complete original stage 9 plotted/instrument scope required')
    with ZipFile(BytesIO(raw)) as z:
        wanted = ('xl/workbook.xml','xl/_rels/workbook.xml.rels','xl/sharedStrings.xml',SHEET)
        if any(z.namelist().count(name)!=1 for name in wanted):
            raise ValueError('one literal original workbook member required')
        book, rels, texts, sheet = (z.read(name) for name in wanted)
    def xml(b):
        return fromstring(b,forbid_dtd=True,forbid_entities=True,forbid_external=True)
    name, = [x for x in xml(book).iter(X+'sheet') if x.get('name')=='Ephys hiK_choline_SUPFig3']
    ref, = [x.get('Target') for x in xml(rels) if x.get('Id')==name.get(R)]
    if ref not in (SHEET,'worksheets/sheet24.xml','/'+SHEET):
        raise ValueError('original stage 9 worksheet relationship differs')
    shared = [''.join(t.text or '' for t in v.iter(X+'t')) for v in xml(texts).iter(X+'si')]
    entries = list(xml(sheet).iter(X+'c'))
    cells = {c.get('r'):c for c in entries}
    if len(cells)!=len(entries):
        raise ValueError('duplicate stage 9 workbook coordinate')
    def numeric(ref):
        c = cells[ref]
        if c.get('t') not in (None,'n') or len(c)!=1 or c[0].tag!=X+'v' or len(c[0]):
            raise ValueError('original direct numeric cell required, excluding formula caches')
        return c[0].text
    counts = Counter()
    plots = set()
    instruments = set()
    result = []
    sheet_sha = sha256(sheet).hexdigest()
    for row in rows:
        p,i = row['source_plot_signed_mV_cell'],row['original_instrument_reading_cell']
        label = cells[row['native_condition_cell']]
        if label.get('t')!='s' or len(label)!=1 or label[0].tag!=X+'v':
            raise ValueError('exact original treatment header required')
        condition = shared[int(label[0].text)]
        lexeme, instrument = numeric(p),numeric(i)
        value = float(Decimal(lexeme))
        if (p in plots or i in instruments or condition!=row['native_condition']
                or lexeme!=row['source_plot_signed_mV_lexeme'] or instrument!=row['original_recorded_numeric_lexeme']
                or Decimal(lexeme)!=-Decimal(instrument) or not math.isfinite(value)):
            raise ValueError('original signed stage 9 point or one-to-one polarity evidence differs')
        plots.add(p)
        instruments.add(i)
        counts[condition]+=1
        proof=BioelectricWorkbookNumericCell(workbook['path'],workbook['bytes'],workbook['sha256'],
            SHEET,sheet_sha,p,cells[p].get('t'),lexeme)
        scalar=ScalarTarget('reported_point',value,None,None,None,None,None,lexeme,
            'exact-original-openxml-v-number-lexeme',b'{}',None,None,
            b'{"native_unit":"mV","coordinate":"published stage 9 signed plot value"}')
        obs=NativeBioelectricObservation(
            NativeBioelectricBaseline('Xenopus laevis','embryo animal pole','embryo','stage 9',None),
            (NativeBioelectricEvent('alter_ionic_environment','reported ionic solution condition',
                'ionic_environment',(),condition,None,None,None),),
            NativeBioelectricEndpoint('membrane potential','sharp intracellular electrode; stage 9 published signed plot',
                'mV','native_scalar',None),'primary_observation','reported_comparator',WORK,WORK+'#stage9-original-worksheet',
            WORK,(proof,),numeric_target=scalar)
        obs.verify_evidence(root)
        result.append(obs)
    if counts!=GROUP_COUNTS:
        raise ValueError('all four exact source treatment groups required')
    return tuple(result)
