"""Target-free adapter for the exact Sempou native stage-10 voltage points.

This is data-only source projection. It does not infer biological identities,
perform split assignment, or authorize training.
"""
from __future__ import annotations
import hashlib, json, math, struct, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from mora3.data.bioelectric_evidence import (
    BioelectricWorkbookNumericCell, NativeBioelectricBaseline,
    NativeBioelectricEndpoint, NativeBioelectricEvent,
    NativeBioelectricObservation,
)
from mora3.data.source_scalar_native_rows import ScalarTarget

AXES = Path('outputs/data-sources/readiness-20261107-sempou-native-voltage-source-axes-v2')
ORIGINAL = Path('outputs/data-sources/readiness-20261107-sempou-exact-native-voltage-original-source-v2')
MANIFEST_SHA = 'd0b501e9a242b736bd564363720b3203449406cc441ea7316b67b896204eba68'
SHEET_NAME = 'Ephys MO_Fig3'
NS = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
def sha(b): return hashlib.sha256(b).hexdigest()
def exact(root, rel, n, digest):
    p=(root/rel).resolve(); p.relative_to(root.resolve()); b=p.read_bytes()
    if len(b)!=n or sha(b)!=digest: raise ValueError('immutable source byte identity mismatch: '+str(rel))
    return b
def project(root):
    root=Path(root).resolve()
    mp=AXES/'manifest.json'; mraw=exact(root,mp,3299,MANIFEST_SHA); m=json.loads(mraw)
    if m.get('schema')!='Mora3.SempouNativeVoltageSourceAxes/v1' or m.get('training_admission') is not False:
        raise ValueError('expected non-admitted source axes receipt')
    source=m['acquired_original_source_receipt']; source_m=json.loads(exact(root,Path(source['path']),source['bytes'],source['sha256']))
    wb=source_m['workbook']; wbraw=exact(root,Path(wb['path']),wb['bytes'],wb['sha256'])
    ix=m['exact_measurement_position_index']; index=json.loads(exact(root,Path(ix['path']),ix['bytes'],ix['sha256']))
    rows=[r for r in index if r.get('native_developmental_stage')==10]
    if len(rows)!=20 or m['stage10_exact_signed_measurement_positions']!=20: raise ValueError('complete exact stage10 direct point set required')
    with zipfile.ZipFile(__import__('io').BytesIO(wbraw)) as z:
        names=set(z.namelist()); shared=[]
        if 'xl/sharedStrings.xml' in names:
            ss=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(t.text or '' for t in si.iter('{%s}t'%NS['m'])) for si in ss.findall('m:si',NS)]
        book=ET.fromstring(z.read('xl/workbook.xml')); rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        relmap={x.attrib['Id']:x.attrib['Target'] for x in rels}
        sheet_target=None
        for sh in book.findall('m:sheets/m:sheet',NS):
            if sh.attrib.get('name')==SHEET_NAME:
                sheet_target=relmap[sh.attrib['{%s}id'%NS['r']]]; break
        if sheet_target is None: raise ValueError('native stage10 worksheet absent')
        member='xl/'+sheet_target.lstrip('/') if not sheet_target.startswith('xl/') else sheet_target
        sheet=z.read(member); shroot=ET.fromstring(sheet)
        cells={c.attrib['r']:c for c in shroot.findall('m:sheetData/m:row/m:c',NS)}
        def val(coord):
            c=cells.get(coord)
            if c is None: raise ValueError('required original source cell missing: '+coord)
            v=c.find('m:v',NS)
            if v is None or v.text is None: raise ValueError('literal cell value missing: '+coord)
            return shared[int(v.text)] if c.attrib.get('t')=='s' else v.text
        # Exact headers are assay-condition evidence, not biological identity.
        if val('C1')!='Control MO' or val('E1')!='kcnh6 MO': raise ValueError('native condition headers changed')
        sheet_sha=sha(sheet); observations=[]; manifest_rows=[]
        for r in rows:
            coord=r['measurement_cell']; col='C' if coord.startswith('C') else 'E' if coord.startswith('E') else None
            if col is None or r.get('source_workbook_sheet')!=SHEET_NAME: raise ValueError('unexpected stage10 measurement position')
            lex=r['signed_membrane_potential_mV_lexeme']
            if val(coord)!=lex or cells[coord].attrib.get('t') not in (None,'n'):
                raise ValueError('original signed mV point differs')
            cond=val(col+'1')
            if cond!=r['native_condition']: raise ValueError('index condition differs from workbook header')
            cell=BioelectricWorkbookNumericCell(str(wb['path']),wb['bytes'],wb['sha256'],member,sheet_sha,coord,
                cells[coord].attrib.get('t'),lex)
            f=float(lex)
            if not math.isfinite(f): raise ValueError('nonfinite native voltage')
            target=ScalarTarget('reported_point',f,None,None,None,None,None,lex,
                'exact-original-openxml-v-number-lexeme',b'{}',None,None,
                b'{"native_unit":"mV","coordinate":"direct signed membrane potential"}')
            event=NativeBioelectricEvent('other_reported_perturbation',
                'source-reported morpholino condition','unknown',(),cond,None,None,None)
            obs=NativeBioelectricObservation(
                NativeBioelectricBaseline('Xenopus laevis','embryo animal pole','embryo',
                    'stage 10',None,'native source-reported developmental stage'),(event,),
                NativeBioelectricEndpoint('membrane potential','sharp intracellular electrode; Axon 200B current clamp',
                    'mV','native_scalar',None),'primary_observation','reported_comparator',
                'PMC9637099.1','PMC9637099.1#'+SHEET_NAME,'PMC9637099.1',(cell,),numeric_target=target)
            obs.verify_evidence(root)
            observations.append(obs)
            # Keep provenance coordinate and target only; source record labels/headings are excluded.
            manifest_rows.append({'cell':coord,'condition':cond,'stage':10,'reported_mV':f,
                'lexeme':lex,'float64_bits':struct.pack('>d',f).hex(),'condition_sha256':obs.condition_sha256()})
    return observations, {'schema':'Mora3.SempouNativeVoltageAdapterReceipt/v1','source_axes_manifest_sha256':MANIFEST_SHA,
        'original_workbook_sha256':wb['sha256'],'original_workbook_path':wb['path'],'native_sheet_member':member,
        'native_sheet_sha256':sheet_sha,'included_stage10_direct_signed_observations':len(observations),
        'stage9_plot_signed_positions_excluded':True,'stage8_orientation_unresolved_positions_excluded':True,
        'outcomes_or_record_labels_used_as_condition_inputs':False,'biological_identity_inference':False,
        'split_assignment':False,'training_authorized':False,'rows':manifest_rows}
