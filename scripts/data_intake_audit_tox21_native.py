from pathlib import Path
import json,hashlib,zipfile,io,collections,urllib.request,time,datetime,re,copy
from rdkit import Chem,RDLogger
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-tox21-original-and-evaluation-v1';out=B/'source-native-audit-v2';out.mkdir();start=time.perf_counter();RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
acq=json.loads((B/'acquisition-receipt.json').read_text())
for r in acq['assets']:
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
extra=json.loads((B/'acquisition-receipt-0002.json').read_text())
for r in extra['assets']:
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
allstats=[]
for name,split in [('training-original-sdf.zip','source_training'),('leaderboard-testing-labeled-sdf.zip','source_leaderboard_testing'),('final-evaluation-sdf.zip','source_final_evaluation_structures')]:
 z=zipfile.ZipFile(B/'raw'/name);member=next(n for n in z.namelist() if n.endswith('.sdf'));raw=z.read(member);blocks=raw.decode().split('$$$$');blocks=[b for b in blocks if b.strip()];mols=list(Chem.ForwardSDMolSupplier(io.BytesIO(raw),sanitize=False,removeHs=False));assert len(blocks)==len(mols);props=collections.Counter();calls=collections.defaultdict(collections.Counter);valid=0;bad=[];smiles=set();names=set();outpath=out/(split+'.jsonl')
 with outpath.open('x') as f:
  for i,(block,mol) in enumerate(zip(blocks,mols)):
   md={} if mol is None else {k:mol.GetProp(k) for k in mol.GetPropNames()};identifier=None if mol is None else mol.GetProp('_Name') if mol.HasProp('_Name') else None;names.add(identifier);props.update(md.keys());error=None;iso=None
   for key,value in md.items():
    if key.startswith(('NR-','SR-')):calls[key][value]+=1
   try:
    if mol is None:raise ValueError('source molblock parse failed')
    work=Chem.Mol(mol);Chem.SanitizeMol(work);iso=Chem.MolToSmiles(work,canonical=True,isomericSmiles=True);assert iso;valid+=1;smiles.add(iso)
   except Exception as exc:error=repr(exc);bad.append(dict(source_row=i,source_name=identifier,error=error))
   row=dict(source_split=split,source_row=i,source_archive=name,source_member=member,source_name=identifier,source_properties=md,original_sdf_record=block+'$$$$',derived_canonical_isomeric_smiles=iso,structure_error=error,coordinate_role='source_SDF_depiction_not_experimental_geometry')
   f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
 stat=dict(source_split=split,source_records=len(blocks),distinct_source_names=len(names),sanitized_source_graphs=valid,invalid_source_graphs=bad,distinct_canonical_isomeric_structures=len(smiles),property_key_counts=dict(props),assay_label_values={k:dict(v) for k,v in calls.items()},native_jsonl=dict(path=str(outpath.relative_to(R)),bytes=outpath.stat().st_size,sha256=hashlib.sha256(outpath.read_bytes()).hexdigest()))
 allstats.append(stat);print('NATIVE',json.dumps({k:stat[k] for k in ['source_split','source_records','distinct_source_names','sanitized_source_graphs','distinct_canonical_isomeric_structures']}),'invalid',len(bad),'labelkeys',list(calls),flush=True)
with (out/'receipt.json').open('x') as f:json.dump(dict(schema='mora3.original_Tox21_native_source_integrity_audit.v1',source_stats=allstats,all_raw_assets_complete_hash_verified=True,all_source_records_preserved=True,structure_method='sanitize explicit molecule copy;originalMolBlock retained;no stereochemistry repair/no invented geometry',split_role_status='source_protocol_roles preserved;newglobalchemicalsplitreviewpending',rights_status='sourceproviderevidenceacquired;separatereviewbeforeadmission',admitted_tasks=0,seconds=time.perf_counter()-start),f,indent=2)
print('COMPLETE_NATIVE',round(time.perf_counter()-start,3),flush=True)
