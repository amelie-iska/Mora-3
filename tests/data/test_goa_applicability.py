import hashlib
import json
import copy
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from test_goa_sequence import setup as sequence_setup
from mora3.data import goa_packets as g
from mora3.data.annotation_axes import AnnotationAxes


def setup(tmp_path):
    packet,kb=sequence_setup(tmp_path)
    packet=g.bind_canonical_sequence(packet,root=tmp_path,page_receipt=kb,results_index=0)
    p=packet.provenance();p.update(groups=['accession:P12345'],authoritative_groups={'master_sha256':'a'*64})
    packet=g.VerifiedGOAPacket(packet.condition_bytes,packet.target_bytes,json.dumps(p).encode())
    axis=['GO:0000001','GO:0000002']
    def save(name,rows):
        path=tmp_path/name;pq.write_table(pa.Table.from_pylist(rows),path);raw=path.read_bytes()
        return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    ar=save('axis.parquet',[dict(axis_id=x) for x in axis])
    br=save('bindings.parquet',[dict(axis_id=x,raw_annotation_ID_not_remapped=True,annotation_source_versions_separate=True) for x in axis]);br['rows']=2
    registry=AnnotationAxes(json.dumps(dict(rows={'GO':[dict(axis_id=x) for x in axis]},binding={'axis_files':[dict(ar,axis='GO')]})))
    m=dict(schema='mora3.GOCurrentReferenceAxisBinding/v1',source_axis=ar,binding=br,ontology_version='pinned',ontology_timestamp='time',same_snapshot=False,no_automatic_obsolete_or_alt_ID_remapping=True,no_GO_ancestry_label_propagation=True,no_dense_negative_labels=True)
    path=tmp_path/'ontology.json';path.write_text(json.dumps(m))
    return packet,registry,path,hashlib.sha256(path.read_bytes()).hexdigest()


def bind(args,tmp_path):
    p,r,path,sha=args
    return g.bind_goa_applicability(p,root=tmp_path,registry=r,ontology_manifest=path,ontology_sha256=sha)


def test_native_masks_keep_unknown_raw_targets_and_sequence_only_inputs(tmp_path):
    args=setup(tmp_path);v=bind(args,tmp_path)
    assert v.status=='applicable'
    assert v.labels==(1.,0.) and v.observed==(True,False)
    assert v.packet.inputs()=={'protein_sequence':'ACDE'}
    assert v.packet.target_bytes==args[0].target_bytes
    assert v.packet.provenance()['goa_objective']['policy']=='exact-subject-coarse-GOA-observed-BCE-v1'
    assert v.packet.provenance()['generation_eligible'] is False


def test_conflict_quarantines_whole_packet_without_losing_assertions(tmp_path):
    args=list(setup(tmp_path));p=args[0];t=p.targets();a=copy.deepcopy(t['source_GO_annotations'][0]);a['qualifier']='NOT|enables';t['source_GO_annotations'].append(a)
    args[0]=g.VerifiedGOAPacket(p.condition_bytes,json.dumps(t).encode(),p.provenance_bytes)
    v=bind(args,tmp_path)
    assert v.status=='quarantined_polarity_conflict'
    assert v.observed==(False,False) and v.labels==(0.,0.)
    assert len(v.packet.targets()['source_GO_annotations'])==2


@pytest.mark.parametrize('evidence,qualifier,extensions,expected',[
 ('ND','enables',None,(0.,False)),('IDA','NOT|enables',None,(0.,True)),
 ('IDA','enables',[{'x':'context'}],(0.,False)),('future','enables',None,(0.,False))])
def test_applicability_semantics(tmp_path,evidence,qualifier,extensions,expected):
    args=list(setup(tmp_path));p=args[0];t=p.targets();a=t['source_GO_annotations'][0];a.update(goEvidence=evidence,qualifier=qualifier,extensions=extensions)
    args[0]=g.VerifiedGOAPacket(p.condition_bytes,json.dumps(t).encode(),p.provenance_bytes)
    v=bind(args,tmp_path);assert (v.labels[0],v.observed[0])==expected


@pytest.mark.parametrize('bad',['sequence','groups','manifest_hash','binding_hash','axis','policy'])
def test_native_binding_rejects_unverified_or_changed_dependencies(tmp_path,bad):
    args=list(setup(tmp_path));p=args[0];pr=p.provenance()
    if bad=='sequence':pr['sequence_source_verified']=False
    if bad=='groups':pr.pop('authoritative_groups')
    args[0]=g.VerifiedGOAPacket(p.condition_bytes,p.target_bytes,json.dumps(pr).encode())
    if bad=='manifest_hash':args[3]='0'*64
    if bad=='binding_hash':(tmp_path/'bindings.parquet').write_bytes(b'changed')
    if bad in ('axis','policy'):
        m=json.loads(args[2].read_bytes())
        if bad=='axis':m['source_axis']['sha256']='0'*64
        else:m['no_dense_negative_labels']=False
        args[2].write_text(json.dumps(m));args[3]=hashlib.sha256(args[2].read_bytes()).hexdigest()
    with pytest.raises(ValueError):bind(args,tmp_path)
