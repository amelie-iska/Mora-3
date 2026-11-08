from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import canonical, digest
from mora3.data.uniprot_full_record import (
    FullRecordFieldAxis, FullRecordUniProtAxes, _project_full_record,
)
from mora3.model.uniprot_full_record_model import SharedUniProtFullRecordModel, uniprot_full_record_parameter_roles
from mora3.model.parameterization import parameter_ledger, width_family
from test_pubchem_admet_mup import pubchem


def axes():
    return FullRecordUniProtAxes('a'*64,'b'*64,3,20,(
        FullRecordFieldAxis('comments','list',128,2),
        FullRecordFieldAxis('organism','dict',128,3),
    ))


def native(include_comments=True):
    sequence='ACDXU'
    source=dict(source_release='2026_03',raw_file_sha256='c'*64,source_row=9,entry_version=2,sequence_version=1)
    packet={'organism':{'scientificName':'literal provider name','taxonId':9606}}
    if include_comments:
        packet['comments']=[{'id':'native-target-only','number':4.25},{'label':'α'},{'label':'α'}]
    properties=[
        dict(parameter_type='sequence_length',value=5,unit='residue',source_pointer='/sequence/length',
             source_value_json='5',measurement_context=None,evidence_role='provider_computed_from_canonical_sequence'),
        dict(parameter_type='sequence_mass',value=503,unit='Da',source_pointer='/sequence/molWeight',
             source_value_json='503',measurement_context=None,evidence_role='provider_computed_from_canonical_sequence'),
    ]
    return dict(schema='mora3.CorpusTask/v1',task='source_full_UniProt_annotation_prediction',
                conditions={'canonical_sequence':sequence},
                targets={'annotation_packet':packet,'provider_computed_sequence_properties':properties},
                audit=dict(identity={'primary_accession':'audit-only','sequence_sha256':digest(sequence.encode()),
                                     'sequence_length':len(sequence),'entry_version':2,'sequence_version':1},source_ref=source),missing=[])


def row(doc=None):
    if doc is None:doc=native()
    raw=canonical(doc)
    return _project_full_record(dict(
        source_family='UniProt-full-record-annotations',source_revision='UniProt2026_03;entryVersion2;sequenceVersion1',
        task_name='source_full_UniProt_annotation_prediction',native_schema='mora3.CorpusTask/v1',
        split='train',source_sha256='c'*64,native_payload=raw,native_payload_sha256=digest(raw),
        record_id='audit-only',admitted_for_task=True,model_adapter_supported=False,
        admission_receipt_json=canonical(dict(license='CC-BY-4.0',reviewed=True,canonical_sequence_exact_baseline=True,
                                              source_ref=doc['audit']['source_ref'])).decode(),
    ),'synthetic.parquet',7,axes())


def test_exact_reversible_field_wire_and_separate_computed_properties():
    source=row()
    field=axes().field('comments')
    wire=source.target(field.name)
    value=field.decode(wire)
    assert value==native()['targets']['annotation_packet']['comments']
    assert len(value)==3 and type(value[0]['number']) is float and value[1]==value[2]
    assert field.encode(value)==wire
    assert source.condition.canonical_sequence=='ACDXU'
    assert not hasattr(source.condition,'source_coordinate')
    assert [(x.parameter_type,x.value,x.unit) for x in source.computed_properties]==[
        ('sequence_length',5,'residue'),('sequence_mass',503,'Da')]
    assert row(native(False)).target('comments') is None
    assert row(native(False)).condition==source.condition
    with pytest.raises(ValueError):field.decode(wire+b' ')
    with pytest.raises(ValueError):field.encode({'wrong':'root type'})


@pytest.mark.parametrize('failure', ('target-field-in-condition','target-metadata-unknown','numeric-value-drift'))
def test_refuses_field_projection_source_drift(failure):
    d=native()
    if failure=='target-field-in-condition':d['conditions']['organism']=9606
    elif failure=='target-metadata-unknown':d['targets']['annotation_packet']['unreviewed_source_field']=['x']
    else:d['targets']['provider_computed_sequence_properties'][0]['value']=4
    with pytest.raises(ValueError):row(d)


def test_source_field_loss_matches_full_reference_and_absent_rows_never_enter_neural():
    m=SharedUniProtFullRecordModel(pubchem(16),axes()).double().eval()
    active=row();missing=row(native(False))
    seen=[]
    shared=m.decoder.shared
    hook=shared.payload.register_forward_pre_hook(
        lambda _m,args,kwargs:seen.append((args[0],kwargs['input_fields'])),with_kwargs=True,
    )
    result=m._reference_field_objective((missing,active),'comments',chunk_size=8,recompute=True)
    hook.remove()
    assert result['observed']==1 and result['missing']==1 and len(seen)==1
    graph,fields=seen[0]
    assert {(x.name,x.value) for x in graph.nodes[0].measurements}=={
        ('canonical_amino_acid_sequence','ACDXU'),('requested_source_field','comments')}
    assert len(fields)==2 and b'audit-only' not in graph.canonical_bytes() and b'native-target-only' not in graph.canonical_bytes()
    context=m._context((active.condition,),'comments')
    ids=torch.tensor(m.decoder.codec.encode(active.target('comments')))
    h,_=m.decoder.rnn(m.decoder.embedding(ids[:-1][None]),context[None].contiguous())
    expected=F.cross_entropy(m.decoder._readout(h)[0],ids[1:])
    torch.testing.assert_close(result['loss'],expected,rtol=1e-13,atol=1e-13)
    result['loss'].backward()
    assert torch.count_nonzero(m.decoder.rnn.weight_hh_l0.grad)
    def denied(*_args,**_kwargs):pytest.fail('unavailable or invalid targets must refuse before model')
    shared.payload.forward=denied
    skipped=m._reference_field_objective((missing,),'comments',chunk_size=8,recompute=True)
    assert skipped['loss'] is None and skipped['observed']==0 and skipped['missing']==1
    assert skipped['missing_reason']=='source_field_unavailable'
    invalid=replace(active,field_targets=(('comments',b'{}'),active.field_targets[1]))
    with pytest.raises(ValueError):m._reference_field_objective((invalid,),'comments',chunk_size=8,recompute=True)


def test_alias_only_width_family_keeps_all_registered_decoder_parameter_names():
    owners=tuple(SharedUniProtFullRecordModel(pubchem(w),axes()) for w in (8,12,16))
    roles=tuple(uniprot_full_record_parameter_roles(x,identifier_axis=('protein-coordinate',)) for x in owners)
    family=width_family(*(parameter_ledger(m,r) for m,r in zip(owners,roles,strict=True)))
    assert family.total_numel==tuple(sum(p.numel() for p in m.decoder.parameters()) for m in owners)
    assert all(name.startswith('decoder.') for name in roles[-1])
