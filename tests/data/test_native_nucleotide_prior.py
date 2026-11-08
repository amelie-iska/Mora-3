"""Native nucleotide prior gates; synthetic fixtures are not source coverage."""
import hashlib
import json
from dataclasses import replace
import pytest
import torch
from mora3.data import nucleotide_prior as np


def row(sequence='aCTuRnx'):
    ref=dict(native_payload='source.parquet',native_payload_sha256='d'*64,source_row_index=0)
    payload=dict(schema='mora3.CorpusTask/v1',task=np.TASK,
        conditions=dict(prior_domain='source_DNA_sequence',source_organism='human'),
        targets=dict(source_native_sequence=sequence),audit=dict(source_ref=ref,source_proteins_admitted=False),
        missing=['parent_homology_taxon_heldout_proof'])
    raw=json.dumps(payload).encode();sha=hashlib.sha256(sequence.encode()).hexdigest()
    tid=hashlib.sha256(np.canonical(['source_DNA_sequence','human',sha])).hexdigest()
    return dict(task_id=tid,record_id='audit-only',source_family='DNA-native-reference-prior',
        source_revision='4e6860536e67a2799c206bb66b0ccce4931e78f1',source_sha256='d'*64,
        task_name=np.TASK,native_schema='mora3.CorpusTask/v1',native_payload=raw,
        native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy=np.POLICY,
        admitted_for_task=True,evidence_class=np.EVIDENCE,full_axis_values=len(sequence),
        admission_receipt_json=json.dumps(dict(source_ref=ref,native_sequence_sha256=sha,
            full_axis=True,native_case_TU_preserved=True,source_license='CC-BY-4.0',new_uniprot_sequence_admission=False)),
        missing_context_json=json.dumps(payload['missing']),condition_context_json=json.dumps(dict(source_organism='human',
            target_length_audit_only=True,raw_image=False,mask=False,cell_perturbation_response=False)))


def view(r=None):
    r=row() if r is None else r
    seq=json.loads(r['native_payload'])['targets']['source_native_sequence']
    return np.nucleotide_prior_view(r,effective_groups=['nucleotide_native:'+hashlib.sha256(seq.encode()).hexdigest(),'corrected-parent'],
        corpus_binding=dict(manifest_sha256='a'*64,shard_sha256='b'*64,groups_sha256='c'*64))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_native_symbols_masks_and_variable_lengths(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no cuda')
    sequence=np.ALPHABET*550
    v=view(row(sequence)); ids,observed,known=v.target_tensors(device=device)
    assert v.decode(ids)==sequence and len(ids)>16384
    assert observed.all() and known.tolist()==[s.upper() in 'ACGTU' for s in sequence]
    assert ids.device.type==device
    assert v.inputs()==dict(prior_domain='source_DNA_sequence',source_organism='human')
    assert v.canonical_bytes()!=view().canonical_bytes()


@pytest.mark.parametrize('changes',[dict(admitted_for_task=1),dict(split='test'),dict(source_family='other'),
    dict(source_revision='unknown'),dict(evidence_class='model_produced'),dict(full_axis_values=9),
    dict(native_payload_sha256='0'*64),dict(split_policy='random'),dict(admission_receipt_json='{}'),
    dict(condition_context_json='{}')])
def test_admission_rejects_changes(changes):
    r=row();r.update(changes)
    with pytest.raises(ValueError):view(r)


def test_targets_audit_lengths_excluded_from_conditions_and_immutable():
    a=view();b=view(row('U'*9000))
    assert a.input_bytes()==b.input_bytes()
    r=row();d=json.loads(r['native_payload']);d['conditions']['target_length']=7
    r['native_payload']=json.dumps(d).encode();r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    with pytest.raises(ValueError):view(r)
    meta=a.provenance();meta['effective_groups'].clear()
    assert a.provenance()['effective_groups']
    with pytest.raises(ValueError):replace(a,sequence='AAAA')
    with pytest.raises(ValueError):view(row('AC-G'))


def make_corpus(tmp_path,group_task=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    r=row()
    pq.write_table(pa.Table.from_pylist([dict(sequence='aCTuRnx',organism='human')]),tmp_path/'source.parquet')
    sha=hashlib.sha256((tmp_path/'source.parquet').read_bytes()).hexdigest()
    d=json.loads(r['native_payload']);d['audit']['source_ref']['native_payload_sha256']=sha
    receipt=json.loads(r['admission_receipt_json']);receipt['source_ref']['native_payload_sha256']=sha
    r['admission_receipt_json']=json.dumps(receipt);r['source_sha256']=sha
    r['native_payload']=json.dumps(d).encode();r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    pq.write_table(pa.Table.from_pylist([r]),tmp_path/'rows.parquet')
    seq=json.loads(r['native_payload'])['targets']['source_native_sequence']
    pq.write_table(pa.Table.from_pylist([dict(task_id=group_task or r['task_id'],effective_groups_json=json.dumps([
        'nucleotide_native:'+hashlib.sha256(seq.encode()).hexdigest(),'corrected-parent']))]),tmp_path/'groups.parquet')
    def rec(name):
        raw=(tmp_path/name).read_bytes();return dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),rows=1)
    m=dict(shards=[dict(rec('rows.parquet'),source_family='DNA-native-reference-prior',split='train')],
        effective_group_lookup={'rows.parquet':rec('groups.parquet')})
    raw=json.dumps(m).encode();(tmp_path/'manifest.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_reader_pins_groups_and_integrity(tmp_path):
    sha=make_corpus(tmp_path)
    kw=dict(root=tmp_path,expected_manifest_sha256=sha,source_family='DNA-native-reference-prior',limit=1)
    v,=np.read_nucleotide_priors('manifest.json',**kw)
    assert v.provenance()['effective_groups'][-1]=='corrected-parent'
    (tmp_path/'groups.parquet').write_bytes(b'bad')
    with pytest.raises(ValueError):np.read_nucleotide_priors('manifest.json',**kw)


def test_reader_rejects_group_misalignment(tmp_path):
    sha=make_corpus(tmp_path,group_task='wrong')
    with pytest.raises(ValueError,match='alignment'):
        np.read_nucleotide_priors('manifest.json',root=tmp_path,expected_manifest_sha256=sha,
            source_family='DNA-native-reference-prior',limit=1)


def test_reader_rejects_modified_original_source(tmp_path):
    sha=make_corpus(tmp_path)
    (tmp_path/'source.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source integrity'):
        np.read_nucleotide_priors('manifest.json',root=tmp_path,expected_manifest_sha256=sha,
            source_family='DNA-native-reference-prior',limit=1)


def test_rfam_unknown_organism_is_preserved():
    r=row('aTUNx');d=json.loads(r['native_payload'])
    d['conditions']=dict(prior_domain='RNA_family_reference_native_symbols',source_organism=None)
    d['missing'].append('source_organism')
    r.update(source_family='Rfam15.1-native-reference-prior',source_revision='Rfam15.1;January2026;originalCC0FASTA',
        native_payload=json.dumps(d).encode(),missing_context_json=json.dumps(d['missing']))
    r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    receipt=json.loads(r['admission_receipt_json']);receipt['source_license']='CC0-1.0'
    r['admission_receipt_json']=json.dumps(receipt)
    r['task_id']=hashlib.sha256(np.canonical(['RNA_family_reference_native_symbols',None,receipt['native_sequence_sha256']])).hexdigest()
    c=json.loads(r['condition_context_json']);c['source_organism']=None;r['condition_context_json']=json.dumps(c)
    v=view(r)
    assert v.inputs()['source_organism'] is None and v.decode(v.target_tensors()[0])=='aTUNx'
    d['missing'].remove('source_organism');r['native_payload']=json.dumps(d).encode()
    r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    r['missing_context_json']=json.dumps(d['missing'])
    with pytest.raises(ValueError,match='missingness'):view(r)
