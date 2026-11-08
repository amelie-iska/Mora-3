"""Full actual dictionary + new sparse numeric equations; no source admission."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pytest
import pyarrow as pa
import torch
from torch.nn import functional as F
from mora3.data.sccello_numeric_dictionary import decode_numeric_mapping
from mora3.data.sccello_native_rows import TokenVocabulary,SourceBinding,SOURCE_CLS_ID
from mora3.data.sccello_selected_rows import SelectedNativeGeneBatch
from mora3.model.cell_native import SharedNativeCellModel,NativeCellProtocol,CellMaskedBatch,GENE_COUNT
from mora3.model.cell_mup import cell_mup_parameter_roles,cell_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from test_annotation_mup import make as annotation

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
PROTOCOL=ROOT/'outputs/data-sources/readiness-20261106-sccello-native/source-protocol-roles-and-exposure.json'
PROTOCOL_SHA='a0b700ae63f22f485576cc1d2911bc6796a6bf3e66de8aeb690205465db9a74c'


@pytest.fixture(scope='module')
def source_assets():
    m=json.loads((ROOT/'outputs/data-intake/readiness-20261106-corpus/numeric-dictionaries-v1/manifest.json').read_bytes())
    ref=next(x['source'] for x in m['sources'] if x['ordinal']==1411);raw=Path(ref['path']).read_bytes()
    assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
    vocabulary=TokenVocabulary.from_passive_mapping(decode_numeric_mapping(raw))
    assert len(vocabulary.gene_keys)==25426 and vocabulary.gene_keys[:2]==('<pad>','<mask>')
    protocol=NativeCellProtocol.from_source_document(PROTOCOL.read_bytes(),expected_sha256=PROTOCOL_SHA)
    return vocabulary,protocol


def make(source_assets,w=8):
    return SharedNativeCellModel(annotation(w),*source_assets,max_batch_size=32)


def batch(source_assets):
    _,p=source_assets;release,repo,rev=p.sources[0]
    binding=SourceBinding('/synthetic-unit-never-read','a'*64,'b'*64,0,release,repo,rev,
        'synthetic-reference-no-admission',None,'gene_token_ids',10,())
    ids=pa.array([[SOURCE_CLS_ID,2,3,4],[SOURCE_CLS_ID,2,3,4,5]],type=pa.list_(pa.int64()))
    values=pa.array([[0.,0.,1.234567890123,3.],[0.,1.,2.,3.,4.]],type=pa.list_(pa.float64()))
    source=SelectedNativeGeneBatch(binding,pa.array([1,4],type=pa.uint64()),ids,values,{})
    return CellMaskedBatch(source,((True,True,False,True),(True,True,False,False,True)),
                           ((False,False,True,False),(False,False,True,True,False)))


def test_sparse_full_axis_input_and_readout_equations_all_parameter_gradients(source_assets):
    m=make(source_assets,16).double();b=batch(source_assets);ids,values=m.validate_batch(b)
    graphs=[]
    for tokens,vals,known in zip(ids,values,b.input_available):
        v=m.value_input.bias+m.source_cls.weight[0]
        for token,value,available in zip(tokens[1:],vals[1:],known[1:]):
            if available:v=v+m.value_input.weight[:,token-2]*value+m.availability_input.weight[:,token-2]
        actual=m._project_row(tokens,vals,known);torch.testing.assert_close(actual,v,rtol=1e-12,atol=1e-12)
        graphs.append(dict(features=v[None],endpoints=torch.tensor([[0,0]]),roles=torch.tensor([0]),identifiers=m.shared.sequence_identifiers()))
    context=m.shared.backbone(graphs)[0][:,0]
    expected=F.linear(context*.75,m.value_prediction.weight,m.value_prediction.bias)
    actual=m(b);torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    params=tuple(m.parameters());a=torch.autograd.grad(actual.square().sum(),params,allow_unused=True)
    c=torch.autograd.grad(expected.square().sum(),params,allow_unused=True)
    for x,y in zip(a,c):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)


def test_target_masked_from_both_inputs_audit_excluded_observed_zero_distinct(source_assets):
    m=make(source_assets);b=batch(source_assets);before=m(b)
    raw=b.source.gene_expression_nums.to_pylist();raw[0][2]=99.;raw[1][2]=100.;raw[1][3]=200.
    changed=replace(b,source=replace(b.source,gene_expression_nums=pa.array(raw,type=pa.list_(pa.float64())),
        audit_metadata={'cell_type':pa.array(['target-label','other']),'cell_counts':pa.array([999.,1.])}))
    torch.testing.assert_close(m(changed),before,rtol=0,atol=0)
    assert not torch.equal(m.reference_objective(before,b),m.reference_objective(before,changed))
    inputs=list(b.input_available);inputs[0]=(True,False,False,True)
    unknown_zero=replace(b,input_available=tuple(inputs))
    assert not torch.equal(m(unknown_zero)[0],before[0]),'known literal zero is distinguished by input availability; absent is unknown'
    assert b.source.gene_expression_nums.to_pylist()[0][2]==1.234567890123


def test_native_float64_targets_per_cell_then_batch_weighting_and_no_absent_gene_loss(source_assets):
    m=make(source_assets);b=batch(source_assets)
    z=torch.zeros(2,GENE_COUNT,dtype=torch.float32,requires_grad=True)
    loss=m.reference_objective(z,b)
    expected=(torch.tensor(1.234567890123,dtype=torch.float64).square()+torch.tensor([2.,3.],dtype=torch.float64).square().mean())/2
    torch.testing.assert_close(loss,expected,rtol=0,atol=0);assert loss.dtype==torch.float64
    loss.backward();assert z.grad[0].count_nonzero()==1 and z.grad[1].count_nonzero()==2
    assert z.grad[:,100:].count_nonzero()==0
    none=replace(b,target_observed=tuple(tuple(False for _ in row) for row in b.target_observed))
    zero=m.reference_objective(z,none);assert torch.isfinite(zero) and zero==0 and zero.requires_grad


def test_complete_source_gene_axis_all_nonzero_values_retained_not_topk_and_gradients(source_assets):
    m=make(source_assets);b=batch(source_assets)
    # New explicit synthetic all-axis unit, not a fabricated native source row.
    tokens=[SOURCE_CLS_ID,*range(2,25426)];values=[0.,*([.5]*GENE_COUNT)]
    known=[True]*(GENE_COUNT+1);known[-1]=False
    p=m._project_row(tokens,values,known);p.sum().backward()
    assert m.value_input.weight.grad[:,:-1].count_nonzero()==8*(GENE_COUNT-1)
    assert m.availability_input.weight.grad[:,:-1].count_nonzero()==8*(GENE_COUNT-1)
    assert m.value_input.weight.grad[:,-1].count_nonzero()==0
    assert m.availability_input.weight.grad[:,-1].count_nonzero()==0
    assert m.source_cls.weight.grad.count_nonzero()==8


def test_all_masked_has_only_bias_cls_no_nan_and_native_cls_never_a_target(source_assets):
    m=make(source_assets);b=batch(source_assets)
    inputs=tuple((True,)+tuple(False for _ in row[1:]) for row in b.input_available)
    b=replace(b,input_available=inputs);ids,values=m.validate_batch(b)
    expected=m.value_input.bias+m.source_cls.weight[0]
    for row,value,mask in zip(ids,values,inputs):torch.testing.assert_close(m._project_row(row,value,mask),expected,rtol=0,atol=0)
    assert torch.isfinite(m(b)).all()
    changed=replace(b,target_observed=((True,False,True,False),b.target_observed[1]))
    with pytest.raises(ValueError,match='CLS'):m(changed)


@pytest.mark.parametrize('failure',['overlap','mask-shape','mask-type','unknown-protocol','partial-vocab','source-row-order'])
def test_refusals_before_model_rng_or_parameter_mutation(source_assets,failure):
    m=make(source_assets);b=batch(source_assets);before=[p.detach().clone() for p in m.parameters()];rng=torch.get_rng_state().clone()
    if failure=='overlap':b=replace(b,input_available=((True,True,True,True),b.input_available[1]))
    elif failure=='mask-shape':b=replace(b,input_available=((True,),b.input_available[1]))
    elif failure=='mask-type':b=replace(b,input_available=((True,1,False,True),b.input_available[1]))
    elif failure=='unknown-protocol':b=replace(b,source=replace(b.source,binding=replace(b.source.binding,release_index=7)))
    elif failure=='source-row-order':b=replace(b,source=replace(b.source,source_row_ordinals=pa.array([4,1],type=pa.uint64())))
    with pytest.raises(ValueError):
        if failure=='partial-vocab':SharedNativeCellModel(m.shared,replace(source_assets[0],gene_keys=source_assets[0].gene_keys[:-1]),source_assets[1])
        else:m(b)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,v) for p,v in zip(m.parameters(),before))


def test_complete_source_family_axes_storage_lr_and_strict_checkpoint(source_assets):
    ms=tuple(make(source_assets,w) for w in (8,12,16));family=cell_mup_width_family(*ms,identifier_axis=('protein-coordinate',));m=ms[-1]
    roles=cell_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))
    assert family.total_numel==tuple(sum(p.numel() for p in t.parameters()) for t in ms)
    extra=3*GENE_COUNT*16+2*16+GENE_COUNT
    assert sum(p.numel() for n,p in m.named_parameters() if not n.startswith('shared.'))==extra
    assert roles['value_input.weight'].role=='input' and roles['availability_input.weight'].role=='input'
    assert roles['value_prediction.weight'].role=='output' and roles['source_cls.weight'].role=='embedding'
    opt=MuAdamW(m,family,roles,base_lr=.01,weight_decay=0.,decay_aliases=());assert not opt.state
    by={n:g for g in opt.param_groups for n in g['mup_aliases']}
    for n in ('value_input.weight','availability_input.weight','value_prediction.weight','source_cls.weight'):assert by[n]['lr']==.01
    before=[p.detach().clone() for p in m.parameters()];state=copy.deepcopy(m.state_dict())
    state['value_input.weight']+=3;state['_extra_state']['protocol']['coordinate']='unqualified-GEO'
    with pytest.raises(ValueError):m.load_state_dict(state)
    assert all(torch.equal(p,v) for p,v in zip(m.parameters(),before))
    assert ms[0].model_binding()==ms[-1].model_binding()


@pytest.mark.skipif(os.environ.get('MORA3_NEW_CELL_NATIVE24_BOUNDARY')!='1',reason='Explicit new bounded native source/model one-shot; ordinary suite uses source-axis unit checks')
def test_new_native_24row_source_masked_model_boundary_and_independent_oracle(source_assets,tmp_path):
    out=ROOT/'outputs/implementation/readiness-20261106-native-cell-v1/actual-24row-model-boundary.json'
    assert not out.exists(),'one-shot receipt already exists; do not replay historical native NN boundary'
    from mora3.data.sccello_native_rows import bind_audited_source
    from mora3.data.sccello_selected_rows import iter_selected_native_gene_batches
    import pyarrow.parquet as pq
    proof_path=ROOT/'outputs/data-sources/readiness-20261106-sccello-selected-reader/actual-source-roundtrip.json'
    raw=proof_path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='a23e8e685922adf2bd4ed94560634ed68b8d0e5aab2f62fd4ac6a11f5a794312'
    proof=json.loads(raw);torch.manual_seed(2026110691);m=make(source_assets)
    reports=[];seen=0;positions=0
    for source_report in proof['reports']:
        ref=source_report['source_receipt'];binding=bind_audited_source(ref['path'],receipt_sha256=ref['sha256'])
        selected=source_report['selected_original_row_ordinals']
        batches=tuple(iter_selected_native_gene_batches(binding,selected,batch_size=8))
        assert all(b.binding is binding and not b.audit_metadata for b in batches)
        source=SelectedNativeGeneBatch(binding,pa.concat_arrays([b.source_row_ordinals for b in batches]),
            pa.concat_arrays([b.gene_token_ids for b in batches]),
            pa.concat_arrays([b.gene_expression_nums for b in batches]),{})
        assert source.num_rows==8 and not source.audit_metadata
        assert source.source_row_ordinals.to_pylist()==selected
        # Independent direct original-coordinate row-group oracle, no logical-
        # subset offsets or renamed synthetic source identities.
        oracle=[];start=0
        with pq.ParquetFile(binding.path,memory_map=False,pre_buffer=False,arrow_extensions_enabled=False) as pf:
            for group in range(pf.metadata.num_row_groups):
                stop=start+pf.metadata.row_group(group).num_rows
                ids=[i for i in selected if start<=i<stop]
                if ids:
                    table=pf.read_row_group(group,columns=[binding.gene_field,'gene_expression_nums'],use_threads=False)
                    oracle.extend(table.slice(i-start,1).to_pylist()[0] for i in ids)
                start=stop
        native_ids=source.gene_token_ids.to_pylist();native_values=source.gene_expression_nums.to_pylist()
        assert [r[binding.gene_field] for r in oracle]==native_ids
        assert [r['gene_expression_nums'] for r in oracle]==native_values
        inputs=[];targets=[]
        for row in native_ids:
            chosen={1,len(row)-1} if len(row)>1 else set()
            inputs.append(tuple(i not in chosen for i in range(len(row))))
            targets.append(tuple(i in chosen for i in range(len(row))))
        masked=CellMaskedBatch(source,tuple(inputs),tuple(targets))
        with torch.no_grad():prediction=m(masked);loss=m.reference_objective(prediction,masked)
        assert prediction.shape==(8,25424) and torch.isfinite(prediction).all() and torch.isfinite(loss)
        # Full original paired arrays are unchanged after the native model boundary.
        assert source.gene_token_ids.to_pylist()==native_ids and source.gene_expression_nums.to_pylist()==native_values
        row_positions=sum(map(len,native_ids));seen+=8;positions+=row_positions
        reports.append(dict(source_file_ordinal=binding.source_ordinal,source_release_index=binding.release_index,
            original_source_sha256=binding.source_sha256,source_receipt_sha256=ref['sha256'],
            original_row_ordinals=selected,source_paired_positions=row_positions,
            reader_rowgroup_batch_sizes=[b.num_rows for b in batches],explicit_same_binding_arrow_coalescing=True,
            masked_target_positions=sum(sum(x) for x in targets),prediction_shape=list(prediction.shape),
            native_coordinate_mean_per_cell_mse=float(loss),source_arrays_unchanged=True,
            direct_native_coordinate_oracle_equal=True,source_role='audit-only-not-a-training-grant'))
    assert seen==24 and positions==40516
    receipt=dict(schema='Mora3.NativeCellModelNew24RowBoundary/v1',source_proof_sha256=hashlib.sha256(raw).hexdigest(),
        model_binding=m.model_binding(),model_width=8,gene_axis=25424,actual_rows=seen,actual_paired_positions=positions,
        reports=reports,optimizer_updates=0,training=False,source_admission_issued=False,old_one_shot_diagnostics_replayed=False)
    out=ROOT/'outputs/implementation/readiness-20261106-native-cell-v1/actual-24row-model-boundary.json'
    # This uniquely named receipt belongs to this new check, never historical outputs.
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
