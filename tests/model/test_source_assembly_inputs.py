"""Concrete source assets and refusal boundaries; no model construction."""
from pathlib import Path
import hashlib
import json
import pytest
from mora3.model.source_assembly_inputs import _pin,load_actual_source_assets,assemble_source_inputs,PINS

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_immutable_pin_hash_refuses_actual_changed_bytes(tmp_path):
    p=tmp_path/'source.json';p.write_bytes(b'{"native":1}')
    expected=hashlib.sha256(p.read_bytes()).hexdigest();assert _pin(tmp_path,('source.json',expected))==b'{"native":1}'
    p.write_bytes(b'{"native":2}')
    with pytest.raises(ValueError,match='SHA'):_pin(tmp_path,('source.json',expected))


@pytest.fixture(scope='module')
def actual_assets():return load_actual_source_assets(ROOT)


def test_actual_complete_finite_axes_payloadfit_and_protocol_boundaries(actual_assets):
    a=actual_assets;c=a.facts['axis_counts']
    assert (c['response'],c['GO'],c['EC'],c['residue'])==(5000,28167,5124,36)
    assert (c['physical_parameters'],c['physical_methods'],c['Rfam'],c['cell_genes'],c['mcsa_roles'])==(8,8,4091,25424,53)
    assert c['Rosetta']=={'CellPainting':1783,'L1000':977}
    assert c['LINCS']=={'CellPainting':1783,'L1000':978}
    assert a.vocabulary.specification_json.encode()==(ROOT/PINS['payload_vocabulary'][0]).read_bytes()
    assert a.facts['neural_forward_calls']==a.facts['optimizer_updates']==0
    assert a.facts['source_axes_are_schema_not_training_admission']is True
    assert a.facts['residue_list_max_target_length']==47
    assert len(set(a.response_axis))==4997
    assert a.response_axis.count('Y_RNA')==3 and a.response_axis.count('STPG4')==2
    assert a.facts['response_axis_identity']=='source_column_position'
    from mora3.data.goa_streamed import StreamedGOASourcePool
    assert type(a.goa_pool)is StreamedGOASourcePool
    assert len(a.goa_pool.source_paths)==1688
    assert len(a.native_view._ordinal_exclusions)==114


def test_production_inputs_never_synthesize_runtime_or_override_source_axes(actual_assets):
    with pytest.raises(ValueError,match='unresolved'):
        assemble_source_inputs(actual_assets,codec=None,max_decoder_tokens=1,goa_packets=(),runtime={})
    with pytest.raises(ValueError,match='overriding'):
        assemble_source_inputs(actual_assets,codec=None,max_decoder_tokens=1,goa_packets=(),runtime={'response_axis':('fake',)})


def test_complete_source_codec_immutable_order_controls_and_no_heldout_fit(actual_assets):
    a=actual_assets
    assert a.codec.symbols[:3]==('<PAD>','<BOS>','<EOS>')
    assert a.codec.size==57 and a.max_decoder_tokens==158
    assert a.facts['source_inverse_training_rows']==18309
    assert a.facts['source_inverse_condition_groups']==18293
    assert a.facts['source_inverse_unknown_symbols']=={'test':[],'validation':[]}
    from mora3.model.profile_decoder import SelfiesCodec
    for symbols in (('<PAD>','<BOS>','<EOS>','[C]','[C]'),('<PAD>','<BOS>','<EOS>','[C][O]'),('<BOS>','<PAD>','<EOS>','[C]')):
        with pytest.raises(ValueError):SelfiesCodec.from_ordered_source_symbols(symbols)
    ids=a.codec.encode('CCO',max_tokens=a.max_decoder_tokens)
    assert a.codec.decode(ids)=='CCO'


def test_source_image_and_protein_capacity_refusal_precedes_any_owner_construction(actual_assets):
    a=actual_assets
    assert a.facts['grayscale_required_native_geometry_rank']==336
    assert a.facts['grayscale_max_attention_entities']==672
    assert a.facts['known_full_protein_entity_token_bound']==35320
    assert a.facts['rgb_source_view_bound']is False
    runtime=dict(reconstruction_dimension=32,reconstruction_horizon=8,reconstruction_hidden=64,
        reconstruction_source={'test':'not-a-production-owner'},geometry_id_width=335,
        color_id_width=None,instance_slots=None,instance_max_output_bytes=None,payload_max_tokens=35320,
        color_input_policy=None,instance_slot_policy=None)
    with pytest.raises(ValueError,match='no crop'):assemble_source_inputs(a,runtime=runtime)
    runtime.update(geometry_id_width=336,payload_max_tokens=8192)
    with pytest.raises(ValueError,match='no crop'):assemble_source_inputs(a,runtime=runtime)


def test_current_effective_source_policy_binds_all_additional_actual_axes():
    assets = load_actual_source_assets(ROOT, source_view_policy='effective-study-v2')
    assert len(assets.native_view._ordinal_exclusions) == len(assets.goa_pool.native_view._ordinal_exclusions) == 227
    assert assets.facts['retained_taskviews'] == 80_656_029
    assert assets.facts['schema'] == 'Mora3.ActualNativeSourceAssemblyAssets/v2'
    assert assets.facts['neural_forward_calls'] == assets.facts['optimizer_updates'] == 0
    assert assets.facts['source_axes_are_schema_not_training_admission'] is True
    axes = assets.current_route_axes
    assert axes.pdb_coordinate_axes.max_atoms == 1821
    assert len(axes.pdb_coordinate_axes.source_coordinates) == 2
    assert len(axes.scalar_source_axis.partitions) == 16
    assert sum(rows for _, rows in axes.scalar_source_axis.partitions) == 5444
    assert sorted(len(axis.axis_ids) for axis in axes.rosetta_flow_axes) == [978, 978, 1783, 1783]
    assert len({axis.sha256 for axis in axes.rosetta_flow_axes}) == 4
    assert (axes.grayscale_source_envelope.max_height, axes.grayscale_source_envelope.max_width) == (512, 512)
    assert axes.grayscale_source_envelope.max_instances is None
    assert assets.facts['grayscale_complete_source_image_count'] == 12
    assert assets.facts['grayscale_required_native_geometry_rank'] == 336
