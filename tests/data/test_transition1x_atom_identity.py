"""Detect stale source receipts, axis/ordinal rebinding and target leakage."""
from pathlib import Path
import importlib
import shutil
import pytest

MANIFEST=Path('outputs/data-sources/20261031-legacy-mint-v1/curated/transition1x-original-native-roles-v1/manifest.json')
PAYLOAD=MANIFEST.parent/'native-reactions-000.parquet'
RECORD='Transition1x_original:v4:C2H2N2O:rxn9455'
GROUP='Transition1x:v4:C2H2N2O:rxn9455'
AXIS=(8,6,6,7,7,1,1)


def api():
    assert importlib.util.find_spec('mora3.data.transition1x_atom_identity'), 'source-bound atom reader missing'
    return importlib.import_module('mora3.data.transition1x_atom_identity')


def read(**changes):
    args=dict(manifest_path=MANIFEST,payload_index=0,ordinal=13,
              expected_record_id=RECORD,expected_reaction_group=GROUP,
              expected_native_split='val',expected_atomic_numbers=AXIS)
    args.update(changes)
    return api().read_retained_atom_identity(**args)


def test_real_complete_order_unknowns_and_no_targets():
    c=read()
    assert c.context.atomic_numbers==AXIS
    assert c.context.ordered_atom_ids==tuple(f'{RECORD}:atom:{i}' for i in range(7))
    assert c.context.source_native_split=='val' and c.context.source_reaction_group==GROUP
    assert c.context.source_payload_sha256=='2da99f8b29473c1e3676e8fd902e26eab606cb0f548fd2fbad199f0d0f53b3cf'
    assert c.context.electronic_charge is None and c.context.spin_multiplicity is None
    assert len(c.context.missing_context)==5
    assert not hasattr(c,'roles') and not hasattr(c,'positions')
    assert c.ordinal==13 and c.payload_index==0
    from mora3.data.native_atom_context import encode_native_atom_identity
    x=encode_native_atom_identity(c.context)
    assert x.identity.argmax(-1).tolist()==[7,5,5,6,6,0,0]
    assert x.context_missing.all()


@pytest.mark.parametrize('key,value',[
 ('ordinal',12),('ordinal',-1),('ordinal',1000),('ordinal',True),
 ('payload_index',True),('payload_index',-1),('payload_index',11),
 ('expected_record_id','other'),('expected_reaction_group','other'),
 ('expected_native_split','train'),('expected_atomic_numbers',(6,8,6,7,7,1,1)),
 ('expected_atomic_numbers',(8,6)),('expected_atomic_numbers',(8,6,6,7,7,True,1)),
])
def test_rebinding_and_incomplete_axis_refused(key,value):
    with pytest.raises(ValueError):read(**{key:value})


def test_fresh_manifest_hash_on_every_call(tmp_path):
    p=tmp_path/'manifest.json';shutil.copyfile(MANIFEST,p)
    read(manifest_path=p)
    p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='manifest'):read(manifest_path=p)


def test_fresh_payload_hash_even_same_length_and_timestamp(tmp_path):
    p=tmp_path/'payload.parquet';shutil.copyfile(PAYLOAD,p)
    before=read(payload_path=p)
    import os
    stat=p.stat();data=bytearray(p.read_bytes());data[len(data)//2]^=1;p.write_bytes(data)
    os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='payload'):read(payload_path=p)
    assert before.context.atomic_numbers==AXIS


def test_truncated_payload_refused(tmp_path):
    p=tmp_path/'payload.parquet';p.write_bytes(b'PAR1')
    with pytest.raises(ValueError,match='payload'):read(payload_path=p)


@pytest.mark.parametrize('method',['require_model_use','require_training_use','require_task_use'])
def test_export_authentication_never_grants_admission(method):
    with pytest.raises(ValueError,match='admission'):getattr(read(),method)()
