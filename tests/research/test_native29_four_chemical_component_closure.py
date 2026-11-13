"""Synthetic fixtures only; native source readback has separate runtime evidence."""
import pytest

from mora3.research.cross_source_chemical_components import derive_literal_components


def item(i, mol, scaffold):
    return {"task_id": f"{i:064x}", "molecule":f"molecule:{mol:064x}",
            "scaffold":f"scaffold:{scaffold:064x}"}


def fixtures():
    return {
        "AZ":{"train":(item(1,11,30),item(2,12,30))},
        "FreeSolv":{"train":(item(3,11,31),),"validation":(item(4,99,99),),"test":(item(5,90,90),)},
        "Tox21":{"train":(item(6,50,30),),"validation":(item(7,99,99),),"test":(item(8,80,90),)},
        "ExpansionRx":{"train":(item(9,61,62),),"test":(item(10,70,75),)}
    }


def test_native_hash_components_preserve_role_axis_and_transitivity():
    result=derive_literal_components(fixtures())
    molecule=result['axes']['molecule'];scaffold=result['axes']['scaffold'];either=result['axes']['combined']
    assert molecule['total_components']==8
    assert molecule['largest_component_rows']==2
    assert scaffold['total_components']==6
    assert scaffold['largest_component_rows']==3
    assert either['largest_component_rows']==4
    assert either['heldout_train_mixed_components']==0
    assert either['cross_source_heldout_components']==2
    assert either['largest_training_component_rows']==4
    assert either['source_group_hash_domains_independently_equated'] is False
    assert either['global_split_admitted'] is False
    assert result['total_original_roles']==9
    assert result['total_original_rows']==10


def test_same_literal_hash_different_axes_never_joins():
    p=fixtures();p['ExpansionRx']['test']=(item(10,61,70),)
    result=derive_literal_components(p)
    assert result['axes']['molecule']['heldout_train_mixed_components']==1
    assert result['axes']['scaffold']['heldout_train_mixed_components']==0


def test_invalid_population_rejected_before_component_union():
    p=fixtures();p['FreeSolv']['validation']=(item(1,99,99),)
    with pytest.raises(ValueError,match='task'):derive_literal_components(p)
    p=fixtures();p['Tox21']['test']=({**item(8,80,90),'molecule':'scaffold:'+f'{80:064x}'},)
    with pytest.raises(ValueError,match='group'):derive_literal_components(p)


def test_real_original_sidecar_closure_matches_previous_immutable_role_audit(monkeypatch):
    """Original 197 source sidecars; no heldout numerical files opened."""
    import json
    from pathlib import Path
    from mora3.data.admet import canonical
    from hashlib import sha256
    from mora3.research.cross_source_chemical_components import audit_original_four_family_components

    root=Path('/home/iska/Documents/amelie/bio/Mora-3')
    previous=Path(__file__).resolve().parents[2]/'outputs/common-source-four-chemical-role-exposure-20261111/actual-original-four-family-group-exposure-v2.json'
    if not root.is_dir() or not previous.is_file():
        pytest.skip('Actual immutable Iska original-source root/receipt unavailable')
    before=json.loads(previous.read_text())
    previous_sha=sha256(previous.read_bytes()).hexdigest()
    monkeypatch.chdir(root)
    result=audit_original_four_family_components(root)
    assert sha256(previous.read_bytes()).hexdigest()==previous_sha
    assert result['parent_sidecar_readback_sha256']==sha256(canonical(before)).hexdigest()
    assert result['parent_sidecar_commitment_sha256']==before['source_role_file_commitment']
    assert result['parent_ordered_retained_commitment_sha256']==before['ordered_retained_group_commitment']
    assert result['closure']['total_original_rows']==79852
    assert result['original_sidecar_shards']==197
    assert result['closure']['axes']['combined']['total_components']==9354
    assert result['closure']['axes']['combined']['heldout_train_mixed_components']==0
    assert result['original_numerical_source_shards_read'] is False
    assert result['original_heldout_targets_decoded'] is False
    assert result['training_admission'] is False
    assert result['optimizer_updates']==0
    assert result['actual_campaign_runs']==0
