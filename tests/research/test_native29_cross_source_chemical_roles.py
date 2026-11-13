"""Pure test-only fixtures; no campaign histories, optimizers or training."""
import pytest

from mora3.research.cross_source_chemical_roles import compare_scoped_role_groups, ROLES


def row(n, molecule, scaffold):
    return {"task_id":f"{n:064x}", "molecule":f"molecule:{molecule:064x}",
            "scaffold":f"scaffold:{scaffold:064x}"}


def tiny_roles():
    return {
        "AZ":{"train":(row(1,1,1),)},
        "FreeSolv":{"train":(row(2,1,1),),
                    "validation":(row(3,3,1),),"test":(row(4,4,4),)},
        "Tox21":{"train":(row(5,5,5),),
                 "validation":(row(6,1,1),),"test":(row(7,6,4),)},
        "ExpansionRx":{"train":(row(8,8,8),),"test":(row(9,9,5),)}
    }


def test_expected_roles_and_scaffold_not_molecule_equivalence():
    d=compare_scoped_role_groups(tiny_roles())
    assert set(d["pairwise"])==set(d["required_pairs"])
    assert d["train_pairwise"]["AZ.train|FreeSolv.train"]["molecule"]["shared_hashes"]==1
    c=d["pairwise"]["FreeSolv.validation|AZ.train"]
    assert c["scaffold"]["shared_hashes"]==1
    assert c["molecule"]["shared_hashes"]==0
    assert c["scaffold"]["left_tasks"]==c["scaffold"]["right_tasks"]==1
    assert d["pairwise"]["Tox21.validation|AZ.train"]["molecule"]["shared_hashes"]==1
    assert d["pairwise"]["FreeSolv.test|Tox21.test"]["scaffold"]["shared_hashes"]==1
    assert d["exact_stereochemical_identity_proven"] is False
    assert d["global_split_admitted"] is False


def test_reject_missing_role_and_duplicate_task_id():
    d=tiny_roles()
    del d["Tox21"]["validation"]
    with pytest.raises(ValueError,match="role"):compare_scoped_role_groups(d)
    d=tiny_roles()
    d["ExpansionRx"]["test"]=(row(1,9,5),)
    with pytest.raises(ValueError,match="task"):compare_scoped_role_groups(d)


def test_reject_malformed_group_hashes_and_unbound_extra_source():
    d=tiny_roles()
    d["FreeSolv"]["test"]=({"task_id":f"{4:064x}",
                            "molecule":"molecule:UNKNOWN",
                            "scaffold":f"scaffold:{4:064x}"},)
    with pytest.raises(ValueError,match="group"):compare_scoped_role_groups(d)
    d=tiny_roles();d["Other"]={"train":(row(100,100,100),)}
    with pytest.raises(ValueError,match="role"):compare_scoped_role_groups(d)


def test_reject_same_hash_used_as_different_group_types():
    d=tiny_roles()
    d["FreeSolv"]["test"]=({"task_id":f"{4:064x}",
                            "molecule":f"scaffold:{4:064x}",
                            "scaffold":f"scaffold:{4:064x}"},)
    with pytest.raises(ValueError,match="group"):compare_scoped_role_groups(d)
