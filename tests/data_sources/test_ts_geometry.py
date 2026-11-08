from pathlib import Path
import copy
import sys
import numpy as np
import pytest
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT / "scripts"))
from data_sources_ts import ts_record, ts_condition, check_source_split_masks


def bundle():
    d = {}
    for role in ("reactant", "transition_state", "product"):
        d[role] = {"num_atoms": [2], "charges": [[6, 8]], "positions": [np.array([[0., 0., 0.], [1.2, 0., 0.]])], "fragments": [[[0], [1]]], "rxn": ["rxn1"], "formula": ["CO"], "wB97x_6-31G(d).energy": [-2000.], "wB97x_6-31G(d).forces": [[[0., 0., 0.], [0., 0., 0.]]], "wB97x_6-31G(d).atomization_energy": [1.]}
    d["single_fragment"] = [0]
    return d


def test_ts_preserves_full_multifragment_native_axis():
    r = ts_record(bundle(), 0, "source_train")
    assert r["atomic_numbers"] == [6, 8]
    assert r["single_fragment"] is False
    assert r["roles"]["reactant"]["fragments"] == [[0], [1]]
    assert r["coordinate_unit"] == "angstrom"
    assert r["source_split"] == "source_train"


@pytest.mark.parametrize("field,value", [("charges", [6, 7]), ("positions", [[0,0,0]]), ("fragments", [[0], [0]]), ("positions", [[0,0,0],[float("nan"),0,0]])])
def test_corrupted_native_axis_is_rejected(field, value):
    d = bundle(); d["product"][field][0] = value
    with pytest.raises(ValueError): ts_record(d, 0, "source_train")


def test_heldout_masks_must_be_disjoint_exhaustive_and_unique():
    assert check_source_split_masks([0, 2], [1], 3) == ["source_train", "source_heldout", "source_train"]
    for train, heldout in [([0, 1], [1, 2]), ([0], [1]), ([0, 0], [1, 2])]:
        with pytest.raises(ValueError): check_source_split_masks(train, heldout, 3)


def test_ts_targets_and_guesses_cannot_change_conditions():
    r = ts_record(bundle(), 0, "source_train")
    changed = copy.deepcopy(r)
    changed["roles"]["transition_state"]["positions"] = [[999,999,999], [999,999,999]]
    changed["roles"]["reactant"]["wB97x_6-31G(d).energy"] = 999
    changed["source_extra_guess_keys"] = ["ts_guess_true"]
    assert ts_condition(r) == ts_condition(changed)
