from pathlib import Path
import io
import pickle
import sys
import numpy as np
import pytest
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(ROOT / "scripts"))
from data_sources_ts import restricted_numpy_load


def test_restricted_numpy_decoder_roundtrips_plain_native_arrays():
    data = {"positions": np.array([[1.25, 2.5, 3.75]], dtype=np.float64), "atoms": np.array([6], dtype=np.int64)}
    decoded = restricted_numpy_load(io.BytesIO(pickle.dumps(data, protocol=4)))
    assert np.array_equal(decoded["positions"], data["positions"])
    assert decoded["positions"].dtype == data["positions"].dtype


def test_non_numpy_global_is_rejected_before_reconstruction():
    class Payload:
        def __reduce__(self):
            return eval, ("1+1",)
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        restricted_numpy_load(io.BytesIO(pickle.dumps(Payload())))


def test_object_arrays_are_rejected():
    with pytest.raises(ValueError, match="object"):
        restricted_numpy_load(io.BytesIO(pickle.dumps(np.array([{"a": 1}], dtype=object), protocol=4)))
