"""Read author-published plain NumPy TS bundles without general pickle globals."""
import pickle
import numpy as np


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np._core.multiarray, name)
        raise pickle.UnpicklingError("forbidden pickle global: " + module + "." + name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("forbidden persistent pickle reference")


def restricted_numpy_load(stream):
    value = _NumpyOnlyUnpickler(stream).load()
    def check(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise ValueError("object NumPy array forbidden")
        elif isinstance(x, dict):
            for k, v in x.items():
                check(k); check(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                check(v)
        elif isinstance(x, np.generic):
            if x.dtype.hasobject:
                raise ValueError("object NumPy scalar forbidden")
        elif x is not None and not isinstance(x, (str, bytes, int, float, bool)):
            raise ValueError("unsupported decoded object type")
    check(value)
    return value


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


def check_source_split_masks(train, heldout, n):
    a, b = list(map(int, train)), list(map(int, heldout))
    if len(a) != len(set(a)) or len(b) != len(set(b)):
        raise ValueError("duplicate split index")
    if set(a) & set(b) or set(a) | set(b) != set(range(n)):
        raise ValueError("source split masks must be disjoint and exhaustive")
    a = set(a)
    return ["source_train" if i in a else "source_heldout" for i in range(n)]


def ts_record(bundle, index, source_split):
    roles = {}; species = None; rid = None; formula = None
    for role in ("reactant", "transition_state", "product"):
        raw = {k: v[index] for k, v in bundle[role].items()}
        n = int(raw["num_atoms"])
        z = np.asarray(raw["charges"])
        pos = np.asarray(raw["positions"])
        if n <= 0 or z.shape != (n,) or pos.shape != (n, 3):
            raise ValueError("native atom/coordinate axis mismatch")
        if not np.all(np.isin(z, [1, 6, 7, 8])) or not np.isfinite(pos).all():
            raise ValueError("invalid species or nonfinite coordinates")
        if species is not None and not np.array_equal(species, z):
            raise ValueError("reactant/TS/product atom order not conserved")
        species = z
        if rid is not None and (rid != raw["rxn"] or formula != raw["formula"]):
            raise ValueError("source reaction identity differs by role")
        rid, formula = raw["rxn"], raw["formula"]
        fragments = [int(i) for frag in raw["fragments"] for i in frag]
        if sorted(fragments) != list(range(n)):
            raise ValueError("fragment partition must cover each source atom once")
        for key, value in raw.items():
            if key == "wB97x_6-31G(d).forces" or key.endswith("positions"):
                arr = np.asarray(value)
                if arr.shape != (n, 3) or not np.isfinite(arr).all():
                    raise ValueError("nonfinite or misaligned native vector payload")
            elif "energy" in key and not np.isfinite(value):
                raise ValueError("nonfinite native energy")
        roles[role] = _plain(raw)
        roles[role]["source_positions_dtype"] = str(pos.dtype)
    return dict(source_record_id="Transition1x:v4:"+str(formula)+":"+str(rid),
        source_row=index, source_split=source_split,
        source_reaction_group="Transition1x:v4:"+str(formula)+":"+str(rid),
        evidence_class="computational_quantum_chemistry", source_version="ReactOT:Zenodo13131875:v1",
        underlying_source="Transition1x:Figshare19614657:v4", atomic_numbers=list(map(int,species)),
        source_charges_field_meaning="atomic_numbers_not_formal_or_partial_charge",
        source_atom_axis="author paired same atom order; isotope/atom-map IDs unavailable",
        coordinate_unit="angstrom", frame="author_released_preprocessed_frame; no additional centering/reflection/alignment",
        method="wB97x/6-31G(d)", single_fragment=bool(bundle["single_fragment"][index]), roles=roles,
        energy_unit_claim="eV inferred from original ASE database producer; derivative rescaling not separately documented",
        force_unit_claim="eV/angstrom inferred from original ASE database producer; derivative rescaling not separately documented",
        source_extra_guess_keys=sorted(k for k in bundle if k.startswith("ts_guess")),
        source_guess_disposition="retained immutable in both raw assets; all guesses excluded from predictor pending derivation review, especially ts_guess_true",
        condition_candidate_fields=["atomic_numbers", "roles.reactant.positions", "roles.product.positions"],
        target_candidate_fields=["roles.transition_state.positions"],
        input_exclusion_fields=["roles.transition_state", "all forces and energies", "all ts_guess fields", "source IDs", "fragment topology derived from heldout target"],
        geometry_available=True, task_admitted=False,
        missing_context=["enzyme", "cellular_assay", "solvent", "independent_wet_experiment"],
        grouping_required=["source_reaction_group", "source official use_ind holdouts", "cross-source reactant/product identity", "reaction family"])


def ts_condition(record):
    return {"atomic_numbers": record["atomic_numbers"],
        "reactant_positions": record["roles"]["reactant"]["positions"],
        "product_positions": record["roles"]["product"]["positions"]}
