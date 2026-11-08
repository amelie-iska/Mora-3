"""Source-preserving PDBx/mmCIF, RGD1 and STRING adapters.

No external source code is executed. Scientific eligibility is separate from parsing.
Coordinates are Angstrom, retained in the deposited ASU frame without assembly expansion.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

RGD1_POLICY_PATH = Path(__file__).resolve().parents[1] / "planning/rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json"
RGD1_POLICY_SHA256 = "0f77c5310a55161c5d1723c0688b04addc56b5ede7241d6afcb522fd8414133b"


def _tokens(text):
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == ";" and (i == 0 or text[i - 1] == "\n"):
            j = text.find("\n;", i + 1)
            if j < 0:
                raise ValueError("unterminated CIF multiline text")
            yield text[i + 1:j], True
            end = text.find("\n", j + 2)
            i = n if end < 0 else end + 1
            continue
        if c in ("'", '"'):
            j = i + 1
            while True:
                j = text.find(c, j)
                if j < 0:
                    raise ValueError("unterminated CIF quoted text")
                if j + 1 == n or text[j + 1].isspace():
                    break
                j += 1
            yield text[i + 1:j], True
            i = j + 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        yield text[i:j], False
        i = j


def parse_cif(text):
    """Read one archival mmCIF data block, retaining every tag and row order."""
    ts = list(_tokens(text)); result = {}; i = 0; blocks = 0
    def control(t):
        value, quoted = t
        return not quoted and (value.startswith("_") or value.lower().startswith(("data_", "save_")) or value.lower() in ("loop_", "stop_", "global_"))
    while i < len(ts):
        value, quoted = ts[i]
        if not quoted and value.lower().startswith("data_"):
            blocks += 1
            if blocks > 1:
                raise ValueError("multiple CIF data blocks unsupported")
            i += 1
        elif not quoted and value.lower() == "loop_":
            i += 1; tags = []
            while i < len(ts) and not ts[i][1] and ts[i][0].startswith("_"):
                tags.append(ts[i][0]); i += 1
            if not tags:
                raise ValueError("empty CIF loop")
            vals = []
            while i < len(ts) and not control(ts[i]):
                vals.append(ts[i][0]); i += 1
            if len(vals) % len(tags):
                raise ValueError("incomplete CIF loop row")
            for j, tag in enumerate(tags):
                if tag in result:
                    raise ValueError("duplicate CIF tag")
                result[tag] = vals[j::len(tags)]
        elif not quoted and value.startswith("_"):
            if i + 1 >= len(ts) or control(ts[i + 1]):
                raise ValueError("CIF scalar missing value")
            if value in result:
                raise ValueError("duplicate CIF tag")
            result[value] = [ts[i + 1][0]]; i += 2
        elif not quoted and value.lower() == "stop_":
            i += 1
        else:
            raise ValueError("unsupported CIF token: " + value[:50])
    return result


def category_rows(cif, prefix):
    cols = {k[len(prefix):]: v for k, v in cif.items() if k.startswith(prefix)}
    if not cols:
        return []
    n = len(next(iter(cols.values())))
    if any(len(v) != n for v in cols.values()):
        raise ValueError("inconsistent category length")
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


def _number(s):
    if s in (None, ".", "?"):
        return None
    x = float(s)
    if not math.isfinite(x):
        raise ValueError("nonfinite numeric value")
    return x


def atom_rows(cif):
    rows = category_rows(cif, "_atom_site.")
    if not rows:
        raise ValueError("no atom_site coordinate rows")
    for i, row in enumerate(rows):
        row["source_atom_row_index"] = i
        for axis in "xyz":
            val = _number(row.get("Cartn_" + axis))
            if val is None:
                raise ValueError("coordinate missing")
            row[axis + "_angstrom"] = val
        row["occupancy_numeric"] = _number(row.get("occupancy"))
        row["b_iso_angstrom2"] = _number(row.get("B_iso_or_equiv"))
        occ = row["occupancy_numeric"]
        if occ is not None and not 0 <= occ <= 1:
            raise ValueError("occupancy outside [0,1]")
    return rows


ENERGY_MEANINGS = {
    "DE_F": "forward_electronic_activation_energy",
    "DE_B": "reverse_electronic_activation_energy",
    "DG_F": "forward_gibbs_activation_energy",
    "DG_B": "reverse_gibbs_activation_energy",
    "DH": "forward_reaction_enthalpy",
}


def _mapped_molecule(s):
    from rdkit import Chem
    params = Chem.SmilesParserParams(); params.removeHs = False
    mol = Chem.MolFromSmiles(s, params)
    if mol is None:
        raise ValueError("invalid sanitized mapped SMILES")
    axis = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if any(n <= 0 for n in axis) or len(set(axis)) != len(axis):
        raise ValueError("atom maps missing or duplicated")
    signature = {a.GetAtomMapNum(): (a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms()}
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    copy = Chem.RemoveHs(copy)
    fragments = sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in Chem.GetMolFrags(copy, asMols=True))
    return axis, signature, charge, fragments


def mapped_reaction(row):
    ra, rs, rc, rk = _mapped_molecule(row["reactant"])
    pa, ps, pc, pk = _mapped_molecule(row["product"])
    if rs != ps:
        raise ValueError("atom-map element/isotope conservation failure")
    if rc != pc:
        raise ValueError("total formal charge conservation failure")
    measured = {}
    for key, meaning in ENERGY_MEANINGS.items():
        val = _number(row[key])
        if val is None:
            raise ValueError("energy missing")
        measured[key] = dict(value=val, unit="kcal/mol", meaning=meaning)
    rid = row["reaction"]
    if not re.fullmatch(r"MR_\d+_\d+", rid):
        raise ValueError("unexpected RGD1 reaction/conformation identifier")
    direction_independent = sorted([".".join(rk), ".".join(pk)])
    identity = hashlib.sha256("<=>".join(direction_independent).encode()).hexdigest()
    return dict(source_record_id="RGD1:v6:" + rid, source_native=dict(row),
        evidence_class="computational_quantum_chemistry", source_release="RGD1.figshare.21066901.v6",
        reaction_family_key="RGD1:" + rid.rsplit("_", 1)[0],
        canonical_reaction_group="reaction:sha256:" + identity,
        reactant_entity_keys=["smiles:" + s for s in rk], product_entity_keys=["smiles:" + s for s in pk],
        reactant_atom_axis=ra, product_atom_axis=pa, atom_count=len(ra),
        atom_map_conserved=True, total_formal_charge=rc, measurements=measured,
        quantum_method="B3LYP-D3/TZVP", geometry_available=False,
        geometry_payload=None, experimental_validation=False, task_admitted=False,
        missing_context=["enzyme", "solvent", "cellular_assay", "transition_state_geometry_in_csv"],
        condition_candidate_fields=["source_native.reactant", "source_native.product"],
        target_candidate_fields=["measurements"],
        input_exclusion_fields=["source_record_id", "source_native.reaction", "source_native.DE_F", "source_native.DE_B", "source_native.DG_F", "source_native.DG_B", "source_native.DH", "measurements"],
        grouping_required=["reaction_family_key", "canonical_reaction_group", "reactant_entity_keys", "product_entity_keys"])


def physical_link(row):
    scores = {k: int(row[k]) for k in ("experimental", "database", "textmining", "combined_score")}
    if any(not 0 <= x <= 1000 for x in scores.values()):
        raise ValueError("STRING score outside [0,1000]")
    a, b = row["protein1"], row["protein2"]
    if not a.startswith("9606.") or not b.startswith("9606."):
        raise ValueError("unexpected species in registered human file")
    pair = "|".join(sorted([a, b]))
    return dict(source_record_id="STRING:v12.0:physical:9606:" + a + ":" + b,
        evidence_class="source_integrated_physical_association", source_native=dict(row),
        entity_keys=["string:v12.0:" + a, "string:v12.0:" + b],
        unordered_pair_group="string:v12.0:physical:" + pair,
        directed=False, score_channels=scores, score_scale="integer_0_to_1000",
        experimental_channel_supported=scores["experimental"] > 0,
        direct_binding_demonstrated=None, negative_interaction_label=None,
        protein_sequence_available=False, uniprot_sequence_admission=False,
        task_admitted=False, missing_context=["assay_publication_detail", "directness", "cell_line", "dose", "time", "sequence_annotation_score"],
        condition_candidate_fields=["entity_keys"], target_candidate_fields=["score_channels"],
        input_exclusion_fields=["score_channels", "source_native.experimental", "source_native.database", "source_native.textmining", "source_native.combined_score"],
        grouping_required=["entity_keys", "unordered_pair_group"])


def reaction_condition(record, *, audit_only=False):
    """Audit projection only: no RGD1 release currently qualifies for model use.

    Legacy admitted flags and claimed corrected release names cannot grant energy
    supervision, conditioning or quality evaluation. The explicit audit escape
    preserves codec/target-exclusion checks; its result is not a model task.
    A future qualified release requires a separately reviewed native interface.
    """
    try:
        policy_digest = hashlib.sha256(RGD1_POLICY_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("RGD1 quarantine policy integrity unavailable") from exc
    if policy_digest != RGD1_POLICY_SHA256:
        raise ValueError("RGD1 quarantine policy integrity mismatch")
    if audit_only is not True:
        raise ValueError("RGD1 energy quarantine: no qualified model-use release; audit only")
    raw = record["source_native"]
    return {k: raw[k] for k in ("reactant", "product")}


def known_atom_axis_condition(rows):
    """Observed complete atom-identity axis only; no claim of unknown-topology generation."""
    allowed = ("type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "label_entity_id", "pdbx_formal_charge")
    return [{k: r[k] for k in allowed if k in r} for r in rows]
