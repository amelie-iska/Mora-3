"""Original NCI AIDS source-native assay views, not clinical or causal claims.

Only pinned v19/v20 corpus admission is consumed. Unknown source stereo and
computed Corina coordinates never become observed structure labels. Native
decimal literals remain exact; no float conversion, censor bounds or IC/EC ratio
is created. File verification and projection do not establish neural support.
"""
import csv
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import io
from itertools import islice
import json
from pathlib import Path
import re

from .uniprot_annotations import _canonical, _verified_bytes

BASE = "outputs/data-intake/20261101-NCI-AIDS-original-v1"
MASTER_SHA256S = frozenset({
    "4fa73767f29703d851a487690e16327537af4029ac3bad654d41c382821c1d7e",
    "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"})
SOURCE_REF = dict(path=BASE + "/curated-native-v2/manifest.json", bytes=58597,
    sha256="cdebc59bace1f2776c0e2bd6c04a939e737b2944fc6e2291dd737ac510ffcca9")
REVISION = "officialNCIcurrentpublicSHA_snapshot;provider1999/2004releaseclaims_distinct"
POLICY = "NSC+molecule+scaffoldconnectedgroups;oldallocationsauthoritative;newseeded80/10/10;publicwithincollectionchemicalholdout_notblindstudy"
EVIDENCE = "originalsource_assay_conclusion_or_computed_measurement_summary;referenceconnectivity;notclinicalvalidation"
PANEL = "NCI_public_AIDS_screen_source_protocol"
READOUT = {"EC50": "protection_of_infected_cells", "IC50": "growth_inhibition_of_uninfected_cells"}
CLASS_AXIS = ("CA", "CM", "CI")
PAIRED_AXIS = ("infected_cell_protection_EC50", "uninfected_cell_growth_IC50")
FAMILIES = {
    "NCI-AIDS-source-conclusion-prior": "source_NCI_assay_conclusion_",
    "NCI-AIDS-source-endpoint-prior": "source_NCI_assay_endpoint_",
    "NCI-AIDS-paired-assay-profile-prior": "source_NCI_paired_assay_profile_"}
MISSING = ["verified_stereochemical_identity_and_lot",
    "exact_cell_line_strain_duration_and_replicate_context",
    "same_cell_image_expression_or_segmentation_pair", "clinical_outcome_or_mechanism_validation",
    "independent_study_blind_evaluation"]
CONTEXT = dict(source_collection="NCI_DTP_public_AIDS", source_study_ID=None,
    unknown_stereochemistry_not_achirality=True, profile_is_not_image=True,
    source_cell_line=None, source_exposure_duration=None)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _binding(ib, tb, pb):
    return _sha(_canonical(dict(schema="mora3.NciAidsView/v1",
        inputs=_sha(ib), targets=_sha(tb), provenance=_sha(pb))))


@dataclass(frozen=True)
class NciEndpointState:
    endpoint: str
    source_nsc: str
    unit: str
    value: Decimal
    highest_concentration: Decimal
    standard_deviation: Decimal
    point_target_available: bool
    positive_uncertainty_available: bool
    source_flag: str
    value_literal: str
    highest_literal: str
    source_record_id: str


@dataclass(frozen=True)
class NciAidsView:
    source_family: str
    direction: str
    split: str
    axis: tuple[str, ...]
    input_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    binding_sha256: str

    def inputs(self):
        return json.loads(self.input_bytes)

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)

    @property
    def class_axis(self):
        return tuple(self.provenance()["class_axis"])

    @property
    def source_point_values(self):
        return tuple(Decimal(v) for v in self.provenance()["source_point_literals"])

    @property
    def source_point_mask(self):
        return tuple(self.provenance()["source_point_mask"])


def nci_endpoint_state(row):
    """Replay native decimal/mask semantics; never grant task admission.

    Censored aggregates and unit V retain exact source values, with a false
    point mask. An aggregate censor flag is not a bound on its reported mean.
    """
    try:
        return _endpoint_state(row)
    except (KeyError, TypeError, InvalidOperation, OverflowError) as exc:
        raise ValueError("malformed original NCI endpoint record") from exc


def _endpoint_state(row):
    native = json.loads(row["source_native_fields_json"])
    fields = {k.strip(): v.strip() for k, v in native.items()}
    if len(fields) != len(native):
        raise ValueError("ambiguous native endpoint columns")
    endpoint, nsc = row["endpoint_kind"], fields["NSC"]
    if endpoint not in READOUT or not re.fullmatch("[0-9]+", nsc):
        raise ValueError("explicit original endpoint and NSC required")
    value_literal, high_literal = fields["Log10" + endpoint], fields["Log10HiConc"]
    value, high, sd = (Decimal(value_literal), Decimal(high_literal), Decimal(fields["StdDev"]))
    num = int(fields["NumExp"])
    flag, unit = fields["Flag"], fields["ConcUnit"]
    if (not all(v.is_finite() for v in (value, high, sd)) or sd < 0 or num <= 0
            or flag not in ("=", ">", "<") or unit not in ("M", "u", "V")):
        raise ValueError("finite native endpoint/count/flag/unit required")
    p = json.loads(row["projection_json"])
    for key, exact in (("source_value", value), ("source_log10_highest_concentration", high), ("source_stddev", sd)):
        if type(p[key]) not in (int, float) or Decimal(str(p[key])) != exact:
            raise ValueError("source decimal precision/value mismatch; no float coercion")
    known = unit in ("M", "u")
    gap = endpoint == "IC50" and (unit == "V" or flag == "<")
    available = flag == "=" and known and not gap
    high_key = format(high.normalize(), "f")
    expected = dict(source_NSC=nsc, endpoint=endpoint, source_native_fields=native,
        source_value=p["source_value"], source_value_scale="log10_concentration_explicit_native_header",
        source_log10_highest_concentration=p["source_log10_highest_concentration"],
        source_ConcUnit=unit, physical_unit_mapping_available=known,
        unit_meaning="molar" if unit == "M" else "micrograms_per_mL" if unit == "u" else "unresolved_volumetric_definition",
        source_flag=flag, censoring_role="all_source_experiments_reached_endpoint" if flag == "=" else
            "source_aggregate_flag_at_least_one_experiment;not_bound_of_aggregate_mean",
        point_target_available=available, source_NumExp=num, source_stddev=p["source_stddev"],
        positive_uncertainty_target_available=sd > 0 and num > 1 and available,
        zero_SD_is_not_perfect_precision=True, source_documentation_gap=gap,
        context_key=[nsc, endpoint, unit, high_key], compound_highestdose_context_key=[nsc, unit, high_key],
        evidence_class="source_computed_summary_of_assay_measurements_not_prediction",
        same_cell_or_replicate_pair_asserted=False)
    idx = row["source_row_index"]
    if (_canonical(p) != _canonical(expected) or row["source_NSC"] != nsc or type(idx) is not int or idx < 0
            or row["source_record_id"] != "NCI-AIDS-May2004-" + endpoint + ":" + str(idx)
            or row["source_schema_status"] != "native_schema_pass" or row["source_schema_error"] is not None
            or row["assay_readout"] != READOUT[endpoint]
            or row["same_cell_or_assay_replicate_pair_asserted"] is not False
            or row["source_cell_line"] is not None or row["source_exposure_time"] is not None):
        raise ValueError("native endpoint identity/availability/context mismatch")
    return NciEndpointState(endpoint, nsc, unit, value, high, sd, available,
        expected["positive_uncertainty_target_available"], flag, value_literal,
        high_literal, row["source_record_id"])


def _structure_projection(row):
    """Replay source CTAB on an in-memory M END syntax copy, preserving raw bytes."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
    if importlib.metadata.version("rdkit") != "2026.3.1":
        raise ValueError("pinned source RDKit runtime required")
    raw = row["all_native_record_bytes"]
    if (type(raw) is not bytes or _sha(raw) != row["native_record_sha256"]
            or row["native_record_bytes"] != len(raw)
            or row["graph_projection_status"] != "graph_projection_pass"
            or row["graph_projection_error"] is not None
            or row["source_geometry_observed"] is not False
            or row["source_stereochemistry_known"] is not False):
        raise ValueError("native CTAB integrity/graph/stereo role mismatch")
    lines = raw.decode("ascii").splitlines()
    if len(lines) < 4:
        raise ValueError("complete original CTAB required")
    start = next((i for i, line in enumerate(lines) if line.startswith(">")),
        len(lines) - int(lines[-1].strip() == "$$$$"))
    ct = lines[:start]
    while ct and not ct[-1].strip():
        ct.pop()
    na, nb = int(ct[3][:3]), int(ct[3][3:6])
    if na < 1 or nb < 0 or len(ct) < 4 + na + nb:
        raise ValueError("full native atom/bond axes required")
    atoms, bonds, extra = ct[4:4+na], ct[4+na:4+na+nb], ct[4+na+nb:]
    if any(line.strip() and not line.startswith(("M  CHG", "M  ISO", "M  RAD", "M  END")) for line in extra):
        raise ValueError("unsupported original CTAB extension")
    has_end = any(line.strip() == "M  END" for line in ct)
    adapted = "\n".join(ct + ([] if has_end else ["M  END"])) + "\n"
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(adapted, sanitize=False, removeHs=False, strictParsing=True)
    if mol is None or mol.GetNumAtoms() != na or mol.GetNumBonds() != nb:
        raise ValueError("native CTAB strict graph projection mismatch")
    periodic = Chem.GetPeriodicTable()
    for i, line in enumerate(atoms):
        symbol = line[31:34].strip()
        expected = 1 if symbol in ("D", "T") else periodic.GetAtomicNumber(symbol)
        if expected <= 0 or mol.GetAtomWithIdx(i).GetAtomicNum() != expected:
            raise ValueError("native atom element mismatch")
    orders = {1: 1., 2: 2., 3: 3., 4: 1.5}
    for line in bonds:
        a, b, order = int(line[:3])-1, int(line[3:6])-1, int(line[6:9])
        bond = mol.GetBondBetweenAtoms(a, b)
        if order not in orders or bond is None or bond.GetBondTypeAsDouble() != orders[order]:
            raise ValueError("native bond endpoint/order mismatch")
    with rdBase.BlockLogs():
        Chem.SanitizeMol(mol)
    projected = Chem.Mol(mol)
    Chem.RemoveStereochemistry(projected)
    projected.RemoveAllConformers()
    with rdBase.BlockLogs():
        heavy = Chem.RemoveHs(projected)
        smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
        back = Chem.MolFromSmiles(smiles)
    if (back is None or Chem.MolToSmiles(back, canonical=True, isomericSmiles=True) != smiles
            or Chem.GetFormalCharge(back) != Chem.GetFormalCharge(projected)
            or rdMolDescriptors.CalcMolFormula(back) != rdMolDescriptors.CalcMolFormula(projected)):
        raise ValueError("derived source connectivity formula/charge roundtrip mismatch")
    properties, i = {}, start
    while i < len(lines):
        match = re.match(r"^>.*<([^<>]+)>", lines[i])
        if match:
            key, values = match[1], []
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(">") and lines[i].strip() != "$$$$":
                values.append(lines[i])
                i += 1
            if key in properties:
                raise ValueError("duplicate native SDF property")
            properties[key] = "\n".join(values)
        else:
            i += 1
    nsc = properties["NSC"].strip()
    expected = dict(native_NSC=nsc, native_title=lines[0], native_program_header=lines[1:3],
        native_count_line=lines[3], native_atom_count=na, native_bond_count=nb,
        native_atom_lines=atoms, native_bond_lines=bonds, native_extra_CTAB_lines=extra,
        native_properties=properties, native_record_sha256=_sha(raw), derived_syntax_added_M_END=not has_end,
        all_literal_native_elements_bond_endpoints_and_orders_verified=True,
        derived_nonstereo_connectivity_smiles=smiles, derived_charge=Chem.GetFormalCharge(back),
        derived_formula=rdMolDescriptors.CalcMolFormula(back), source_stereochemistry_known=False,
        source_geometry_role="source_computed_Corina_no_optimization_not_observed",
        derived_graph_coordinates_used=False, source_hydrogen_axes_retained_in_native_tables=True,
        derived_SMILES_preserves_chemical_formula_charge_and_isotopes=True)
    index = row["source_record_index"]
    if (_canonical(json.loads(row["projection_json"])) != _canonical(expected) or not re.fullmatch("[0-9]+", nsc)
            or row["source_NSC"] != nsc or type(index) is not int or index < 0
            or row["source_record_id"] != "NCI-AIDS-Oct1999-structure:" + str(index)
            or type(row["native_byte_offset"]) is not int or row["native_byte_offset"] < 0):
        raise ValueError("original native/derived projection identity mismatch")
    return expected


def nci_aids_view(row, structure, outcomes, *, context_link=None, effective_groups, corpus_binding):
    """Project independently verified native rows; file reader below verifies IO."""
    try:
        return _project(row, structure, outcomes, context_link, effective_groups, corpus_binding)
    except (KeyError, TypeError, IndexError, InvalidOperation, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed source-native NCI task") from exc


def _project(row, structure, outcomes, link, effective_groups, binding):
    raw = row["native_payload"]
    if type(raw) is not bytes or _sha(raw) != row["native_payload_sha256"]:
        raise ValueError("native task payload integrity mismatch")
    d = json.loads(raw)
    family = row["source_family"]
    direction = next((x for x in ("prediction", "generation")
        if family in FAMILIES and row["task_name"] == FAMILIES[family] + x + "_prior"), None)
    if (direction is None or set(d) != {"schema", "task", "conditions", "targets", "audit", "missing"}
            or _canonical(d) != raw or d["schema"] != "mora3.CorpusTask/v1"
            or row["native_schema"] != d["schema"] or d["task"] != row["task_name"]
            or row["source_revision"] != REVISION or row["source_sha256"] != SOURCE_REF["sha256"]
            or row["admitted_for_task"] is not True or row["split"] not in ("train", "validation", "test")
            or row["split_policy"] != POLICY or row["evidence_class"] != EVIDENCE):
        raise ValueError("original NCI task/source admission mismatch")
    if (set(binding) != {"manifest_sha256", "shard_sha256", "groups_sha256", "split"}
            or binding["manifest_sha256"] not in MASTER_SHA256S
            or any(not _hex(binding[k]) for k in ("manifest_sha256", "shard_sha256", "groups_sha256"))
            or binding["split"] != row["split"]):
        raise ValueError("pinned master/shard/effective split bindings required")
    native = _structure_projection(structure)
    nsc, smiles = native["native_NSC"], native["derived_nonstereo_connectivity_smiles"]
    ref = d["audit"]["source_ref"]
    paired, conclusion = "paired" in family, "conclusion" in family
    source_rows = [ref["structure"]] + (ref["outcomes"] + [ref["context_link"]] if paired else [ref["outcome"]])
    if any(type(rr["row_index"]) is not int or rr["row_index"] < 0 for rr in source_rows):
        raise ValueError("exact native integer source row indices required")
    ref_keys = {"native_source", "structure", "source_NSC"} | (
        {"outcomes", "context_link"} if paired else {"outcome"})
    if (set(ref) != ref_keys or ref["native_source"] != SOURCE_REF or ref["source_NSC"] != nsc
            or ref["structure"]["native_record_sha256"] != structure["native_record_sha256"]
            or ref["structure"]["row_index"] != structure["source_record_index"] % 1000):
        raise ValueError("original structure/source reference mismatch")
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    graph = Chem.MolFromSmiles(smiles)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(0)
    graph = Chem.RemoveHs(graph)
    Chem.RemoveStereochemistry(graph)
    base = Chem.MolToSmiles(graph, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=graph, includeChirality=False) or base
    guards = {"NCI-source-NSC:" + nsc, "molecule:" + _sha(base.encode()), "scaffold:" + _sha(scaffold.encode())}
    original, effective = json.loads(row["group_ids_json"]), tuple(effective_groups)
    if (not isinstance(original, list) or any(not isinstance(g, str) or not g for g in original)
            or len(original) != 4 or len(set(original)) != 4
            or not guards <= set(original) or not any(g.startswith("component:") and _hex(g[10:]) for g in original)
            or not effective or len(set(effective)) != len(effective)
            or any(not isinstance(g, str) or not g for g in effective) or not set(original) <= set(effective)):
        raise ValueError("complete NSC/chemical/scaffold/component effective groups required")
    receipt = dict(source_ref=ref, rights="NCIpublicnonconfidentialdata;NCIcopyrightfreepolicy_withcredit",
        source_stereochemistry_known=False, source_geometry_notobserved=True,
        original_structure_count_discrepancy297=True, source_lot_study_time_cell_context_incomplete=True,
        withincollectionchemicalholdout_only=True, new_independent_experiment=False)
    if (_canonical(json.loads(row["admission_receipt_json"])) != _canonical(receipt) or d["missing"] != MISSING
            or json.loads(row["missing_context_json"]) != MISSING
            or json.loads(row["condition_context_json"]) != CONTEXT):
        raise ValueError("source rights/missingness/context admission mismatch")
    base_conditions = dict(assay_panel=PANEL, source_stereochemistry_known=False)
    audit = dict(source_ref=ref, future_generated_outputs_require_model_produced_provenance=True)
    point_literals, states, label, unit = [], [], None, None
    if conclusion:
        if len(outcomes) != 1 or link is not None:
            raise ValueError("one original source conclusion required")
        outcome = outcomes[0]
        fields = {k.strip(): v.strip() for k, v in json.loads(outcome["source_native_fields_json"]).items()}
        label = fields["Conclusion"]
        rid = "NCI-AIDS-May2004-conclusion:" + str(outcome["source_row_index"])
        if (label not in CLASS_AXIS or fields["NSC"] != nsc or outcome["source_NSC"] != nsc
                or outcome["source_conclusion"] != label or outcome["source_record_id"] != rid
                or ref["outcome"]["row_index"] != outcome["source_row_index"]
                or outcome["source_geometry_or_stereospecific_claim"] is not False
                or outcome["source_conclusion_role"] != "manual_overall_judgment_from_individual_dose_response_curves"
                or outcome["secondary_binary_projection_CA_CM_positive"] != int(label in ("CA", "CM"))
                or outcome["secondary_binary_projection_is_not_new_observation"] is not True
                or structure["source_record_id"] not in json.loads(outcome["source_structure_record_ids_json"])):
            raise ValueError("original source conclusion identity/label mismatch")
        if direction == "prediction":
            conditions = dict(base_conditions, molecule_source_connectivity_smiles=smiles)
            targets = dict(source_conclusion=label)
        else:
            conditions = dict(base_conditions, desired_source_conclusion=label)
            targets = dict(molecule_source_connectivity_smiles=smiles)
        axis = ("source_conclusion",)
        audit.update(evidence="source_manual_overall_judgment_of_experimental_dose_response_curves;not_direct_binding_or_clinical_efficacy",
            source_geometry="Corina_computed_not_input_or_observed_target",
            source_missing_stereochemistry_is_not_achirality=True)
    else:
        if len(outcomes) != (2 if paired else 1):
            raise ValueError("complete source endpoint axis required")
        states = [nci_endpoint_state(o) for o in outcomes]
        if any(s.source_nsc != nsc or not s.point_target_available for s in states):
            raise ValueError("only known-unit uncensored source means are point tasks")
        refs = ref["outcomes"] if paired else [ref["outcome"]]
        if any(rr["row_index"] != o["source_row_index"] for rr, o in zip(refs, outcomes)):
            raise ValueError("original outcome row binding mismatch")
        point_literals = [s.value_literal for s in states]
        projections = [json.loads(o["projection_json"]) for o in outcomes]
        unit = states[0].unit
        base_conditions.update(concentration_unit=unit,
            source_log10_highest_tested_concentration=projections[0]["source_log10_highest_concentration"])
        if paired:
            ec, ic = states
            high_key = format(ec.highest_concentration.normalize(), "f")
            if (link is None or (ec.endpoint, ic.endpoint) != ("EC50", "IC50")
                    or ic.unit != ec.unit or ic.highest_concentration != ec.highest_concentration
                    or link["source_NSC"] != nsc or link["source_ConcUnit"] != unit
                    or link["source_log10_highest_concentration_literal"] != high_key
                    or link["exact_one_per_endpoint_and_both_source_point_masks"] is not True
                    or link["paired_assay_context"] is not True
                    or json.loads(link["EC50_source_record_ids_json"]) != [ec.source_record_id]
                    or json.loads(link["IC50_source_record_ids_json"]) != [ic.source_record_id]
                    or link["same_cell_or_replicate_link"] is not False
                    or link["source_lot_identity_or_stereochemistry_verified"] is not False
                    or link["release_dates_equal"] is not False
                    or link["unique_successful_source_connectivity_graph"] is not True
                    or link["multiple_successful_source_connectivity_graphs"] is not False
                    or structure["source_record_id"] not in json.loads(link["all_source_structure_record_ids_json"])
                    or link["join_level"] != "literalNSC+same_source_concentrationunit+same_log10highesttestedconcentration;infected/uninfectedseparateassays"):
                raise ValueError("strict separate-assay compound/unit/highest-dose context mismatch")
            rid = "NCI-original-paired-context:" + _sha(_canonical([nsc, unit, high_key]))
            axis = PAIRED_AXIS
            base_conditions["endpoint_axis"] = list(axis)
            values = [p["source_value"] for p in projections]
            desired, target = "desired_source_log10_endpoint_means", "source_log10_endpoint_means"
            audit.update(evidence="two_source_assay_summary_means;compound_and_dose_range_context_link",
                same_cell_or_replicate_pair=False, source_lot_or_full_protocol_identity_verified=False,
                selectivity_ratio_or_therapeutic_index_derived=False, source_censor_flags=["=", "="])
        else:
            if link is not None:
                raise ValueError("single endpoint has no paired-context assertion")
            rid, axis = states[0].source_record_id, ("source_log10_endpoint_mean",)
            base_conditions["assay_readout"] = READOUT[states[0].endpoint]
            values = projections[0]["source_value"]
            desired, target = "desired_source_log10_endpoint_mean", "source_log10_endpoint_mean"
            audit.update(source_endpoint=states[0].endpoint,
                evidence="source_computed_summary_of_experimental_assay_measurements;allreportedexperimentsreachedendpoint",
                source_flag="=", native_unit_role="M_molar_or_u_micrograms_per_mL;no_mass_to_molar_conversion",
                source_mean_scale="log10concentration_nativeheader",
                reported_SD_and_NumExp="retained_source_refs;notpredictors;zeroSDnotperfectprecision;noSEM_inference",
                source_geometry="computed_notobserved")
        if direction == "prediction":
            conditions = dict(base_conditions, molecule_source_connectivity_smiles=smiles)
            targets = {target: values}
        else:
            conditions = dict(base_conditions, **{desired: values})
            targets = dict(molecule_source_connectivity_smiles=smiles)
    expected_task = dict(schema="mora3.CorpusTask/v1", task=row["task_name"],
        conditions=conditions, targets=targets, audit=audit, missing=MISSING)
    if (row["record_id"] != rid or raw != _canonical(expected_task)
            or type(row["full_axis_values"]) is not int
            or row["full_axis_values"] != (len(smiles) if direction == "generation" else 2 if paired else 1)
            or row["task_id"] != _sha(_canonical([family, rid, raw.decode()]))):
        raise ValueError("original source target/condition/direction/full-axis mismatch")
    ib, tb = _canonical(conditions), _canonical(targets)
    p = dict(schema="mora3.NciAidsAdmission/v1", envelope={k: v for k, v in row.items() if k != "native_payload"},
        native_payload=d, source_ref=ref, original_native_structure={k: v for k, v in structure.items() if k != "all_native_record_bytes"},
        original_native_outcomes=outcomes, original_context_link=link,
        effective_groups=effective, corpus_binding=binding, axis=axis,
        class_axis=CLASS_AXIS if conclusion else (), source_class_label=label,
        source_point_literals=point_literals, source_point_mask=[s.point_target_available for s in states],
        point_role="observed_target" if direction == "prediction" else "desired_condition",
        concentration_unit=unit, source_stereochemistry_known=False, unknown_stereo_is_not_achirality=True,
        source_geometry_observed=False, source_geometry_role="source_computed_Corina_not_observed_or_model_input",
        graph_role="derived_nonstereo_reference_connectivity_not_verified_exact_stereo_or_lot",
        same_cell_or_replicate_pair=False, selectivity_ratio_derived=False,
        blind_or_new_study_evaluation=False, evaluation_scope="public_within_collection_chemical_group_holdout",
        raw_source_readback_verified=False, neural_support=False, new_independent_experiments=0)
    pb = _canonical(p)
    return NciAidsView(family, direction, row["split"], axis, ib, tb, pb, _binding(ib, tb, pb))


def collate_nci_aids_views(views):
    """Lossless homogeneous byte/Decimal collation; no model/tensor conversion."""
    views = tuple(views)
    if not views or len(views) > 256 or any(not isinstance(v, NciAidsView) for v in views):
        raise ValueError("bounded native NCI views required")
    first = views[0]
    def signature(v):
        p = v.provenance()
        if (v.binding_sha256 != _binding(v.input_bytes, v.target_bytes, v.provenance_bytes)
                or v.axis != tuple(p["axis"]) or v.split != p["envelope"]["split"]
                or v.source_family != p["envelope"]["source_family"]
                or p["envelope"]["task_name"] != FAMILIES[v.source_family] + v.direction + "_prior"):
            raise ValueError("native view binding mismatch")
        return (v.source_family, v.direction, v.split, v.axis, p["concentration_unit"],
            v.inputs().get("assay_readout"), p["point_role"])
    signature_first = signature(first)
    if any(signature(v) != signature_first for v in views):
        raise ValueError("collation requires matching family/direction/split/axis/unit/role")
    return dict(source_family=first.source_family, direction=first.direction, split=first.split,
        axis=first.axis, input_bytes=tuple(v.input_bytes for v in views),
        target_bytes=tuple(v.target_bytes for v in views),
        source_point_values=tuple(v.source_point_values for v in views),
        source_point_masks=tuple(v.source_point_mask for v in views),
        point_role=first.provenance()["point_role"], concentration_unit=first.provenance()["concentration_unit"],
        source_class_labels=tuple(v.provenance()["source_class_label"] for v in views),
        bindings=tuple(v.binding_sha256 for v in views))


class _NativeSources:
    """Per-call bounded read-only native/source caches; never persist indexes."""
    def __init__(self, root):
        self.root, self.tables, self.raw_text, self.verified, self.receipts = root, {}, {}, set(), {}
        self.curated = json.loads(_verified_bytes(root, SOURCE_REF))
        acquisition = self.curated["source_acquisition"]
        self.acquisition = json.loads(_verified_bytes(root, acquisition))
        c = self.curated
        self.allowed = {r["path"]: r for r in [*c["structure_files"], c["class_file"], *c["endpoint_files"], c["context_links"]]}
        self.assets = {a["filename"]: a["raw"] for a in self.acquisition["assets"]}

    def row(self, ref):
        import pyarrow.parquet as pq
        receipt = self.allowed.get(ref["path"])
        if receipt is None or receipt["sha256"] != ref["sha256"]:
            raise ValueError("native table absent from pinned source manifest")
        key = ref["path"]
        if key not in self.tables:
            if len(self.tables) >= 4:
                self.tables.pop(next(iter(self.tables)))
            table = pq.read_table(io.BytesIO(_verified_bytes(self.root, receipt)), use_threads=False)
            if len(table) != receipt["rows"]:
                raise ValueError("native source table row-count mismatch")
            self.tables[key] = table
        index = ref["row_index"]
        if type(index) is not int or not 0 <= index < len(self.tables[key]):
            raise ValueError("native source row index out of range")
        return self.tables[key].slice(index, 1).to_pylist()[0]

    def verify_stream(self, receipt):
        key = receipt["path"]
        if key in self.verified:
            return
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path.stat().st_size != receipt["bytes"]:
            raise ValueError("raw source path/size mismatch")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(2**20):
                h.update(chunk)
        if h.hexdigest() != receipt["sha256"]:
            raise ValueError("raw source integrity mismatch")
        self.verified.add(key)
        self.receipts[key] = receipt

    def structure(self, ref):
        row = self.row(ref)
        receipt = self.curated["source_native_SDF"]
        if (row["source_native_SDF_path"] != receipt["path"]
                or row["source_native_SDF_sha256"] != receipt["sha256"]
                or row["native_record_sha256"] != ref["native_record_sha256"]):
            raise ValueError("original SDF reference mismatch")
        self.verify_stream(receipt)
        self.verify_stream(self.assets["aido99sd.bin"])
        with (self.root / receipt["path"]).open("rb") as stream:
            stream.seek(row["native_byte_offset"])
            raw = stream.read(row["native_record_bytes"])
        if raw != row["all_native_record_bytes"] or _sha(raw) != row["native_record_sha256"]:
            raise ValueError("literal original SDF record readback mismatch")
        return row

    def outcome(self, ref):
        row = self.row(ref)
        endpoint = row.get("endpoint_kind")
        filename = "aids_conc_may04.txt" if endpoint is None else "aids_" + endpoint.lower() + "_may04.txt"
        receipt = self.assets[filename]
        if filename not in self.raw_text:
            self.raw_text[filename] = _verified_bytes(self.root, receipt).decode()
            self.receipts[receipt["path"]] = receipt
        reader = csv.DictReader(io.StringIO(self.raw_text[filename]))
        index = row["source_row_index"]
        if type(index) is not int or index < 0:
            raise ValueError("raw outcome source row index mismatch")
        raw = next(islice(reader, index, index+1), None)
        if raw != json.loads(row["source_native_fields_json"]):
            raise ValueError("original native outcome fields readback mismatch")
        return row


def read_nci_aids_views(manifest, *, root, expected_manifest_sha256, source_family,
                        direction, split, limit, task_ids=None):
    """Bounded admitted reader with exact original native bytes and effective splits."""
    import pyarrow.parquet as pq
    if (expected_manifest_sha256 not in MASTER_SHA256S or source_family not in FAMILIES
            or direction not in ("prediction", "generation") or split not in ("train", "validation", "test")
            or type(limit) is not int or not 1 <= limit <= 256):
        raise ValueError("pinned admitted source, direction, split and bounded limit required")
    requested = None if task_ids is None else tuple(task_ids)
    if requested is not None and (not requested or len(requested) > limit
            or len(set(requested)) != len(requested) or any(not _hex(t) for t in requested)):
        raise ValueError("bounded unique exact task IDs required")
    root = Path(root).resolve()
    path = (root / manifest).resolve()
    if not path.is_relative_to(root):
        raise ValueError("master outside project")
    raw = path.read_bytes()
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError("master manifest integrity mismatch")
    master, sources, out, seen = json.loads(raw), None, [], set()
    task = FAMILIES[source_family] + direction + "_prior"
    for shard in master["shards"]:
        if (not shard["path"].startswith(BASE + "/admitted-source-tasks-v1/shards/")
                or shard.get("split") != split or shard.get("source_family") not in (None, source_family)):
            continue
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard)), use_threads=False).to_pylist()
        side = master["effective_group_lookup"].get(shard["path"])
        if side is None:
            raise ValueError("missing authoritative NCI group sidecar")
        groups_rows = pq.read_table(io.BytesIO(_verified_bytes(root, side)), use_threads=False).to_pylist()
        groups = {g["task_id"]: g["effective_groups_json"] for g in groups_rows}
        ids = [r["task_id"] for r in rows]
        if (len(rows) != shard["rows"] or len(groups_rows) != side["rows"]
                or len(groups) != len(groups_rows) or len(set(ids)) != len(ids) or set(ids) != set(groups)):
            raise ValueError("NCI shard/group identity allocation mismatch")
        for row in rows:
            if row["split"] != split:
                raise ValueError("NCI shard/row split mismatch")
            if (row["source_family"] != source_family or row["task_name"] != task
                    or (requested is not None and row["task_id"] not in requested)):
                continue
            if row["task_id"] in seen or _sha(row["native_payload"]) != row["native_payload_sha256"]:
                raise ValueError("duplicate task or native payload integrity mismatch")
            seen.add(row["task_id"])
            ref = json.loads(row["native_payload"])["audit"]["source_ref"]
            if ref["native_source"] != SOURCE_REF:
                raise ValueError("unpinned native original NCI source")
            if sources is None:
                sources = _NativeSources(root)
            structure = sources.structure(ref["structure"])
            outcomes = [sources.outcome(rr) for rr in ref.get("outcomes", [ref.get("outcome")])]
            link = sources.row(ref["context_link"]) if "context_link" in ref else None
            view = nci_aids_view(row, structure, outcomes, context_link=link,
                effective_groups=json.loads(groups[row["task_id"]]), corpus_binding=dict(
                    manifest_sha256=expected_manifest_sha256, shard_sha256=shard["sha256"],
                    groups_sha256=side["sha256"], split=split))
            p = view.provenance()
            p.update(raw_source_readback_verified=True, raw_readback_receipts=list(sources.receipts.values()))
            pb = _canonical(p)
            out.append(replace(view, provenance_bytes=pb,
                binding_sha256=_binding(view.input_bytes, view.target_bytes, pb)))
            if len(out) == limit:
                return tuple(out)
    if requested is not None and set(requested) != seen:
        raise ValueError("requested task IDs absent from admitted allocation")
    if not out:
        raise ValueError("no admitted matching original NCI tasks")
    return tuple(out)
