"""ADMET contracts: synthetic biology fixtures plus bounded real candidate gates."""
import copy
from decimal import Decimal
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest

ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
BASE=Path("outputs/data-intake/20261101-ADMET-corpus-v1/native-observations-v2-source-cell-counts")
SHA="cd2ff3bde8dc01f92fdb6a88ff46d09a10d4d058f07989584f2912fe8a2f79f0"


def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()


def api():
    path=os.environ.get("MORA3_ADMET_DATA_MODULE")
    if path:
        if not Path(path).is_file():pytest.fail("typed ADMET contracts not implemented")
        spec=importlib.util.spec_from_file_location("mora3.data.admet",path)
        m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
    try:return importlib.import_module("mora3.data.admet")
    except ModuleNotFoundError:pytest.fail("typed ADMET contracts not implemented")


def record(mod,molecule_class="peptide",literal=" <= 1.2300e-7 ",role="source_curated_measurement"):
    rep={"small_molecule":"C[C@H](O)N","peptide":"aCDc","protein":"EVQLV;DIQMT","RNA":"AcGuTN","DNA":"aCTtN"}[molecule_class]
    entity=dict(molecule_class=molecule_class,representation_type="source_native_chemical_or_polymer_notation",representation=rep,
        native_modifications=[dict(site=2,native_chemistry="source D-residue or modifiedbase;not inferred",stereochemistry="source literal")],
        construct=dict(topology="source cyclic/conjugated",chains=["heavy","light"],terminal_groups="source cap"),
        source_identity_missingness=["unreported formulation"])
    endpoint=dict(endpoint_key="SYNTHETIC-STUDY:native_half_life",source_endpoint="native_half_life",
        description="synthetic endpoint fixture only",readout="half-life in declared source matrix",unit="min",unit_basis="source_literal",
        scale="native",measurement_type="synthetic assay fixture",assay_protocol_context={"source_id":"SECRET-PROTOCOL","citation":"audit-only"})
    return dict(schema="mora3.ADMETNativeObservation/v1",entity=entity,endpoint=endpoint,cell=mod.NativeCell.parse(literal).as_mapping(),
        known_context=dict(species="source human",matrix="source serum",dose={"value":"10.0","unit":"nM","unit_basis":"source_literal"},
            route="source route",formulation="source carrier"),source_ref={"source_id":"SECRET-SOURCE","native_row_index":0},
        native_source_record={"native_half_life":literal,"source_id":"SECRET-SOURCE","unprojected_source_field":{"kept":[None,"raw"]}},
        evidence_role=role,source_split={"native_export":"test","groups":["SECRET-GROUP"]},
        admission=dict(source_rights=True,exact_identity=False,endpoint_definition=False,
            task_units_or_declared_native_coordinate=False,split_group_review=False),
        absence_is_not_negative=True,native_fields_not_projected_remain_in_source_record=True,
        inferred_unit_conversions=[],source_native_axis_cell_multiplicity=1,independent_experimental_measurement_count=None)


def registry(mod,raw,kind="regression",applicable="applicable"):
    mapping={c:"unknown" for c in mod.MOLECULE_CLASSES};mapping[raw["entity"]["molecule_class"]]=applicable
    endpoint=mod.EndpointDefinition.from_native(raw["endpoint"],objective_kind=kind,applicability=mapping,
        binary_readout="native source positive class explicitly declared" if kind=="binary" else None)
    return mod.EndpointRegistry((endpoint,))


@pytest.mark.parametrize("literal,relation,decimal,status",[
    ("  <= 1.2300e-7 ","<=","1.2300e-7","censored_bound"),("≥-2.00",">=","-2.00","censored_bound"),
    ("= 0.000","=","0.000","exact"),("9007199254740993","=","9007199254740993","exact"),
    ("",None,None,"missing"),(None,None,None,"missing"),("unknown qualifier",None,None,"unparsed")])
def test_native_literal_decimal_and_censor_are_not_coerced(literal,relation,decimal,status):
    mod=api();cell=mod.NativeCell.parse(literal)
    assert cell.native_literal==literal and cell.relation==relation and cell.decimal_literal==decimal and cell.status==status
    assert mod.NativeCell.from_mapping(cell.as_mapping()).as_mapping()==cell.as_mapping()
    if decimal is not None:assert cell.decimal_value()==Decimal(decimal)
    if literal=="9007199254740993":assert cell.decimal_value()==Decimal(2**53+1)


def test_explicit_interval_retains_source_bounds_and_inequality_roles():
    mod=api();a=mod.NativeCell.interval("source interval [1.00,2.000)",lower="1.00",upper="2.000",lower_inclusive=True,upper_inclusive=False)
    assert a.censor_kind=="interval" and a.lower_decimal_literal=="1.00" and a.upper_decimal_literal=="2.000"
    assert a.as_mapping()["upper_inclusive"] is False
    with pytest.raises(ValueError):mod.NativeCell.interval("invalid",lower="2",upper="1")
    with pytest.raises(ValueError):mod.NativeCell.interval("zero width",lower="1",upper="1")


@pytest.mark.parametrize("molecule_class",["small_molecule","peptide","protein","RNA","DNA"])
def test_five_class_identity_context_and_unknown_native_fields_are_lossless(molecule_class):
    mod=api();raw=record(mod,molecule_class);obs=mod.AdmetObservation.from_native(canonical(raw),applicability="unknown")
    assert obs.native()==raw and obs.entity.as_mapping()==raw["entity"]
    assert obs.context.as_mapping()==raw["known_context"]
    assert obs.entity.as_mapping()["representation"]==raw["entity"]["representation"]
    assert obs.applicability=="unknown" and obs.training_eligibility=="requires_verified_native_task_admission"
    assert obs.native()["independent_experimental_measurement_count"] is None
    assert obs.native()["source_native_axis_cell_multiplicity"]==1
    assert obs.entity.canonical_bytes()!=mod.NativeEntity.from_mapping({**raw["entity"],"native_modifications":None}).canonical_bytes()


def test_forward_prediction_excludes_hidden_cell_masks_IDs_split_audit_and_protocol_refs():
    mod=api();raw=record(mod);reg=registry(mod,raw)
    a=mod.admet_prediction_view(mod.AdmetObservation.from_native(canonical(raw)),reg)
    changed=copy.deepcopy(raw);changed["cell"]=mod.NativeCell.parse(">900.0").as_mapping()
    changed["native_source_record"]["native_half_life"]=">900.0"
    changed["source_ref"]={"source_id":"OTHER"};changed["source_split"]={"native_export":"train"}
    changed["native_source_record"]["other_unknown"]="different";changed["admission"]={k:True for k in changed["admission"]}
    b=mod.admet_prediction_view(mod.AdmetObservation.from_native(canonical(changed)),reg)
    assert a.input_bytes==b.input_bytes and a.target_bytes!=b.target_bytes
    inputs=a.inputs()
    assert set(inputs)=={"role","native_entity","endpoint","assay_context"}
    assert "SECRET" not in json.dumps(inputs) and "assay_protocol_context" not in inputs["endpoint"]
    assert "endpoint_key" not in inputs["endpoint"]
    assert "native_measurement" not in inputs and "numeric_loss_eligible" not in inputs
    record_graph,task=mod.admet_condition_graph(inputs)
    assert record_graph.nodes[0].kind=="molecular-endpoint-query"
    assert task is not None


@pytest.mark.parametrize("field",["observation_present","numeric_loss_eligible"])
def test_numeric_boolean_source_cell_aliases_rejected(field):
    mod=api();raw=record(mod);raw["cell"][field]=1
    with pytest.raises(ValueError):mod.AdmetObservation.from_native(canonical(raw))


@pytest.mark.parametrize("mutation",["cell_literal","unit","cell_count_bool","experiment_count","gate_alias","context_target","nested_context_ID","entity_ID","construct_target"])
def test_native_schema_type_units_axis_and_target_provenance_injections_rejected(mutation):
    mod=api();raw=record(mod)
    if mutation=="cell_literal":raw["cell"]["decimal_literal"]="999"
    elif mutation=="unit":raw["endpoint"]["unit_basis"]="unresolved"
    elif mutation=="cell_count_bool":raw["source_native_axis_cell_multiplicity"]=True
    elif mutation=="experiment_count":raw["independent_experimental_measurement_count"]=1
    elif mutation=="gate_alias":raw["admission"]["source_rights"]=1
    elif mutation=="context_target":raw["known_context"]["target_mask"]=[True]
    elif mutation=="nested_context_ID":raw["known_context"]["dose"]["source_id"]="injected"
    elif mutation=="entity_ID":raw["entity"]["source_id"]="injected"
    else:raw["entity"]["construct"]["target_value"]="injected"
    with pytest.raises(ValueError):mod.AdmetObservation.from_native(canonical(raw))


@pytest.mark.parametrize("role",["computed","predicted","desired","model_generated"])
def test_nonexperimental_roles_never_become_measured_labels(role):
    mod=api();obs=mod.AdmetObservation.from_native(canonical(record(mod,role=role)),applicability="applicable")
    assert obs.experimental_measurement_candidate is False
    with pytest.raises(ValueError):obs.require_source_admission(proof=True)


def test_all_true_admission_flags_are_not_a_source_verifier():
    mod=api();raw=record(mod);raw["admission"]={k:True for k in raw["admission"]}
    obs=mod.AdmetObservation.from_native(canonical(raw),applicability="applicable")
    for proof in [None,True,raw["admission"],{"verified":True,"manifest_sha256":"a"*64}]:
        with pytest.raises(ValueError,match="admission|reader|verified"):obs.require_source_admission(proof=proof)


def test_missing_unknown_and_inapplicable_stay_distinct():
    mod=api();raw=record(mod,literal="")
    a=mod.AdmetObservation.from_native(canonical(raw),applicability="unknown")
    b=mod.AdmetObservation.from_native(canonical(raw),applicability="inapplicable")
    assert a.cell.status==b.cell.status=="missing" and a.applicability!=b.applicability
    assert a.experimental_measurement_candidate is b.experimental_measurement_candidate is False
    reg=registry(mod,raw,applicable="inapplicable")
    assert reg.get(raw["endpoint"]["endpoint_key"]).applicability_for("peptide")=="inapplicable"
    with pytest.raises(ValueError):mod.EndpointRegistry((reg.definitions[0],reg.definitions[0]))


def test_desired_constraints_have_distinct_role_and_no_observed_label_or_target_entity():
    mod=api();raw=record(mod);reg=registry(mod,raw)
    desired=mod.DesiredProperty(raw["endpoint"]["endpoint_key"],mod.NativeCell.parse(">= 120.0"),
        mod.AssayContext.from_mapping({"matrix":"source serum"}),canonical({"request_id":"SECRET-REQUEST"}))
    inputs=mod.desired_admet_inputs("peptide",(desired,),reg)
    assert inputs["role"]=="desired_ADMET_properties" and "native_entity" not in inputs
    assert "SECRET" not in json.dumps(inputs) and desired.observed_label_count==0
    assert inputs["desired_properties"][0]["native_constraint"]["relation"]==">="
    with pytest.raises(ValueError):mod.desired_admet_inputs("peptide",(mod.AdmetObservation.from_native(canonical(raw)),),reg)


def test_real_qualified_v2_candidate_readback_is_not_admission():
    if not (ROOT/BASE/"manifest.json").is_file():pytest.skip("qualified native ADMET candidate source unavailable")
    mod=api()
    obs=mod.read_admet_native_candidates(BASE/"manifest.json",root=ROOT,expected_manifest_sha256=SHA,
        shard_paths=[str(BASE/"observations/0000.parquet")],limit=3)
    assert len(obs)==3 and [x.cell.status for x in obs]==["missing","missing","exact"]
    for x in obs:
        assert x.native()["independent_experimental_measurement_count"] is None
        assert x.training_eligibility=="requires_verified_native_task_admission"
        with pytest.raises(ValueError):x.require_source_admission()
    with pytest.raises(ValueError):mod.read_admet_native_candidates(BASE/"manifest.json",root=ROOT,expected_manifest_sha256="0"*64,
        shard_paths=[str(BASE/"observations/0000.parquet")],limit=1)


@pytest.mark.parametrize("where",["endpoint","desired_property","desired_constraint"])
def test_public_graph_boundary_rejects_injected_target_and_source_metadata(where):
    mod=api();raw=record(mod);reg=registry(mod,raw)
    view=mod.admet_prediction_view(mod.AdmetObservation.from_native(canonical(raw)),reg)
    if where=="endpoint":
        inputs=view.inputs();inputs["endpoint"]["target_hash"]="SECRET-TARGET"
    else:
        desired=mod.DesiredProperty(raw["endpoint"]["endpoint_key"],mod.NativeCell.parse(">=120"),
            mod.AssayContext.from_mapping({}),canonical({}))
        inputs=mod.desired_admet_inputs("peptide",(desired,),reg)
        if where=="desired_property":inputs["desired_properties"][0]["source_id"]="SECRET"
        else:inputs["desired_properties"][0]["native_constraint"]["target_mask"]=[True]
    with pytest.raises(ValueError):mod.admet_condition_graph(inputs)


def test_source_specific_missing_lexicon_is_not_a_boolean_mask_grant():
    mod=api();cell=dict(native_literal="NA",status="missing",decimal_literal=None,relation=None,
        observation_present=False,numeric_loss_eligible=False,missing_lexicon_source=True)
    with pytest.raises(ValueError):mod.NativeCell.from_mapping(cell)

def test_prediction_view_is_immutable_canonical_and_cannot_promote_supervision_flag():
    mod=api();raw=record(mod);view=mod.admet_prediction_view(mod.AdmetObservation.from_native(canonical(raw)),registry(mod,raw))
    with pytest.raises(ValueError):
        mod.AdmetPredictionView(bytearray(view.input_bytes),view.target_bytes,view.provenance_bytes)
    bad=view.provenance();bad["model_supervision_supported"]=True
    with pytest.raises(ValueError):
        mod.AdmetPredictionView(view.input_bytes,view.target_bytes,canonical(bad))
    target=view.targets();target["native_measurement"]=mod.NativeCell.parse(">9").as_mapping()
    with pytest.raises(ValueError):
        mod.AdmetPredictionView(view.input_bytes,canonical(target),view.provenance_bytes)


def test_prediction_registry_endpoint_digests_require_canonical_SHA_types():
    mod=api();raw=record(mod);view=mod.admet_prediction_view(mod.AdmetObservation.from_native(canonical(raw)),registry(mod,raw))
    for key in ("registry_binding","endpoint_binding"):
        for value in (True,"not-a-digest","A"*64):
            p=view.provenance();p[key]=value
            with pytest.raises(ValueError):
                mod.AdmetPredictionView(view.input_bytes,view.target_bytes,canonical(p))
