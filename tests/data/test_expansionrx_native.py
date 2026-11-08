"""Synthetic upstream task-wire tests; native fixtures are separate source coverage."""
from dataclasses import replace
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
from mora3.data.admet import (AdmetObservation,NativeCell,NativeEntity,AssayContext,DesiredProperty,canonical)
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
def sha(x):return hashlib.sha256(x).hexdigest()
@pytest.fixture(scope="module")
def api():
 p=Path(os.environ.get("MORA3_EXPANSIONRX_MODULE","src/mora3/data/expansionrx_native.py"))
 if not p.is_file():pytest.fail("source-bound ExpansionRx task reader is absent")
 spec=importlib.util.spec_from_file_location("mora3.data.expansionrx_native",p);m=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def wire(direction="entity_to_measurement",literal=">90",endpoint="MPPB",parent="source_train",representation="C[C@H](O)F |&1:1|"):
 sys.path.insert(0,str(ROOT/"scripts"))
 from data_intake_admet_observations import observation,native_cell
 from data_intake_admet_endpoint_review import review_expansion_endpoint,_SCHEMA
 from data_intake_openadmet_admission import admitted_observation,source_task_views
 plan=json.loads((ROOT/"outputs/data-intake/20261101-ADMET-corpus-v1/ExpansionRx-component-execution-plan-v1.json").read_text())
 jm=json.loads((ROOT/plan["inputs"]["joint_endpoint_review"]["path"]).read_text())
 graph=dict(source_family="ExpansionRx",native_entity_id="synthetic-entity",native_row_index=2,native_parent_protocol=parent,
  source_graph_scope_json=json.dumps(dict(native_representation=representation,native_predictor_identity_candidate=True,legacy_plain_SMILES_exact_comparison_eligible=True)),
  known_current_heldout_group_matches_json="[]",frozen_pilot_reservation_matches_json="[]",oldtrain_group_matches_json="[]",
  coarse_graph_groups_json=json.dumps(["molecule:synthetic","scaffold:synthetic"]))
 joint={k:graph[k] for k in ("source_family","native_entity_id","native_row_index","native_parent_protocol")}
 joint.update(any_benchmark_heldout_parent_source_families_json="[]" if parent=="source_train" else '["ExpansionRx"]',
  candidate_train_after_joint_known_source_guards=parent=="source_train")
 original=observation(entity=dict(molecule_class="small_molecule",representation_type="native_SMILES",representation=representation),
  endpoint=dict(endpoint_key="ExpansionRx:"+endpoint,source_endpoint=endpoint,unit=None,unit_basis="unresolved"),
  cell=native_cell(literal),known_context={},source_ref=dict(native_entity_namespace="ExpansionRx",native_entity_id="synthetic-entity",
   native_row_index=2,source=dict(path="synthetic/source.csv",bytes=20,sha256="1"*64)),
  native_source_record={endpoint:literal,"SMILES":representation,"unknown_source_column":"native-audit"},
  evidence_role="source_curated_measurement",
  source_split=dict(parent_protocol={"source_train":"native_train","source_test":"native_test","source_unallocated":"raw_only_unallocated"}[parent]),
  admission=dict(source_rights=True,exact_identity=False,endpoint_definition=False,task_units_or_declared_native_coordinate=False,split_group_review=False))
 review=review_expansion_endpoint(endpoint,card_text="\n".join(x["line"] for x in _SCHEMA.values()),card_ref={"path":"synthetic/card"})
 qualified,reason=admitted_observation(original,graph,joint,review)
 if qualified is None:return dict(source=AdmetObservation.from_native(canonical(original)),graph=graph,joint=joint,endpoint_review=review)
 task=source_task_views(qualified)[direction];payload=task.canonical_bytes()
 oid=sha(canonical(["1"*64,2,endpoint]))
 groups=sorted(set(json.loads(graph["coarse_graph_groups_json"]))|{"source_compound:ExpansionRx:synthetic-entity","source_native_axis_cell:"+oid})
 row=dict(task_id=sha(canonical(["ExpansionRx-source-native-ADMET-v1",oid,direction])),record_id=oid,
  source_family="ExpansionRx-source-native-ADMET-readouts",source_revision="openadmet-expansionrx-challenge-data@6b898ccc43d10d25b230fb09e22a6e30c30022b5",
  source_sha256="1"*64,task_name="source_native_ADMET_"+direction,native_schema="mora3.CorpusTask/v1",
  native_payload=payload,native_payload_sha256=sha(payload),split=qualified["admission"]["allocation"],split_policy=reason,
  group_ids_json=json.dumps(groups),evidence_class="source_curated_assay_readout;notindependentexperimentalreplicateorclinicalvalidation",
  admitted_for_task=True,model_adapter_supported=False,direction=direction,full_axis_values=1,
  admission_receipt_json=json.dumps(dict(component_task_scoped=True,global_master_admission=False,
   source_rights=True,native_schema_units_or_coordinate_review=jm["endpoint_registry"],
   known_group_guards=plan["inputs"]["chemical_group_review"],global_novelty_or_study_disjointness_proven=False,
   original_source_parent_protocol=original["source_split"],independent_experimental_measurement_count=None)),
  missing_context_json=json.dumps(json.loads(payload)["missing"]),
  condition_context_json=json.dumps(dict(physical_unit_known=qualified["endpoint"]["unit_basis"]=="official_source_schema",
   native_coordinate_only=review["native_coordinate_only"],no_unit_conversion=True,source_missing_is_unknown_not_negative=True,
   original_official_test_not_train=True,model_training=False)))
 return dict(row=row,source=AdmetObservation.from_native(canonical(original)),graph=graph,joint=joint,endpoint_review=review,
  effective_groups=groups,binding=dict(plan_sha256="49c3e4321b804e2e3f915d1428ee65e1d31baace13463fbd5b8c45494e7115d1",
  task_shard_sha256="2"*64,groups_sha256="3"*64))
def project(api,w=None):return api.expansionrx_task_candidate(**(wire() if w is None else w))

@pytest.mark.parametrize("direction",("entity_to_measurement","measurement_to_entity"))
@pytest.mark.parametrize("literal",("0.00100","<0.02",">90"))
def test_upstream_exact_censored_wire_preserves_Decimal_roles_and_full_enhanced_stereo(api,direction,literal):
 w=wire(direction,literal);v=project(api,w)
 assert isinstance(v.observation,AdmetObservation)
 assert v.observation.cell.native_literal==literal
 assert v.observation.entity.as_mapping()["representation"]=="C[C@H](O)F |&1:1|"
 assert v.direction==direction and v.provenance()["master_admitted"] is False
 inputs=v.prediction_inputs()
 if direction=="entity_to_measurement":
  assert "native_entity" in inputs and "native_measurement" not in inputs
 else:
  assert "native_entity" not in inputs and "observed_native_measurement" in inputs
 assert "native-audit" not in canonical(inputs).decode()

@pytest.mark.parametrize("endpoint",("LogD","Caco-2 Permeability Efflux","RLM CLint","MPPB"))
def test_endpoint_coordinate_vs_schema_unit_and_raw_extra_roles_are_distinct(api,endpoint):
 v=project(api,wire(endpoint=endpoint));e=v.observation.native()["endpoint"]
 assert e["unit"]==({"RLM CLint":"mL/min/kg","MPPB":"% Unbound"}.get(endpoint))
 assert (e["unit_basis"]=="unresolved")== (endpoint in ("LogD","Caco-2 Permeability Efflux"))
 assert "route" not in v.observation.context.as_mapping()
 assert v.definition().objective_kind=="unresolved"
 assert v.definition().applicability_for("small_molecule")=="unknown"

@pytest.mark.parametrize("field",("direction","task_name","source_family","split","full_axis_values","admitted_for_task","model_adapter_supported","record_id","task_id","source_sha256","split_policy"))
def test_source_task_envelope_types_direction_official_split_and_ids_refuse(api,field):
 w=wire();w["row"][field]={"direction":"inverse_design","task_name":"other","source_family":"ASAP",
 "split":"validation","full_axis_values":True,"admitted_for_task":1,"model_adapter_supported":0,
 "record_id":"0"*64,"task_id":"0"*64,"source_sha256":"0"*64,"split_policy":"random"}[field]
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("mutation",("target","condition_target","audit_type","missing_context","native_units","other_endpoint","entity","direction"))
def test_rehashed_task_payload_does_not_bypass_native_source_and_role_binding(api,mutation):
 w=wire();d=json.loads(w["row"]["native_payload"])
 if mutation=="target":d["targets"]["native_measurement"]["relation"]="="
 elif mutation=="condition_target":d["conditions"]["native_measurement"]=d["targets"]["native_measurement"]
 elif mutation=="audit_type":d["audit"]["directions_reuse_one_source_observation"]=1
 elif mutation=="missing_context":d["missing"]=[]
 elif mutation=="native_units":d["conditions"]["endpoint"]["unit"]="fraction bound"
 elif mutation=="other_endpoint":d["conditions"]["other_endpoint_values"]={"HLM":"secret"}
 elif mutation=="entity":d["conditions"]["native_entity"]["representation"]="CC"
 else:d["task"]="source_native_ADMET_measurement_to_entity"
 w["row"]["native_payload"]=canonical(d);w["row"]["native_payload_sha256"]=sha(w["row"]["native_payload"])
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("field",("component_task_scoped","global_master_admission","source_rights","independent_experimental_measurement_count"))
def test_receipt_rehashed_numeric_bool_and_independent_count_aliases_refuse(api,field):
 w=wire();d=json.loads(w["row"]["admission_receipt_json"])
 d[field]=0 if field=="global_master_admission" else 1
 w["row"]["admission_receipt_json"]=json.dumps(d)
 with pytest.raises(ValueError):project(api,w)

@pytest.mark.parametrize("mutation,reason",(("missing","missing_or_unparsed_source_native_axis_cell"),
 ("raw_only","raw_only_original_source_parent_unallocated"),("test_exposed","source_test_known_prior_train_graph_group_exposure"),
 ("train_heldout","known_official_source_heldout_parent_graph_group"),("protected","existing_protected_group_or_frozen_reservation"),
 ("ASAP","ASAP_source_units_and_tabular_transform_unresolved")))
def test_dispositions_never_invent_tasks_for_missing_raw_only_ASAP_or_known_holdout_exposure(api,mutation,reason):
 w=wire(parent="source_test" if mutation=="test_exposed" else "source_train")
 if mutation=="missing":
  d=w["source"].native();d["cell"]=NativeCell.parse("").as_mapping();d["native_source_record"]["MPPB"]=""
  w["source"]=AdmetObservation.from_native(canonical(d))
 elif mutation=="raw_only":
  w=wire(parent="source_unallocated")
 elif mutation=="test_exposed":w["graph"]["oldtrain_group_matches_json"]='["scaffold:old"]'
 elif mutation=="train_heldout":w["joint"]["any_benchmark_heldout_parent_source_families_json"]='["ASAP"]'
 elif mutation=="protected":w["graph"]["known_current_heldout_group_matches_json"]='["scaffold:protected"]'
 else:
  d=w["source"].native();d["source_ref"]["native_entity_namespace"]="ASAP";d["source_split"]["parent_protocol"]="Train"
  w["source"]=AdmetObservation.from_native(canonical(d));w["graph"]["source_family"]=w["joint"]["source_family"]="ASAP"
 result=api.expansionrx_disposition(w["source"],w["graph"],w["joint"],w["endpoint_review"])
 assert result.reason==reason and result.serialized_views==0
 if "row" in w:
  with pytest.raises(ValueError):project(api,w)

def test_original_test_remains_test_even_with_fake_train_flag_and_no_new_validation(api):
 w=wire(parent="source_test");v=project(api,w)
 assert v.provenance()["envelope"]["split"]=="test"
 w["joint"]["candidate_train_after_joint_known_source_guards"]=True
 assert api.expansionrx_disposition(w["source"],w["graph"],w["joint"],w["endpoint_review"]).allocation=="test"
 # Source review audit changed; exact replay rejects even though allocation stays test.
 with pytest.raises(ValueError):project(api,w)
 with pytest.raises(ValueError):v.require_model_use("training",proof={"approved":True})

def test_default_predictors_exclude_hidden_targets_masks_unknown_fields_ids_and_desired_provenance(api):
 a=project(api,wire(literal="1"));b=project(api,wire(literal=">999"))
 assert a.prediction_inputs()==b.prediction_inputs()
 assert set(a.prediction_inputs())=={"role","native_entity","endpoint","assay_context"}
 # Explicit desired requests never auto-fill from the observed source cell.
 req=DesiredProperty(a.definition().key,NativeCell.parse("<2"),a.observation.context,canonical({"caller_id":"audit"}))
 desired=a.desired_inputs(req)
 assert desired["role"]=="desired_ADMET_properties" and req.observed_label_count==0
 assert "caller_id" not in canonical(desired).decode()
 assert desired!=a.prediction_inputs()
 with pytest.raises(ValueError):a.desired_inputs(None)

def test_inverse_observed_association_is_not_aliased_to_explicit_desired_or_causal_design(api):
 v=project(api,wire("measurement_to_entity"));query=v.prediction_inputs()
 assert query["role"]=="source_native_ADMET_observational_property_to_entity_query"
 assert query["observed_native_measurement"]["relation"]==">"
 assert "native_entity" not in query
 assert v.provenance()["causal_inverse_design_validated"] is False

@pytest.mark.parametrize("purpose",("training","evaluation","loss","teacher_forcing","generation","split_use"))
def test_eligible_task_is_not_positive_supervision_or_model_qualification(api,purpose):
 v=project(api)
 for proof in (None,True,{"verified":True,"master_version":23}):
  with pytest.raises(ValueError,match="master|qualification|proof"):v.require_model_use(purpose,proof=proof)
 with pytest.raises(ValueError):v.positive_loss_projection()
 with pytest.raises(ValueError):v.as_model_view()

@pytest.mark.parametrize("indices",((),(0,0),(True,),(-1,),[0],tuple(range(17))))
def test_task_selector_exact_unique_bounded_types(api,indices):
 with pytest.raises(ValueError):api.ExpansionRxSelector("tasks",indices)


@pytest.mark.parametrize("index",(2.0,True))
def test_graph_and_joint_source_ordinals_cannot_alias_native_integer_types(api,index):
 w=wire();w["graph"]["native_row_index"]=w["joint"]["native_row_index"]=index
 with pytest.raises(ValueError):api.expansionrx_disposition(w["source"],w["graph"],w["joint"],w["endpoint_review"])


def test_inverse_hidden_native_entity_and_enhanced_stereo_target_do_not_enter_observed_property_inputs(api):
 a=project(api,wire("measurement_to_entity"))
 b=project(api,wire("measurement_to_entity",representation="N[C@@H](F)Cl"))
 assert a.observation.entity.native_bytes!=b.observation.entity.native_bytes
 assert a.prediction_inputs()==b.prediction_inputs()
 assert a.native_payload!=b.native_payload

@pytest.mark.parametrize("field",("physical_unit_known","native_coordinate_only","no_unit_conversion","source_missing_is_unknown_not_negative","original_official_test_not_train","model_training"))
def test_native_task_context_bool_int_aliases_refuse(api,field):
 w=wire();context=json.loads(w["row"]["condition_context_json"])
 context[field]=int(context[field]);w["row"]["condition_context_json"]=json.dumps(context)
 with pytest.raises(ValueError):project(api,w)

def test_native_groups_and_target_cell_cannot_allocate_duplicate_task_identity(api):
 w=wire();w["effective_groups"].append(w["effective_groups"][0])
 with pytest.raises(ValueError):project(api,w)


FIXTURE=Path("outputs/data-intake/20261101-ADMET-corpus-v1/ExpansionRx-closed-native-reader-fixture-v1/manifest.json")
FIXTURE_SHA="3be3313d247f5fa21fcabaa26faee1afb89c775875c190eb7675166c3e45f3fd"
@pytest.fixture(scope="module")
def real_source(api):
 if not (ROOT/FIXTURE).is_file():pytest.skip("constructor closed ExpansionRx task fixture unavailable")
 return api.ExpansionRxNativeTaskSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=FIXTURE_SHA)
@pytest.fixture(scope="module")
def real(api,real_source):return real_source.read(api.ExpansionRxSelector("tasks",tuple(range(12))))

def test_actual_both_directions_exact_censored_native_units_coordinates_and_RLM_parent_roles(api,real_source,real):
 assert len(real)==12 and {v.direction for v in real}=={"entity_to_measurement","measurement_to_entity"}
 assert {v.observation.cell.status for v in real}=={"exact","censored_bound"}
 assert any(v.provenance()["envelope"]["split"]=="test" for v in real)
 endpoints={v.observation.native()["endpoint"]["source_endpoint"]:v for v in real}
 assert {"LogD","Caco-2 Permeability Efflux","RLM CLint"}<=set(endpoints)
 for name in ("LogD","Caco-2 Permeability Efflux"):
  assert endpoints[name].definition().unit is None
  assert endpoints[name].definition().unit_basis=="unresolved"
 assert endpoints["RLM CLint"].definition().unit=="mL/min/kg"
 assert endpoints["RLM CLint"].observation.native()["source_split"]["parent_protocol"]!="raw_only_unallocated"
 for v in real:
  assert real_source.verify(v)==v
  assert v.provenance()["source_native_bytes_verified_by_existing_candidate_reader"] is True
  assert v.provenance()["master_admitted"] is False
  assert v.provenance()["model_endpoint_qualified"] is False
  assert v.provenance()["independent_experimental_measurement_count"] is None

def test_actual_every_available_source_axis_disposition_retained_including_missing_ASAP_and_heldouts(api,real_source):
 values=real_source.read(api.ExpansionRxSelector("dispositions",tuple(range(11))))
 assert len(values)==11
 reasons={x.reason for x in values}
 assert {"missing_or_unparsed_source_native_axis_cell","known_official_source_heldout_parent_graph_group",
  "source_test_known_prior_train_graph_group_exposure","raw_only_original_source_parent_unallocated",
  "ASAP_source_units_and_tabular_transform_unresolved"}<=reasons
 assert sum(x.serialized_views for x in values)==12
 assert any(x.status=="missing" and x.serialized_views==0 for x in values)
 assert real_source.gates()["whole_component_dispositions_inherited"]==78980
 for v in values:
  assert real_source.verify(v)==v
  with pytest.raises(ValueError):v.require_model_use("loss",proof=True)

@pytest.mark.parametrize("purpose",("training","evaluation","loss","teacher_forcing","generation","split_use"))
def test_actual_component_scoped_tasks_do_not_bypass_master_or_model_endpoint_qualification(real,purpose):
 for v in real:
  with pytest.raises(ValueError):v.require_model_use(purpose,proof={"approved":True,"component_sha256":"36df848050985bcbc06f1064574499bc8c24ee681d256400e99835ec93cc41fb"})
  with pytest.raises(ValueError):v.positive_loss_projection()
  with pytest.raises(ValueError):v.as_model_view()

@pytest.mark.parametrize("field",("envelope_type","binding","source_ref","selector"))
def test_actual_hidden_receipt_source_audit_and_selector_mutations_do_not_change_predictors_but_replay_refuses(real_source,real,field):
 v=real[0];p=v.provenance()
 if field=="envelope_type":
  receipt=json.loads(p["envelope"]["admission_receipt_json"]);receipt["global_master_admission"]=0
  p["envelope"]["admission_receipt_json"]=json.dumps(receipt)
 elif field=="binding":p["binding"]["task_shard_sha256"]="0"*64
 elif field=="source_ref":p["closed_fixture_source_case"]["source_disposition"]["source_entity"]="hidden"
 else:p["reader_binding"]["case_index"]=True
 changed=replace(v,provenance_bytes=canonical(p))
 assert changed.prediction_inputs()==v.prediction_inputs()
 with pytest.raises(ValueError):real_source.verify(changed)

@pytest.mark.parametrize("artifact",("fixture","component","plan","approval","native_projection","group_review","endpoint_review",
 "endpoint_registry","group_rows","joint_rows","native_copies","task0","task11","shard0","groups0"))
def test_actual_pinned_source_task_and_review_artifact_substitution_refuses(api,real_source,artifact,monkeypatch):
 target=(ROOT/real_source.source_references()[artifact]["path"]).resolve();old=Path.read_bytes;touched=[]
 def tamper(path):
  raw=old(path)
  if path.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",tamper)
 with pytest.raises(ValueError):
  s=api.ExpansionRxNativeTaskSource(root=ROOT)
  s.read(api.ExpansionRxSelector("tasks",(0,11)))
 assert touched

def test_runtime_safe_formats_no_SQLite_taskindex_opaque_or_model_reads(api,real_source,monkeypatch):
 old=Path.read_bytes;seen=[]
 def safe(path):
  assert path.suffix.lower() in (".json",".parquet",".csv")
  assert ".sqlite" not in str(path) and ".pkl" not in str(path)
  seen.append(str(path));return old(path)
 monkeypatch.setattr(Path,"read_bytes",safe)
 real_source.read(api.ExpansionRxSelector("tasks",(0,1)))
 real_source.read(api.ExpansionRxSelector("dispositions",(6,9,10)))
 assert seen

def test_unknown_task_and_source_disposition_selector_and_fake_future_manifest_refuse(api,real_source,tmp_path):
 with pytest.raises(ValueError):real_source.read(api.ExpansionRxSelector("tasks",(12,)))
 with pytest.raises(ValueError):real_source.read(api.ExpansionRxSelector("dispositions",(11,)))
 path=tmp_path/"fake.json";path.write_text('{"master_union_admission":true}')
 with pytest.raises(ValueError):api.ExpansionRxNativeTaskSource(root=tmp_path,manifest="fake.json")
 with pytest.raises(ValueError):api.ExpansionRxNativeTaskSource(root=ROOT,expected_manifest_sha256=sha(path.read_bytes()))
 assert real_source.gates()["master_admitted"] is False
 assert real_source.gates()["model_endpoint_qualified"] is False


@pytest.mark.parametrize("endpoint",("HLM","MLM","KSOL","LogD","MDR1-MDCKII"))
def test_synthetic_all_five_ASAP_tabular_transforms_stay_closed_despite_assay_context(api,endpoint):
 w=wire();d=w["source"].native();old=d["endpoint"]["source_endpoint"]
 d["source_ref"]["native_entity_namespace"]="ASAP";d["source_split"]["parent_protocol"]="Train"
 d["endpoint"]["source_endpoint"]=endpoint;d["endpoint"]["endpoint_key"]="ASAP:"+endpoint
 d["native_source_record"][endpoint]=d["native_source_record"][old]
 d["known_context"]["assay_system"]="documented assay; tabular transform unresolved"
 w["source"]=AdmetObservation.from_native(canonical(d));w["graph"]["source_family"]=w["joint"]["source_family"]="ASAP"
 result=api.expansionrx_disposition(w["source"],w["graph"],w["joint"],w["endpoint_review"])
 assert result.serialized_views==0 and result.allocation is None

def test_actual_registry_keeps_all_five_ASAP_transform_gates_explicit(real_source):
 assert set(real_source.gates()["ASAP_closed_endpoints"])=={"HLM","MLM","KSOL","LogD","MDR1-MDCKII"}
