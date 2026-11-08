"""Retrosynthesis native contracts: synthetic core fixtures, never source coverage."""
from dataclasses import replace
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def load_api():
 p=Path(os.environ.get("MORA3_RETRO_MODULE","src/mora3/data/retrosynthesis_native.py"))
 if not p.is_file():pytest.fail("native retrosynthesis candidate interface is not implemented")
 spec=importlib.util.spec_from_file_location("mora3.data.retrosynthesis_native",p);m=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
@pytest.fixture(scope="module")
def api():return load_api()
def mol(api,text="CCO",**kw):return api.ChemicalIdentity(text,**kw)
def component(api,text,role="precursor",amount=None):
 return api.ReactionComponent(mol(api,text),role,amount or api.NativeAmount(None,None,"absent"))
def reaction(api,precursors=None,**kw):
 return api.ReactionCandidate((component(api,"C[C@H](O)F","product"),),
  tuple(precursors or (component(api,"C[C@H](Br)F"),component(api,"[OH-]"))),
  b"[C:1][C@@H:2](O)F>>native-raw-source",b"{}",**kw)

def test_synthetic_native_map_and_stereo_serialization_preserved_but_maps_excluded_from_inputs(api):
 raw="[13CH3:7][C@@H:12]([OH:3])[F:29].[Na+:6]"
 m=mol(api,raw)
 assert m.native_text==raw
 assert m.product_input_text()=="[13CH3][C@@H]([OH])[F].[Na+]"
 assert ":12" not in m.product_input_text() and "@@" in m.product_input_text()
 assert m.native_bytes()==raw.encode()
 assert mol(api,"C[C@H](O)F").product_input_text()!=mol(api,"C[C@@H](O)F").product_input_text()

def test_synthetic_extended_or_nonchemical_sequence_formats_refuse_without_stereo_loss(api):
 m=mol(api,"C[C@H](O)F |&1:1|",syntax="CXSMILES")
 assert m.native_text.endswith("|")
 with pytest.raises(ValueError,match="extended|unsupported|stereo"):m.product_input_text()
 with pytest.raises(ValueError):mol(api,"ACDE",syntax="AMINO_ACID_SEQUENCE",molecule_class="peptide")

def test_synthetic_precursor_target_is_unordered_role_amount_multiset_not_one_string_order(api):
 a=component(api,"CCO");b=component(api,"O=C=O")
 x=reaction(api,(a,b,a));y=reaction(api,(b,a,a));z=reaction(api,(a,b))
 assert x.precursor_multiset_key()==y.precursor_multiset_key()
 assert x.precursor_multiset_key()!=z.precursor_multiset_key()
 assert x.native_serialization==b"[C:1][C@@H:2](O)F>>native-raw-source"
 # Native literal-map-renumbering proxy only; no canonical chemical-equivalence claim.
 assert mol(api,"[CH3:1][OH:2]").native_multiset_identity()==mol(api,"[CH3:9][OH:7]").native_multiset_identity()

def test_synthetic_precursor_stoichiometry_and_auxiliary_roles_preserved(api):
 a=component(api,"CCO",amount=api.NativeAmount("2.0","equivalents","present"))
 b=component(api,"CCO",amount=api.NativeAmount("1","equivalents","present"))
 assert reaction(api,(a,)).precursor_multiset_key()!=reaction(api,(b,)).precursor_multiset_key()
 assert a.amount.value=="2.0"
 with pytest.raises(ValueError):reaction(api,(component(api,"O","solvent"),))

@pytest.mark.parametrize("value,unit,presence",((True,"mol","present"),(1.0,"mol","present"),("1",None,"absent"),(None,None,"present")))
def test_synthetic_amount_presence_and_native_lexical_types_are_exact(api,value,unit,presence):
 with pytest.raises(ValueError):api.NativeAmount(value,unit,presence)

def test_synthetic_default_product_conditions_exclude_all_targets_context_gold_and_route_outcomes(api):
 r=reaction(api)
 before=r.prediction_inputs()
 changed=replace(r,precursors=(component(api,"CCCCCCCC"),),
  native_serialization=b"hidden-target-order",audit_bytes=canonical(dict(yield_value="90",failure=False,
  reaction_class="gold-class",route_solution=["hidden"],stock=True,source_id="audit")))
 assert before==changed.prediction_inputs()=={
  "query_role":"product_conditioned_chemical_retrosynthesis","products":[{"syntax":"SMILES","value":"C[C@H](O)F"}]}
 assert r.binding_sha256!=changed.binding_sha256

@pytest.mark.parametrize("name",("precursors","yield","failure","reaction_class","atom_mapping","route_solution","stock_label","target_length"))
def test_synthetic_gold_or_outcome_conditions_cannot_be_declared_known_inputs(api,name):
 with pytest.raises(ValueError):api.ReactionCondition(name,"hidden",None,"known_input","explicit_caller_known")

def test_synthetic_known_context_requires_explicit_role_and_desired_is_separate(api):
 c=api.ReactionCondition("temperature","80","degC","known_input","explicit_caller_known")
 r=reaction(api)
 assert r.prediction_inputs(known_conditions=(c,))["known_conditions"]==[c.as_mapping()]
 hidden=api.ReactionCondition("temperature","80","degC","source_audit","not_inference_known")
 with pytest.raises(ValueError):r.prediction_inputs(known_conditions=(hidden,))
 desired=api.ReactionCondition("temperature","20","degC","desired","caller_desired")
 assert api.desired_reaction_inputs((desired,))["evidence_role"]=="caller_desired_conditions"

def test_synthetic_peptide_chemistry_roles_do_not_become_inverse_folding_or_coverage(api):
 p=api.PeptideSynthesisContext("SPPS",("Fmoc","tBu"),("noncanonical_residue","cyclization"),"synthetic_test")
 assert p.assembly_mode=="SPPS" and p.protecting_groups==("Fmoc","tBu")
 assert p.is_inverse_folding is False
 with pytest.raises(ValueError):api.PeptideSynthesisContext("inverse_folding",(),(),"synthetic_test")
 assert api.peptide_native_coverage()["eligible_native_records"]==0

def nodes(api):
 return (api.RouteMolecule("p",mol(api,"CCO")),api.RouteMolecule("a",mol(api,"C")),
  api.RouteMolecule("b",mol(api,"O")),api.RouteMolecule("c",mol(api,"CO")))
def stock(api,entries=("C","O","CO")):
 return api.StockSnapshot("benchmark_snapshot",tuple(entries),
  canonical(dict(path="synthetic-stock.txt",bytes=1,sha256="0"*64)),"native_literal_SMILES_membership")

def test_synthetic_AND_OR_DAG_alternatives_dependencies_and_stock_are_explicit(api):
 n=nodes(api)
 steps=(api.RouteStep("r1","p",("a","b")),api.RouteStep("r2","p",("c",)))
 route=api.RouteCandidate("p",n,steps,stock(api),b"{}","synthetic_test")
 assert route.alternatives("p")==("r1","r2")
 assert route.precursor_dependencies("r1")==("a","b")
 assert route.stock_termination_status()=="all_leaves_in_declared_snapshot"
 assert route.actual_purchasability_verified is False
 assert route.prediction_inputs()=={"query_role":"product_conditioned_route_query","product":{"syntax":"SMILES","value":"CCO"}}
 assert route.native_serialization==b"{}"

@pytest.mark.parametrize("mutation",("direct_cycle","indirect_cycle","missing_dependency","duplicate_node","duplicate_step","unreachable","empty_reaction","missing_root"))
def test_synthetic_route_cycles_dependency_and_allocation_errors_refuse(api,mutation):
 n=nodes(api);steps=(api.RouteStep("r1","p",("a","b")),);root="p"
 if mutation=="direct_cycle":steps=(api.RouteStep("r1","p",("p",)),)
 elif mutation=="indirect_cycle":steps+=(api.RouteStep("r2","a",("p",)),)
 elif mutation=="missing_dependency":steps=(api.RouteStep("r1","p",("missing",)),)
 elif mutation=="duplicate_node":n=n+(n[0],)
 elif mutation=="duplicate_step":steps=steps+steps
 elif mutation=="unreachable":n=n+(api.RouteMolecule("unused",mol(api,"N")),)
 elif mutation=="empty_reaction":
  with pytest.raises(ValueError):api.RouteStep("r1","p",())
  return
 else:root="missing"
 # c is intentionally absent from this single-alternative graph.
 n=tuple(v for v in n if v.node_id!="c")
 with pytest.raises(ValueError):
  api.RouteCandidate(root,n,steps,stock(api),b"{}","synthetic_test")

def test_synthetic_route_unknown_or_benchmark_stock_is_not_purchase_or_lab_feasibility(api):
 n=nodes(api);steps=(api.RouteStep("r1","p",("a","b","c")),)
 r=api.RouteCandidate("p",n,steps,None,b"{}","synthetic_test")
 assert r.stock_termination_status()=="stock_snapshot_unavailable"
 r=replace(r,stock_snapshot=stock(api,("C",)))
 assert r.stock_termination_status()=="leaf_outside_declared_snapshot"
 assert r.actual_purchasability_verified is False
 with pytest.raises(ValueError):api.StockSnapshot("currently_purchasable",("C",),b"{}","native_literal_SMILES_membership")

@pytest.mark.parametrize("purpose",("training","evaluation","reaction_loss","route_search","split_use","peptide_supervision"))
def test_candidate_bool_digest_or_schema_does_not_admit_reaction_or_route_tasks(api,purpose):
 r=reaction(api)
 for proof in (None,True,{"verified":True,"sha256":"0"*64,"schema":"admitted"}):
  with pytest.raises(ValueError,match="admission|heldout|evaluation|unavailable"):
   r.require_model_use(purpose,admission=proof)

def test_synthetic_official_route_reference_is_evaluation_exposure_control_never_training(api):
 r=api.RouteCandidate("p",nodes(api),(api.RouteStep("r1","p",("a","b","c")),),stock(api),
  b"native-reference","PaRoutes_n1_evaluation_reference")
 with pytest.raises(ValueError,match="evaluation"):r.require_model_use("training",admission={"verified":True})
 with pytest.raises(ValueError,match="admission|proof"):r.require_model_use("evaluation")

@pytest.mark.parametrize("indices",((),(0,0),[0],(True,),(-1,),tuple(range(17))))
def test_source_selectors_are_exact_bounded_unique_native_indices(api,indices):
 with pytest.raises(ValueError):api.RetrosynthesisSelector("ORD",indices)

def test_canonical_chemical_equivalence_is_unavailable_without_verified_semantic_evidence(api):
 r=reaction(api)
 with pytest.raises(ValueError,match="semantic|canonical|unavailable"):r.source_reference_multiset_key()

def test_synthetic_AND_OR_stock_solvability_uses_any_alternative_and_all_precursors(api):
 n=nodes(api);s=(api.RouteStep("and","p",("a","b")),api.RouteStep("alternative","p",("c",)))
 r=api.RouteCandidate("p",n,s,stock(api,("C","CO")),b"{}","synthetic_test")
 assert r.stock_termination_status()=="leaf_outside_declared_snapshot"
 assert r.structurally_stock_solvable() is True  # second alternative; not a chemical feasibility claim
 r=replace(r,steps=(s[0],),molecules=tuple(x for x in n if x.node_id!="c"))
 assert r.structurally_stock_solvable() is False
 assert replace(r,stock_snapshot=None).structurally_stock_solvable() is None

FIXTURE=Path("outputs/data-sources/20261101-retrosynthesis-source-v1/curated/bounded-retrosynthesis-native-fixture-v1/handoff-bounded-retrosynthesis-native-fixture-v1.json")
SHA="5ce3309b4e2c1885ba778bf9a5d2740eb2c53f1ff3735fe06d6c5795875499d4"
@pytest.fixture(scope="module")
def native_source(api):
 if not (ROOT/FIXTURE).is_file():pytest.skip("bounded source-owner safe native fixture unavailable")
 return api.RetrosynthesisNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=SHA)
@pytest.fixture(scope="module")
def actual(api,native_source):
 return dict(ORD=native_source.read_reactions(api.RetrosynthesisSelector("ORD",(0,1,2))),
  n1=native_source.read_routes(api.RetrosynthesisSelector("PaRoutes_n1",(0,))),
  n5=native_source.read_routes(api.RetrosynthesisSelector("PaRoutes_n5",(0,))))

def test_actual_ORD_preserves_native_roles_amounts_source_refs_and_default_target_exclusion(native_source,actual):
 for r in actual["ORD"]:
  assert len(r.products)==1 and len(r.precursors)==2
  assert len(r.auxiliaries)>=3
  assert {x.source_role for x in r.precursors}=={"REACTANT"}
  assert {"REAGENT","CATALYST"}<={x.source_role for x in r.auxiliaries}
  assert all(c.amount.presence=="present" and c.amount.unit=="MOLE" for c in r.precursors)
  inputs=r.prediction_inputs()
  assert set(inputs)=={"query_role","products"}
  assert "reaction_class" not in canonical(inputs).decode()
  assert native_source.verify(r)==r
  loc=json.loads(r.source_locator_bytes)
  assert loc["source_proto_sha256"] and type(loc["source_row"]) is int
  assert loc["serialization_role"]=="source_owner_semantic_JSON_of_native_protobuf"
  assert r.source_reference_multiset_key()
  assert json.loads(r.native_serialization)==json.loads(r.audit_bytes)["semantic"]

@pytest.mark.parametrize("protocol",("n1","n5"))
def test_actual_PaRoutes_reference_DAG_stock_locators_and_eval_roles_are_preserved(native_source,actual,protocol):
 r,=actual[protocol]
 assert r.source_role=="PaRoutes_"+protocol+"_evaluation_reference"
 assert r.stock_snapshot.role=="benchmark_snapshot"
 assert r.stock_snapshot.coverage=="bounded_positive_membership_subset"
 assert len(r.steps)>1 and len(r.molecules)>len(r.steps)
 assert r.stock_termination_status()=="all_leaves_in_declared_snapshot"
 assert r.structurally_stock_solvable() is True
 assert r.actual_purchasability_verified is False
 assert native_source.verify(r)==r
 assert set(r.prediction_inputs())=={"query_role","product"}
 loc=json.loads(r.source_locator_bytes)
 assert loc["protocol"]==protocol and loc["evaluation_only"] is True
 assert "stock_source" in loc and "target_source" in loc
 with pytest.raises(ValueError,match="evaluation"):r.require_model_use("training",admission=True)

@pytest.mark.parametrize("purpose",("training","evaluation","reaction_loss","route_search","split_use","peptide_supervision"))
def test_all_actual_native_reactions_and_routes_refuse_model_supervision(actual,purpose):
 for c in actual["ORD"]+actual["n1"]+actual["n5"]:
  with pytest.raises(ValueError):c.require_model_use(purpose,admission={"verified":True,"sha256":SHA})

def test_actual_gold_context_outcome_native_target_modification_leaves_product_inputs_unchanged_but_replay_refuses(api,native_source,actual):
 r=actual["ORD"][0]
 altered=replace(r,precursors=(component(api,"CCCC"),),audit_bytes=r.audit_bytes)
 assert altered.prediction_inputs()==r.prediction_inputs()
 with pytest.raises(ValueError):native_source.verify(altered)
 route,=actual["n1"];changed=replace(route,stock_snapshot=None)
 assert changed.prediction_inputs()==route.prediction_inputs()
 with pytest.raises(ValueError):native_source.verify(changed)

@pytest.mark.parametrize("source",("ORD","PaRoutes_n1","PaRoutes_n5"))
def test_safe_reader_has_no_raw_parquet_protobuf_or_whole_route_fallback(api,native_source,source):
 with pytest.raises(ValueError,match="fixture|available"):
  if source=="ORD":native_source.read_reactions(api.RetrosynthesisSelector(source,(100,)))
  else:native_source.read_routes(api.RetrosynthesisSelector(source,(1,)))

@pytest.mark.parametrize("artifact",("fixture","ORD","PaRoutes","chemistry","PaRoutes_available","ORD_available","schema"))
def test_consumed_native_payloads_and_source_evidence_hashes_cannot_be_substituted(api,native_source,artifact,monkeypatch):
 target=Path(native_source.source_references()[artifact]["path"]).resolve()
 read=Path.read_bytes;touched=[]
 def changed(path):
  raw=read(path)
  if path.resolve()==target:touched.append(True);return raw+b" "
  return raw
 monkeypatch.setattr(Path,"read_bytes",changed)
 with pytest.raises(ValueError):
  s=api.RetrosynthesisNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=SHA)
  s.read_reactions(api.RetrosynthesisSelector("ORD",(0,)))
 assert touched

def test_runtime_reads_JSON_only_never_decodes_protobuf_pickle_or_generated_source(api,native_source,monkeypatch):
 read=Path.read_bytes
 def safe(path):
  assert path.suffix.lower()==".json"
  return read(path)
 monkeypatch.setattr(Path,"read_bytes",safe)
 native_source.read_reactions(api.RetrosynthesisSelector("ORD",(0,)))
 native_source.read_routes(api.RetrosynthesisSelector("PaRoutes_n1",(0,)))

def test_source_native_peptide_absence_and_whole_QC_pending_remain_explicit(native_source):
 gates=native_source.gates()
 assert gates["peptide_eligible_records"]==0
 assert gates["whole_release_semantic_QC_claimed"] is False
 assert gates["task_admitted_records"]==0
 assert gates["full_dataset_ready"] is False

def test_modified_or_unrelated_manifest_refuses_before_any_native_record_construction(api,tmp_path):
 (tmp_path/"manifest.json").write_bytes(b'{"task_admitted_records":1}')
 with pytest.raises(ValueError):api.RetrosynthesisNativeSource(root=tmp_path,manifest="manifest.json",expected_manifest_sha256=SHA)
 with pytest.raises(ValueError):api.RetrosynthesisNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256="0"*64)


@pytest.mark.parametrize("field",("native_serialization","audit_bytes","source_locator_bytes","derived_reference","amount","source_role"))
def test_actual_native_reaction_typed_structure_and_hidden_metadata_replay_are_exact(api,native_source,actual,field):
 r=actual["ORD"][0]
 if field=="native_serialization":changed=replace(r,native_serialization=r.native_serialization+b" ")
 elif field=="audit_bytes":
  audit=json.loads(r.audit_bytes);audit["task_admitted"]=0
  changed=replace(r,audit_bytes=canonical(audit))
 elif field=="source_locator_bytes":
  loc=json.loads(r.source_locator_bytes);loc["source_row"]=False
  changed=replace(r,source_locator_bytes=canonical(loc))
 elif field=="derived_reference":
  c=r.precursors[0];ident=replace(c.identity,derived_reference_text="CC")
  changed=replace(r,precursors=(replace(c,identity=ident),)+r.precursors[1:])
 elif field=="amount":
  c=r.precursors[0];changed=replace(r,precursors=(replace(c,amount=api.NativeAmount("999","MOLE","present")),)+r.precursors[1:])
 else:changed=replace(r,source_role="synthetic_test")
 assert changed.prediction_inputs()==r.prediction_inputs()
 with pytest.raises(ValueError):native_source.verify(changed)

@pytest.mark.parametrize("field",("native_tree","source_locator","reaction_metadata","stock_ref","source_role"))
def test_actual_route_hidden_gold_stock_source_and_native_tree_replay_are_exact(api,native_source,actual,field):
 r,=actual["n1"]
 if field=="native_tree":changed=replace(r,native_serialization=r.native_serialization+b" ")
 elif field=="source_locator":
  loc=json.loads(r.source_locator_bytes);loc["evaluation_only"]=1
  changed=replace(r,source_locator_bytes=canonical(loc))
 elif field=="reaction_metadata":
  step=r.steps[0]
  changed=replace(r,steps=(replace(step,native_step_bytes=canonical({"gold_reaction_class":"hidden"})),)+r.steps[1:])
 elif field=="stock_ref":
  ref=json.loads(r.stock_snapshot.source_ref_bytes);ref["sha256"]="0"*64
  changed=replace(r,stock_snapshot=replace(r.stock_snapshot,source_ref_bytes=canonical(ref)))
 else:changed=replace(r,source_role="synthetic_test")
 assert changed.prediction_inputs()==r.prediction_inputs()
 with pytest.raises(ValueError):native_source.verify(changed)

def test_source_reference_multiset_verification_cannot_be_granted_by_schema_evidence(api,native_source,actual):
 r=actual["ORD"][0]
 assert native_source.verified_reference_multiset_key(r)==r.source_reference_multiset_key()
 c=r.precursors[0];fake=replace(c.identity,derived_evidence_bytes=canonical({"verified":True,"sha256":"0"*64}))
 forged=replace(r,precursors=(replace(c,identity=fake),)+r.precursors[1:])
 with pytest.raises(ValueError):native_source.verified_reference_multiset_key(forged)

def test_actual_complete_native_serializations_and_chemical_axes_are_never_cropped(native_source,actual):
 for r in actual["ORD"]:
  audit=json.loads(r.audit_bytes)
  assert r.native_serialization==canonical(audit["semantic"])
  all_components=r.products+r.precursors+r.auxiliaries
  assert all(c.identity.native_bytes()==json.loads(c.native_component_bytes)["identifiers"][0]["value"].encode() for c in all_components)
 for protocol in ("n1","n5"):
  r,=actual[protocol];tree=json.loads(r.native_serialization)
  def count(node):
   return 1+sum(count(c) for c in node.get("children",[]))
  assert count(tree)==len(r.molecules)+len(r.steps)
