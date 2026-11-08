"""ProCyon candidate contracts: safe real source reads; no model admission."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import io
import csv
import json
import os
from pathlib import Path
import sys

import pytest
from mora3.tokenization.shortest_path import ByteTokenizer

ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
HANDOFF=Path("outputs/data-sources/20261101-ProCyon-source-v1/handoff-ProCyon-open-native-components-v1.json")
SHA="0cc11e0d7be06d59c0ffbeea4f8d5a3eb38631258af4892508c513d488cc43ef"
FAMILIES=("protein_GO","protein_Reactome","domain_GO","domain_Pfam","protein_GtoP")


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def load_api():
    path=Path(os.environ.get("MORA3_PROCYON_MODULE","src/mora3/data/procyon_native.py"))
    if not path.exists():pytest.fail("ProCyon source-bound native candidate reader is not implemented")
    spec=importlib.util.spec_from_file_location("mora3.data.procyon_native",path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def api():
    return load_api()


@pytest.fixture(scope="module")
def source(api):
    if not (ROOT/HANDOFF).is_file():pytest.skip("pinned native ProCyon source unavailable")
    assert hashlib.sha256((ROOT/HANDOFF).read_bytes()).hexdigest()==SHA
    return api.ProCyonNativeSource(root=ROOT,handoff=HANDOFF,expected_handoff_sha256=SHA)


@pytest.fixture(scope="module")
def real_cases(api,source):
    return {f:source.read_relations(api.RelationSelector(f,(0,1))) for f in FAMILIES}


@pytest.mark.parametrize("family",FAMILIES)
def test_real_original_families_keep_exact_rows_sequences_and_official_splits(real_cases,family):
    a,b=real_cases[family]
    fields=a.native_fields()
    assert a.status=="native_source_candidate_unadmitted"
    assert fields["seq_id"]==a.sequence.native_id
    assert fields["seq_type"]==a.sequence.entity_kind
    assert fields["split"]==a.native_split
    assert a.native_split in ("CL_train","eval_two_shot","eval_zero_shot_hard")
    assert a.source_row_index==0 and b.source_row_index==1
    assert a.binding_sha256!=b.binding_sha256
    assert len(a.sequence.sequence_bytes)>0
    assert ByteTokenizer().decode(ByteTokenizer().encode(a.sequence.sequence_bytes))==a.sequence.sequence_bytes
    audit=a.audit()
    assert audit["dataset_revision"]=="2b99a397917282b3184d616bf01825e340488c88"
    assert audit["instruction_parent"] is None and audit["LLM_rephrase_available"] is False
    assert audit["current_UniProt_qualification_verified"] is False
    assert audit["independent_experimental_measurement_count"] is None
    assert audit["native_source_row_multiplicity"]==1
    assert a.relation_assertions
    assert source_verify_helper(a)


def source_verify_helper(candidate):
    # Constructor invariants retain exact native cells and raw record bytes.
    fields=candidate.native_fields()
    row=next(csv.DictReader(io.StringIO(candidate.raw_csv_header.decode()+candidate.raw_csv_row.decode(),newline="")))
    return row==fields


@pytest.mark.parametrize("family",FAMILIES)
def test_real_bound_candidate_replay_and_predictor_target_exclusion(api,source,real_cases,family):
    v=real_cases[family][0]
    assert source.verify(v)==v
    inputs=v.prediction_inputs()
    assert set(inputs)=={"entity_kind","native_sequence","query_role"}
    assert inputs["native_sequence"].encode()==v.sequence.sequence_bytes
    assert inputs["query_role"]=="known_sequence_for_candidate_relation_query"
    forbidden={"seq_id","text_id","relation","split","native_header","source_path","row_index",
        "protein_name","target_family_name","target_mask","target_length","audit"}
    assert not forbidden & set(inputs)
    assert source_verify_helper(v)


def test_original_GO_ordered_relation_and_GtoP_numeric_recode_are_not_global_labels(real_cases):
    go=real_cases["protein_GO"][0]
    assert go.native_fields()["relation"]=="['involved_in_Process']"
    assert go.relation_assertions==("involved_in_Process",)
    assert go.indexed_fields()["relation"]=="0"
    gtop=real_cases["protein_GtoP"][0]
    assert gtop.native_fields()["text_id"]=="1081" and gtop.indexed_fields()["text_id"]=="0"
    assert gtop.indexed_fields()["target_family_id"]=="1081"
    assert gtop.relation_assertions==("protein_gtop",)
    assert gtop.audit()["indexed_relation_role"]=="source_transformed_index_not_global_relation2id"


@pytest.mark.parametrize("family",("domain_GO","domain_Pfam"))
def test_domain_ID_text_does_not_invent_parent_coordinates_or_binding(real_cases,family):
    v=real_cases[family][0]
    assert "_" in v.sequence.native_id
    assert v.sequence.parent_protein_id is None and v.sequence.parent_coordinates is None
    assert "domain_parent_coordinates_unresolved" in v.gates
    with pytest.raises(ValueError,match="unresolved|unavailable|admission"):
        v.require_model_use("domain_parent_structure")


@pytest.mark.parametrize("purpose",("training","evaluation","QA","caption","retrieval"))
@pytest.mark.parametrize("family",FAMILIES)
def test_real_candidates_never_grant_model_task_admission_from_boolean_or_digest(real_cases,family,purpose):
    v=real_cases[family][0]
    for proof in (None,True,{"verified":True,"sha256":"0"*64,"schema":"admitted"}):
        with pytest.raises(ValueError,match="admission|heldout|unavailable"):
            v.require_model_use(purpose,admission=proof)


def test_official_heldout_suffix_never_becomes_train(real_cases):
    for family in ("protein_Reactome","domain_Pfam"):
        v=real_cases[family][0]
        assert v.native_split.startswith("eval_")
        with pytest.raises(ValueError,match="heldout"):
            v.require_model_use("training")
    with pytest.raises(ValueError,match="admission"):
        real_cases["protein_GO"][0].require_model_use("training")


@pytest.mark.parametrize("bad",(True,0.0,"0",-1))
def test_selector_ordinals_are_exact_nonnegative_integers(api,bad):
    with pytest.raises(ValueError):api.RelationSelector("protein_GO",(bad,))
    with pytest.raises(ValueError):api.SequenceSelector("protein",(bad,))


@pytest.mark.parametrize("indices",((),(0,0),tuple(range(33)),[0]))
def test_selectors_are_immutable_bounded_and_do_not_duplicate_rows(api,indices):
    with pytest.raises(ValueError):api.RelationSelector("protein_GO",indices)


@pytest.mark.parametrize("field,value",(("bytes",True),("bytes",1.0),("sha256","0"*63),("sha256","A"*64),("path",None)))
def test_native_file_reference_types_and_digests_are_strict(api,field,value):
    values=dict(path="file.csv",bytes=1,sha256="0"*64);values[field]=value
    with pytest.raises(ValueError):api.NativeFileRef(**values)


@pytest.mark.parametrize("unsupported",(
 "protein_uniprot","integrated_data/v1/protein_uniprot/random_uniprot_centric/protein_uniprot_relations_indexed.unified.csv",
 "relation2id","domain_info.pkl","../metadata.pkl","DrugBank","generated_instructions"))
def test_indexed_only_opaque_restricted_generated_and_unmapped_sources_refuse(api,source,unsupported):
    with pytest.raises(ValueError,match="unsupported|opaque|unsafe|original"):
        source.read_relations(api.RelationSelector(unsupported,(0,)))


def test_text_description_and_instruction_tasks_stay_unavailable(source,real_cases):
    assert "opaque_text_and_instruction_parent_metadata" in real_cases["protein_GO"][0].gates
    assert source.quarantines()["opaque_pickle_files"]==28
    assert source.quarantines()["indexed_only_UniProt_views"] is True
    with pytest.raises(ValueError,match="opaque|unavailable"):
        source.require_text_capability("QA")


def test_unsafe_pickle_assets_are_never_opened(api,source,monkeypatch):
    original=Path.read_bytes
    def guarded(path):
        assert path.suffix.lower() not in (".pkl",".pickle")
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    checked=api.ProCyonNativeSource(root=ROOT,handoff=HANDOFF,expected_handoff_sha256=SHA)
    checked.read_relations(api.RelationSelector("protein_GtoP",(0,)))
    with pytest.raises(ValueError):checked.read_relations(api.RelationSelector("domain_info.pkl",(0,)))


@pytest.mark.parametrize("family",("protein_GO","protein_GtoP"))
def test_hidden_native_labels_and_audit_mutation_does_not_change_candidate_predictors_but_replay_refuses(
 api,source,real_cases,family):
    v=real_cases[family][0]
    fields=v.native_fields()
    if family=="protein_GO":fields["text_id"]="GO:0000001"
    else:fields["protein_name"]="hidden-audit-mutated"
    text=io.StringIO(newline="");w=csv.DictWriter(text,fieldnames=next(csv.reader(io.StringIO(v.raw_csv_header.decode()))),lineterminator="\n")
    w.writerow(fields)
    indexed=v.indexed_fields()
    if family=="protein_GtoP":indexed["protein_name"]=fields["protein_name"]
    altered=replace(v,native_fields_bytes=canonical(fields),indexed_fields_bytes=canonical(indexed),raw_csv_row=text.getvalue().encode())
    assert altered.prediction_inputs()==v.prediction_inputs()
    assert altered.binding_sha256!=v.binding_sha256
    with pytest.raises(ValueError,match="binding|source|replay"):
        source.verify(altered)


@pytest.mark.parametrize("mutation",("ordinal","original_id","internal_index","indexed_split","native_split","evidence_role"))
def test_mapping_split_identity_and_evidence_mutations_refuse(real_cases,source,mutation):
    v=real_cases["protein_GO"][0]
    kwargs={}
    if mutation=="ordinal":kwargs["source_row_index"]=True
    elif mutation=="original_id":kwargs["native_fields_bytes"]=canonical({**v.native_fields(),"seq_id":"different"})
    elif mutation=="internal_index":kwargs["native_internal_index"]="0"
    elif mutation=="evidence_role":kwargs["evidence_role"]="LLM_generated_experimental_truth"
    else:
        data=v.indexed_fields() if mutation=="indexed_split" else v.native_fields()
        data["split"]="CL_train" if v.native_split!="CL_train" else "eval_zero_shot"
        kwargs["indexed_fields_bytes" if mutation=="indexed_split" else "native_fields_bytes"]=canonical(data)
    with pytest.raises(ValueError):
        changed=replace(v,**kwargs)
        source.verify(changed)


@pytest.mark.parametrize("kind",("protein","domain"))
def test_real_native_FASTA_complete_axis_and_header_byte_binding(api,source,kind):
    v,=source.read_sequences(api.SequenceSelector(kind,(0,)))
    assert v.native_ordinal==0 and v.native_id in v.native_header_bytes.decode()
    assert v.parent_protein_id is None and v.parent_coordinates is None
    assert b"".join(v.raw_record_bytes.splitlines()[1:])==v.sequence_bytes
    assert v.native_observed_mask()==(True,)*len(v.sequence_bytes)
    assert ByteTokenizer().decode(ByteTokenizer().encode(v.sequence_bytes))==v.sequence_bytes
    assert source.verify_sequence(v)==v
    assert v.audit()["internal_index_asserted"] is False


@pytest.mark.parametrize("field,value",(("native_ordinal",True),("sequence_bytes",b"XX"),("native_header_bytes",b">wrong")))
def test_sequence_axis_header_type_mutations_reject(api,source,field,value):
    v,=source.read_sequences(api.SequenceSelector("protein",(0,)))
    with pytest.raises(ValueError):
        other=replace(v,**{field:value})
        source.verify_sequence(other)


@pytest.mark.parametrize("artifact",("handoff","QC","transformation","license","source_csv","mapping","FASTA","sequence_payload"))
def test_corrupted_consumed_source_and_proof_bytes_are_rejected(api,source,artifact,monkeypatch):
    targets=source.source_references("protein_GtoP")
    path=Path(targets[artifact]["path"]).resolve()
    original=Path.read_bytes;touched=[]
    def changed(p):
        raw=original(p)
        if p.resolve()==path:touched.append(True);return raw+b" "
        return raw
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError):
        current=api.ProCyonNativeSource(root=ROOT,handoff=HANDOFF,expected_handoff_sha256=SHA)
        current.read_relations(api.RelationSelector("protein_GtoP",(0,)))
    assert touched


def test_unpinned_or_modified_handoff_never_enters_native_reader(api,tmp_path):
    (tmp_path/"handoff.json").write_bytes(b'{"task_admitted_records":1}')
    with pytest.raises(ValueError):api.ProCyonNativeSource(root=tmp_path,handoff="handoff.json",expected_handoff_sha256=SHA)
    with pytest.raises(ValueError):api.ProCyonNativeSource(root=ROOT,handoff=HANDOFF,expected_handoff_sha256="0"*64)


def test_desired_queries_are_explicit_and_never_observed_labels(api):
    desired=api.DesiredRelation("GO","GO:0003674")
    assert api.desired_procyon_inputs((desired,))=={
        "desired_source_relations":[{"namespace":"GO","native_identifier":"GO:0003674","role":"desired"}]}
    with pytest.raises(ValueError):api.DesiredRelation("GO","GO:0003674",role="observed")
    with pytest.raises(ValueError):api.desired_procyon_inputs((desired,desired))
    with pytest.raises(ValueError):api.DesiredRelation("DrugBank","DB0001")


def test_predictor_projection_rejects_observed_gold_source_and_target_keywords(api,source):
    sequence,=source.read_sequences(api.SequenceSelector("protein",(0,)))
    for key in ("target","source_id","target_mask","target_length","split","audit","gold_context"):
        with pytest.raises(ValueError):api.procyon_prediction_inputs(sequence,request={key:"hidden"})

def test_synthetic_original_GO_multiple_assertions_preserve_order_and_multiplicity(real_cases,source):
    # Synthetic mutation of a real candidate, never an actual source coverage claim.
    v=real_cases["protein_GO"][0]
    fields=v.native_fields()
    fields["relation"]="['involved_in_Process', 'acts_upstream_of_or_within_Process', 'involved_in_Process']"
    text=io.StringIO(newline="")
    csv.DictWriter(text,fieldnames=next(csv.reader(io.StringIO(v.raw_csv_header.decode()))),lineterminator="\n").writerow(fields)
    changed=replace(v,native_fields_bytes=canonical(fields),raw_csv_row=text.getvalue().encode())
    assert changed.relation_assertions==("involved_in_Process","acts_upstream_of_or_within_Process","involved_in_Process")
    assert changed.prediction_inputs()==v.prediction_inputs()
    with pytest.raises(ValueError):source.verify(changed)


@pytest.mark.parametrize("literal",("__import__('os').system('forbidden')","['involved_in_Process', 0]","[]"))
def test_unsafe_or_untyped_relation_literals_never_execute_or_become_labels(real_cases,literal):
    v=real_cases["protein_GO"][0]
    fields=v.native_fields();fields["relation"]=literal
    text=io.StringIO(newline="")
    csv.DictWriter(text,fieldnames=next(csv.reader(io.StringIO(v.raw_csv_header.decode()))),lineterminator="\n").writerow(fields)
    with pytest.raises(ValueError):
        replace(v,native_fields_bytes=canonical(fields),raw_csv_row=text.getvalue().encode())


@pytest.mark.parametrize("field,value",(("text_id",1),("relation",False),("split",None)))
def test_native_target_values_missingness_and_numeric_aliases_remain_typed_strings(real_cases,field,value):
    v=real_cases["protein_GO"][0]
    fields=v.native_fields();fields[field]=value
    with pytest.raises(ValueError):replace(v,native_fields_bytes=canonical(fields))

@pytest.mark.parametrize("kind,ordinal,length",(("protein",15164,8797),("domain",12946,1949)))
def test_real_long_native_axes_never_crop_at_model_context_bound(api,source,kind,ordinal,length):
    v,=source.read_sequences(api.SequenceSelector(kind,(ordinal,)))
    assert len(v.sequence_bytes)==length
    assert len(v.native_observed_mask())==length and all(v.native_observed_mask())
    tokenizer=ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(v.sequence_bytes))==v.sequence_bytes
    assert source.verify_sequence(v)==v


def test_safe_byte_reader_cannot_be_expanded_to_open_pickle_even_by_suffix_argument(source,monkeypatch):
    references=source.source_references("protein_GtoP")
    qc=json.loads(Path(references["QC"]["path"]).read_bytes())
    ref=qc["opaque_metadata"][0]["source"]
    original=Path.read_bytes
    def guarded(path):
        if path.suffix.lower() in (".pkl",".pickle"):
            pytest.fail("unsafe opaque pickle byte read attempted")
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    with pytest.raises(ValueError,match="unsafe|opaque|unsupported"):
        source._bytes(ref,{".pkl"})

def test_real_original_GO_multi_assertion_row_is_one_source_occurrence(api,source):
    v,=source.read_relations(api.RelationSelector("protein_GO",(75,)))
    assert v.relation_assertions==("involved_in_Process","acts_upstream_of_or_within_Process")
    assert v.audit()["native_source_row_multiplicity"]==1
    assert v.audit()["independent_experimental_measurement_count"] is None
    assert v.indexed_fields()["relation"]=="0"


def test_real_original_validation_suffix_survives_native_binding(api,source):
    v,=source.read_relations(api.RelationSelector("protein_GO",(160053,)))
    assert v.native_split=="CL_val_five_shot"
    assert v.indexed_fields()["split"]=="CL_val_five_shot"
    assert v.prediction_inputs()["query_role"]=="known_sequence_for_candidate_relation_query"
    with pytest.raises(ValueError,match="heldout"):
        v.require_model_use("training",admission={"verified":True})
