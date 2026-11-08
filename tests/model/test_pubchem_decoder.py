"""PubChem native-byte decoder: real admitted pilot plus labeled unit controls."""
import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.tokenization.shortest_path import ByteTokenizer

ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
OUT=ROOT/"outputs/implementation/aux-native-readers-20261101/PubChem-neural-v1"
READER_PATH=Path(os.environ.get("MORA3_PUBCHEM_MODULE",ROOT/"outputs/implementation/aux-native-readers-20261101/PubChem-native-v2-strict-types/staging/src/mora3/data/pubchem_native.py"))
MASTER=Path("outputs/data-intake/20261101-PubChem-corpus-pilot-v1/independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json")
SHA="fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
LONG={
    "reference_prior":"303fc17e7bfcf0dd6fa15098acf1e16c910c1efd89c7146ec6d543d26ec23092",
    "smiles_to_selfies":"1faabb9f90bfdfb2c5e109c03a09c2a8e1801a8374ae60065c1cd71da35c9500",
    "selfies_to_smiles":"530faf5263c2640c4c335c8a0eeb4573ba4f5728ed7744b100be80bc7d3f1bc5"}
_reader=None


def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def reader():
    global _reader
    if _reader is None:
        if os.environ.get("MORA3_PUBCHEM_DECODER_MODULE"):
            _reader=load("mora3.data.pubchem_native",READER_PATH)
        else:_reader=importlib.import_module("mora3.data.pubchem_native")
    return _reader


def api():
    sys.modules["mora3.data.pubchem_native"]=reader()
    path=os.environ.get("MORA3_PUBCHEM_DECODER_MODULE")
    if path:
        if not Path(path).is_file():pytest.fail("PubChem native-byte decoder is not implemented")
        return load("mora3.model._pubchem_staged_decoder",path)
    try:return importlib.import_module("mora3.model.pubchem_decoder")
    except ModuleNotFoundError:pytest.fail("PubChem native-byte decoder is not implemented")


@pytest.fixture(scope="module")
def views():
    if not (ROOT/MASTER).is_file():pytest.skip("admitted PubChem pilot unavailable")
    if not (ROOT/"outputs/implementation/I3-annotation-convextok-i43/vocabulary.json").is_file():
        pytest.skip("existing saved reference vocabulary unavailable")
    r=reader()
    return {d:r.read_pubchem_native_views(MASTER,root=ROOT,expected_manifest_sha256=SHA,direction=d,limit=1)[0] for d in r.DIRECTIONS}


@pytest.fixture(scope="module")
def long_views():
    if not (ROOT/MASTER).is_file():pytest.skip("admitted PubChem long axes unavailable")
    if not (ROOT/"outputs/implementation/I3-annotation-convextok-i43/vocabulary.json").is_file():
        pytest.skip("existing saved reference vocabulary unavailable")
    r=reader()
    return {d:r.read_pubchem_native_views(MASTER,root=ROOT,expected_manifest_sha256=SHA,direction=d,task_ids=[tid],limit=1)[0] for d,tid in LONG.items()}


def build(mod,max_tokens=8192,extra_learned=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12601)
        spec=json.loads((ROOT/"outputs/implementation/I3-annotation-convextok-i43/vocabulary.json").read_bytes())
        if extra_learned:spec["learned_hex"].append(b"unfitted-vocabulary-test-only".hex())
        vocab=AnnotationVocabulary(json.dumps(spec,sort_keys=True))
        shared=SharedAnnotationModel(vocab,("GO:0000001",),("1.1.1.1",),width=4,heads=2,layers=2,max_tokens=max_tokens)
        return mod.PubChemDecoder(shared,source_root=ROOT,manifest=MASTER,expected_manifest_sha256=SHA).double().eval()


def equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert type(a) is type(b) and a==b


def mutate(v,operation):
    p=v.provenance();inputs=v.inputs();targets=v.targets();native=v.target_native_bytes
    if operation=="target":
        key=next(iter(targets));targets[key]="CC";native=b"CC"
    elif operation=="native_only":native=native+b"C"
    elif operation=="input":inputs["target_length"]=len(native)
    elif operation=="source":p["envelope"]["source_family"]="other"
    elif operation=="split":p["split"]="test"
    elif operation=="group":p["effective_groups"].pop()
    elif operation=="master":p["corpus_binding"]["manifest_sha256"]="0"*64
    elif operation=="source_row_type":p["source_ref"]["native_record"]["row_index"]=False
    elif operation=="audit_type":p["native_payload"]["audit"]["source_geometry_present"]=0
    elif operation=="axis":p["axes"]["native_smiles_characters"]=1
    elif operation=="direction":return replace(v,direction="unsupported")
    ib=canonical(inputs);tb=canonical(targets);pb=canonical(p)
    binding=sha(canonical(dict(schema="mora3.PubChemNativeView/v1",inputs=sha(ib),targets=sha(tb),native_target=sha(native),provenance=sha(pb))))
    return replace(v,input_bytes=ib,target_bytes=tb,target_native_bytes=native,provenance_bytes=pb,binding_sha256=binding)


def test_fixed_lossless_byte_codec_and_explicit_controls():
    mod=api();codec=mod.PubChemByteCodec(ByteTokenizer((b"[C]",b"CC"),controls=("pad","node","edge")))
    for raw in [b"C[C@@H](O)Cl",b"[C][=C][Branch1]",b"\x00\xff\xc3",bytes(range(256))]:
        ids=codec.encode(raw);assert ids[0]==codec.bos and ids[-1]==codec.eos
        assert codec.decode(ids)==raw and codec.decode(ids+[codec.pad]*2)==raw
    assert codec.pad not in codec.tokenizer.encode(b"\x00")
    for ids in [[codec.bos,codec.bos,codec.eos],[codec.bos,codec.eos,0],[codec.bos,True,codec.eos],[]]:
        with pytest.raises(ValueError):codec.decode(ids)


@pytest.mark.parametrize("direction",["reference_prior","smiles_to_selfies","selfies_to_smiles"])
def test_real_short_admitted_direction_full_target_and_nonzero_shared_head_gradients(views,direction):
    mod=api();m=build(mod);v=views[direction]
    prefix,target=m.teacher_batch([v]);mask=target!=m.codec.pad
    assert m.codec.decode([m.codec.bos]+target[0,mask[0]].tolist())==v.target_native_bytes
    assert target[0,int(mask[0].sum())-1]==m.codec.eos
    loss=m.objective([v]);assert torch.isfinite(loss) and loss.item()>0
    loss.backward()
    for p in [m.shared.payload.embedding.weight,m.shared.backbone.feature.weight,m.rnn.weight_ih_l0,m.output.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert m.shared.go.output.weight.grad is None


def test_mixed_teacher_masks_and_CE_compare_actual_logits(views):
    m=build(api());vs=list(views.values());prefix,target=m.teacher_batch(vs);mask=target!=m.codec.pad
    logits=m([v.inputs() for v in vs],prefix)
    per_token=-logits.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
    expected=per_token[mask].mean()
    torch.testing.assert_close(m.objective(vs),expected,rtol=1e-12,atol=1e-12)
    assert int((target==m.codec.eos).sum())==3 and int((prefix==m.codec.bos).sum())==3
    assert (~mask).any()
    changed=logits.clone();changed[~mask]=1000
    actual=torch.nn.functional.cross_entropy(changed.reshape(-1,m.codec.size),target.reshape(-1),ignore_index=m.codec.pad)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("direction",["reference_prior","smiles_to_selfies","selfies_to_smiles"])
def test_real_long_complete_target_byte_token_roundtrip_and_masks_without_neural_forward(long_views,direction):
    mod=api();m=build(mod);v=long_views[direction]
    ids=m.codec.encode(v.target_native_bytes)
    assert m.codec.decode(ids)==v.target_native_bytes
    prefix,target=m.teacher_batch([v])
    assert len(ids)==prefix.shape[1]+1 and target[0,-1]==m.codec.eos
    assert m.codec.decode([m.codec.bos]+target[0].tolist())==v.target_native_bytes
    assert len(v.target_native_bytes)==(9290 if direction=="smiles_to_selfies" else 3536)
    key={"reference_prior":"native_smiles","smiles_to_selfies":"standard_selfies","selfies_to_smiles":"source_canonical_isomeric_smiles"}[direction]
    assert v.targets()[key].encode()==v.target_native_bytes


@pytest.mark.parametrize("direction",["reference_prior","smiles_to_selfies","selfies_to_smiles"])
def test_hidden_targets_audit_identity_length_cannot_affect_condition_or_generation(views,direction):
    m=build(api());v=views[direction];bad=mutate(v,"target")
    assert torch.equal(m.condition([v.inputs()]),m.condition([bad.inputs()]))
    equal(m.generate([v.inputs()],max_new_tokens=3),m.generate([bad.inputs()],max_new_tokens=3))
    for key in ["target","target_length","target_hash","source_id","audit","chemical_groups"]:
        with pytest.raises(ValueError):m.condition([{**v.inputs(),key:"injected"}])


@pytest.mark.parametrize("mutation",["target","native_only","input","source","split","group","master","source_row_type","audit_type","axis","direction"])
def test_rehashed_view_tampering_rejected_before_model_RNG_or_grad_mutation(views,mutation):
    m=build(api());bad=mutate(views["smiles_to_selfies"],mutation)
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):m.objective([bad])
    equal(state,m.state_dict());assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())


def test_full_known_SELFIES_survives_source_graph_packing_and_context_overflow_rejects(long_views):
    mod=api();v=long_views["selfies_to_smiles"];m=build(mod,max_tokens=8192)
    record,task=mod.pubchem_condition_graph(v.inputs())
    packed=m.shared.payload.codec.encode(record)
    condition=next(x for x in packed["entities"][0]["measurements"] if x["name"]=="condition")
    raw=m.shared.payload.codec.tokenizer.decode(condition["token_ids"])
    assert json.loads(raw)["standard_selfies"]==v.inputs()["standard_selfies"]
    assert len(json.loads(raw)["standard_selfies"])==9290
    before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="bound|context|stream"):m.condition([v.inputs()])
    equal(before,m.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_newhead_strict_state_vocab_and_source_configuration_binding(views):
    mod=api();a=build(mod);b=build(mod);state=copy.deepcopy(a.state_dict())
    b.load_state_dict(copy.deepcopy(state),strict=True);equal(a.state_dict(),b.state_dict())
    p,t=a.teacher_batch([views["reference_prior"]])
    assert torch.equal(a([views["reference_prior"].inputs()],p),b([views["reference_prior"].inputs()],p))
    for other in [build(mod,extra_learned=True),build(mod,max_tokens=16384)]:
        before=copy.deepcopy(other.state_dict())
        with pytest.raises(ValueError):other.load_state_dict(copy.deepcopy(state),strict=True)
        equal(before,other.state_dict())
    for mutate_state in [lambda s:s["_extra_state"].update(version=True),lambda s:s.pop("output.bias"),
                         lambda s:s.__setitem__("output.bias",torch.zeros(1)),lambda s:s.update(unexpected=torch.zeros(1))]:
        bad=copy.deepcopy(state);mutate_state(bad);before=copy.deepcopy(b.state_dict())
        with pytest.raises(ValueError):b.load_state_dict(bad,strict=True)
        equal(before,b.state_dict())


def test_generation_budget_nontermination_invalid_UTF8_and_EOS_are_retained(views):
    m=build(api());c=[views["smiles_to_selfies"].inputs()]
    with torch.no_grad():
        m.output.weight.zero_();m.output.bias.fill_(-100)
        m.output.bias[255]=100
    result=m.generate(c,max_new_tokens=3)
    assert result["terminated"]==(False,) and result["representation_bytes"]==(b"\xff"*3,)
    assert result["text"]==(None,) and result["utf8_valid"]==(False,)
    assert result["representation_validated"] is False and result["evidence_class"]=="model_produced"
    with torch.no_grad():m.output.bias[m.codec.eos]=200
    result=m.generate(c,max_new_tokens=3)
    assert result["terminated"]==(True,) and result["representation_bytes"]==(b"",)
    for n in [0,-1,True]:
        with pytest.raises(ValueError):m.generate(c,max_new_tokens=n)


def test_source_binding_must_be_corrected_reader_and_pilot_manifest(views):
    mod=api();m=build(mod);shared=m.shared;state=copy.deepcopy(shared.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        mod.PubChemDecoder(shared,source_root=ROOT,manifest=MASTER,expected_manifest_sha256="0"*64)
    equal(state,shared.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_rejected_nested_state_container_alias_cannot_partially_load_weights(views):
    mod=api();m=build(mod);state=copy.deepcopy(m.state_dict());bad=copy.deepcopy(state)
    # GO/EC extras use tuple axes; JSON canonical equality alone aliases tuples/lists.
    key=next(k for k,v in bad.items() if k.endswith("_extra_state") and isinstance(v,dict)
             and any(isinstance(x,tuple) for x in v.values()))
    field=next(k for k,v in bad[key].items() if isinstance(v,tuple))
    bad[key][field]=list(bad[key][field])
    bad["shared.payload.embedding.weight"]=bad["shared.payload.embedding.weight"]+.125
    with pytest.raises(ValueError):m.load_state_dict(bad,strict=True)
    equal(state,m.state_dict())


def test_constructor_preserves_existing_shared_parameters_modes_and_mixed_head_dtype(views):
    mod=api()
    vocabulary=AnnotationVocabulary(json.dumps(dict(learned_hex=[],controls=["pad","node","edge"])))
    shared=SharedAnnotationModel(vocabulary,("GO:0000001",),("1.1.1.1",),width=4,heads=2,layers=2)
    shared.go.double()
    shared.eval()
    before=copy.deepcopy(shared.state_dict());identities=[id(p) for p in shared.parameters()]
    mod.PubChemDecoder(shared,source_root=ROOT,manifest=MASTER,expected_manifest_sha256=SHA)
    equal(before,shared.state_dict())
    assert [id(p) for p in shared.parameters()]==identities and shared.training is False


def test_condition_entity_kind_has_no_dataset_source_identity(views):
    mod=api();m=build(mod)
    for v in views.values():
        record,task=mod.pubchem_condition_graph(v.inputs())
        packed=m.shared.payload.codec.encode(record)
        assert all("PubChem" not in e["kind"] and "pubchem" not in e["kind"] for e in packed["entities"])
        field=next(x for x in packed["entities"][0]["measurements"] if x["name"]=="condition")
        assert json.loads(m.shared.payload.codec.tokenizer.decode(field["token_ids"]))==v.inputs()

# Whole-master bridge tests: source validation and actual teacher masks, no NN forward.
WHOLE_MASTER=Path("outputs/data-intake/20261101-PubChem-native-corpus-v1/materialized/independent-final-v1/master-corpus-manifest-v21-PubChem-whole-native-train-v1.json")
WHOLE_SHA="0e85a28928c027b7855e3bf12713a172128e12bed1e17752bee5f1d27b3f70b2"
V22_MASTER=Path("outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json")
V22_SHA="82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2"
WHOLE_PREFIX="outputs/data-intake/20261101-PubChem-native-corpus-v1/materialized/shards/"



def build_whole(mod, manifest=V22_MASTER, digest=V22_SHA):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12601)
        spec=json.loads((ROOT/"outputs/implementation/I3-annotation-convextok-i43/vocabulary.json").read_bytes())
        shared=SharedAnnotationModel(AnnotationVocabulary(json.dumps(spec,sort_keys=True)),
            ("GO:0000001",),("1.1.1.1",),width=4,heads=2,layers=2,max_tokens=8192)
        return mod.PubChemDecoder(shared,source_root=ROOT,manifest=manifest,
            expected_manifest_sha256=digest).eval()


@pytest.fixture(scope="module")
def whole_views():
    r=reader();out={}
    for direction in r.DIRECTIONS:
        shard=WHOLE_PREFIX+("000001.parquet" if direction=="reference_prior" else "000000.parquet")
        # No corpus scan: exact admitted shard, deterministic native first task.
        out[direction]=r.read_pubchem_native_views(V22_MASTER,root=ROOT,expected_manifest_sha256=V22_SHA,
            direction=direction,limit=1,shard_paths=(shard,))[0]
    return out


@pytest.mark.parametrize("manifest,digest",[(WHOLE_MASTER,WHOLE_SHA),(V22_MASTER,V22_SHA)])
def test_whole_constructor_scope_state_shapes_and_pilot_legacy_semantics(manifest,digest,views):
    mod=api();pilot=build(mod);m=build_whole(mod,manifest,digest)
    assert {k:tuple(v.shape) for k,v in m.state_dict().items() if isinstance(v,torch.Tensor)} == {
        k:tuple(v.shape) for k,v in pilot.state_dict().items() if isinstance(v,torch.Tensor)}
    for key,value in m.state_dict().items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,pilot.state_dict()[key])
    state=m.get_extra_state()
    assert state["source_binding"]["manifest_sha256"]==digest
    assert state["source_binding"]["reader_sha256"]==sha(Path(reader().__file__).read_bytes())
    assert m.codec.configuration()==pilot.codec.configuration()
    old=views["reference_prior"]
    with pytest.raises(ValueError):
        m.teacher_batch([old])  # correct native bytes, WRONG configured view scope


def test_whole_teacher_exact_all_directions_and_strict_in_memory_roundtrip(whole_views):
    mod=api();m=build_whole(mod);vs=list(whole_views.values())
    before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    prefix,target=m.teacher_batch(vs);mask=target!=m.codec.pad
    assert int((prefix==m.codec.bos).sum())==3 and int((target==m.codec.eos).sum())==3
    for i,v in enumerate(vs):
        assert m.codec.decode([m.codec.bos]+target[i,mask[i]].tolist())==v.target_native_bytes
    assert torch.equal(rng,torch.get_rng_state())
    equal(before,m.state_dict());assert all(p.grad is None for p in m.parameters())
    other=build_whole(mod);other.load_state_dict(copy.deepcopy(before),strict=True);equal(before,other.state_dict())
    for change in [lambda x:x["_extra_state"]["source_binding"].update(manifest_sha256=SHA),
                   lambda x:x["_extra_state"]["source_binding"].update(reader_sha256="0"*64),
                   lambda x:x["_extra_state"]["source_binding"]["whole_component"].update(construction_proof_sha256="0"*64)]:
        bad=copy.deepcopy(before);change(bad);state=copy.deepcopy(other.state_dict())
        with pytest.raises(ValueError):other.load_state_dict(bad,strict=True)
        equal(state,other.state_dict())


@pytest.mark.parametrize("mutation",["target","native_only","input","source","split","group","master","source_row_type","audit_type","axis","direction"])
def test_whole_teacher_rejects_rehashed_source_scope_tampering_before_state_rng_grads(whole_views,mutation):
    m=build_whole(api());before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):m.teacher_batch([mutate(whole_views["smiles_to_selfies"],mutation)])
    equal(before,m.state_dict());assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())


def test_whole_constructor_rejects_stale_reader_bytes_before_new_parameters(monkeypatch):
    mod=api();pilot=build(mod);before=copy.deepcopy(pilot.shared.state_dict());rng=torch.get_rng_state().clone()
    monkeypatch.setattr(mod,"READER_SHA256","0"*64)
    with pytest.raises(ValueError):mod.PubChemDecoder(pilot.shared,source_root=ROOT,
        manifest=V22_MASTER,expected_manifest_sha256=V22_SHA)
    equal(before,pilot.shared.state_dict());assert torch.equal(rng,torch.get_rng_state())

@pytest.fixture(scope="module")
def whole_long_views(long_views):
    metadata=json.loads((ROOT/V22_MASTER).read_bytes())
    paths={s["sha256"]:s["path"] for s in metadata["shards"]}
    out={}
    for direction,v in long_views.items():
        path=paths[v.provenance()["corpus_binding"]["shard_sha256"]]
        out[direction]=reader().read_pubchem_native_views(V22_MASTER,root=ROOT,
            expected_manifest_sha256=V22_SHA,direction=direction,task_ids=(LONG[direction],),
            limit=1,shard_paths=(path,))[0]
    return out


def test_whole_long_axes_teacher_masks_and_known_context_fail_before_embedding(whole_long_views):
    mod=api();m=build_whole(mod);state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    for direction,v in whole_long_views.items():
        prefix,target=m.teacher_batch([v])
        assert target[0,-1]==m.codec.eos and not (target==m.codec.pad).any()
        assert m.codec.decode([m.codec.bos]+target[0].tolist())==v.target_native_bytes
        assert len(v.target_native_bytes)==(9290 if direction=="smiles_to_selfies" else 3536)
    known=whole_long_views["selfies_to_smiles"].inputs()
    record,task=mod.pubchem_condition_graph(known)
    packed=m.shared.payload.codec.encode(record)
    field=next(x for x in packed["entities"][0]["measurements"] if x["name"]=="condition")
    assert json.loads(m.shared.payload.codec.tokenizer.decode(field["token_ids"]))==known
    called=[]
    hooks=[m.shared.payload.embedding.register_forward_pre_hook(lambda *args:called.append("embedding")),
           m.shared.backbone.register_forward_pre_hook(lambda *args:called.append("backbone"))]
    try:
        with pytest.raises(ValueError,match="bound"):m.condition([known])
    finally:
        for h in hooks:h.remove()
    assert called==[]
    equal(state,m.state_dict());assert torch.equal(rng,torch.get_rng_state())

def test_whole_actual_old_reader_module_rejected_by_bytes_before_version_api_access(monkeypatch):
    mod=api();pilot=build(mod);state=copy.deepcopy(pilot.shared.state_dict());rng=torch.get_rng_state().clone()
    baseline=ROOT/"outputs/implementation/aux-native-readers-20261101/PubChem-whole-native-v1/baseline/src/mora3/data/pubchem_native.py"
    old_reader=load("mora3.data._pubchem_old_reader_contract_test",baseline)
    monkeypatch.setattr(mod,"native_reader",old_reader)
    with pytest.raises(ValueError,match="pinned"):
        mod.PubChemDecoder(pilot.shared,source_root=ROOT,manifest=MASTER,expected_manifest_sha256=SHA)
    equal(state,pilot.shared.state_dict());assert torch.equal(rng,torch.get_rng_state())
