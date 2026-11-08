"""RNAcentral neural bridge: actual admitted views, unchanged legacy architecture."""
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
from mora3.data.nucleotide_prior import NucleotidePriorView
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_vocabulary import AnnotationVocabulary

ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
STAGING=os.environ.get("MORA3_RNACENTRAL_NEURAL_STAGE")
OUT=ROOT/"outputs/implementation/aux-native-readers-20261101/RNAcentral-neural-v1"
READER=ROOT/"outputs/implementation/aux-native-readers-20261101/RNAcentral-prior-v1/staging/src/mora3/data/rnacentral_prior.py"
MASTER=Path("outputs/data-intake/20261101-PubChem-corpus-pilot-v1/independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json")
SHARD="outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1/shards/train/00000.parquet"
IDS=["62b4da33b44c519e59c2096b9af0d1c7e46e794f6c189d4ec789dbb37ec5a58c",
     "3d41feb83f97e5bb278dc1479b3747f6a6aa9806f2eae77d55fb1d4e99a4ea56"]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def sha(x):return hashlib.sha256(x).hexdigest()


@pytest.fixture(scope="module")
def modules():
    if not (ROOT/MASTER).is_file():
        pytest.skip("admitted RNAcentral pilot unavailable")
    if not STAGING:
        return (importlib.import_module("mora3.data.rnacentral_prior"), None,
                importlib.import_module("mora3.model.nucleotide_decoder"),
                importlib.import_module("mora3.model.nucleotide_cooperative"))
    reader=load("mora3.data.rnacentral_prior",READER)
    old=load("mora3.model._rnacentral_original_decoder",OUT/"baseline/src/mora3/model/nucleotide_decoder.py")
    new=load("mora3.model._rnacentral_new_decoder",OUT/"staging/src/mora3/model/nucleotide_decoder.py")
    key="mora3.model.nucleotide_decoder";saved=sys.modules.get(key);sys.modules[key]=new
    try:
        cooperative=load("mora3.model._rnacentral_new_cooperative",OUT/"staging/src/mora3/model/nucleotide_cooperative.py")
    finally:
        if saved is not None:sys.modules[key]=saved
        else:sys.modules.pop(key,None)
    return reader,old,new,cooperative


@pytest.fixture(scope="module")
def real_views(modules):
    reader=modules[0]
    views=reader.read_rnacentral_priors(MASTER,root=ROOT,expected_manifest_sha256=reader.MASTER_SHA256,
        shard_paths=[SHARD],task_ids=IDS,limit=2)
    by_id={v.provenance()["task_id"]:v for v in views}
    return tuple(by_id[i] for i in IDS)


def model(module,seed=12501):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        vocab=AnnotationVocabulary(json.dumps(dict(learned_hex=[],controls=["pad","node","edge"])))
        shared=SharedAnnotationModel(vocab,("GO:0000001",),("1.1.1.1",),width=4,heads=2,layers=2)
        return module.NucleotideDecoder(shared).double().eval()


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert type(a) is type(b) and a==b


def synthetic_legacy(domain,seq="AcTUNx"):
    c=canonical(dict(prior_domain=domain,source_organism=None))
    return NucleotidePriorView(seq,c,canonical(dict(sequence_sha256=sha(seq.encode()),condition_sha256=sha(c))))


def changed_view(v,edit,sequence=None,conditions=None):
    p=v.provenance();edit(p)
    seq=v.sequence if sequence is None else sequence
    c=v.input_bytes() if conditions is None else canonical(conditions)
    p["sequence_sha256"]=sha(seq.encode());p["condition_sha256"]=sha(c)
    return NucleotidePriorView(seq,c,canonical(p))


def test_real_RNAcentral_short_and_ambiguity_through_batch_condition_teacher_and_backward(modules,real_views):
    reader,old,new,coop=modules
    batch=coop.NucleotideBatch(real_views)
    assert batch.views==real_views and isinstance(batch.binding(),str)
    m=model(new)
    prefix,target=m.teacher_batch(batch.views)
    assert prefix.shape==target.shape==(2,201)
    assert (target!=m.codec.pad).sum().item()==232
    assert sum(len(v.sequence) for v in real_views)==230
    assert sum((~v.target_tensors()[2]).sum().item() for v in real_views)==1
    assert all("U" not in v.sequence and "T" in v.sequence for v in real_views)
    h=m.condition(tuple(v.inputs() for v in real_views))
    assert h.shape==(2,4) and torch.isfinite(h).all()
    loss=m.objective(batch.views);assert torch.isfinite(loss) and loss.item()>0
    loss.backward()
    for p in (m.rnn.weight_ih_l0,m.output.weight,m.shared.backbone.feature.weight,m.shared.payload.embedding.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert m.shared.go.output.weight.grad is None


def test_condition_and_fixed_budget_generation_do_not_read_target_or_source_audit(modules,real_views):
    m=model(modules[2]);v=real_views[0]
    hidden=changed_view(v,lambda p:p.update(native_length=999999,source_ref={"source_status":"invented","source_identifier":"other","native_header":"changed"}),sequence="N"*31)
    assert torch.equal(m.condition([v.inputs()]),m.condition([hidden.inputs()]))
    a=m.generate([v.inputs()],max_new_symbols=3)
    b=m.generate([hidden.inputs()],max_new_symbols=3)
    equal(a,b)
    assert len(a["token_ids"][0])<=4
    for injection in ["target_length","target","URS","source_status","source_identifier","audit"]:
        with pytest.raises(ValueError):
            m.condition([{**v.inputs(),injection:"hidden"}])


@pytest.mark.parametrize("mutation",["source","master","group","status","receipt_flag","native_axis","payload_hash","new_full_QC","bool","target"])
def test_wrong_source_scope_provenance_types_and_target_rejected_before_model_state_changes(modules,real_views,mutation):
    m=model(modules[2]);v=real_views[0]
    def edit(p):
        if mutation=="source":p["source_family"]="Rfam15.1-native-reference-prior"
        elif mutation=="master":p["corpus_binding"]["manifest_sha256"]="0"*64
        elif mutation=="group":p["effective_groups"].pop()
        elif mutation=="status":p["source_ref"]["source_status"]="inactive"
        elif mutation=="receipt_flag":p["native_file_bytes_verified"]=False
        elif mutation=="native_axis":p["native_length"]=199
        elif mutation=="payload_hash":p["task_native_payload_sha256"]="0"*64
        elif mutation=="new_full_QC":p["source_ref"]["native_payload"]="outputs/unadmitted.parquet"
        elif mutation=="bool":p["source_ref"]["native_row_index"]=False
    bad=changed_view(v,edit,sequence=("A"*200 if mutation=="target" else None))
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):modules[3].NucleotideBatch([bad])
    with pytest.raises(ValueError):m.objective([bad])
    equal(state,m.state_dict());assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())


def test_unbound_reader_projection_and_duplicate_pool_are_not_admission(modules,real_views):
    v=real_views[0]
    bad=changed_view(v,lambda p:p.update(native_file_bytes_verified=False))
    with pytest.raises(ValueError):modules[3].NucleotideBatch([bad])
    with pytest.raises(ValueError):modules[3].NucleotideBatch([v,v])


def test_legacy_DNA_Rfam_logits_state_and_strict_roundtrip_unchanged(modules,real_views):
    baseline=model(modules[1] or modules[2]);staged=model(modules[2],seed=42)
    original=copy.deepcopy(baseline.state_dict())
    staged.load_state_dict(copy.deepcopy(original),strict=True)
    equal(baseline.get_extra_state(),staged.get_extra_state())
    equal(original,staged.state_dict())
    for domain in ["source_DNA_sequence","RNA_family_reference_native_symbols"]:
        views=[synthetic_legacy(domain)]
        p0,t0=baseline.teacher_batch(views);p1,t1=staged.teacher_batch(views)
        assert torch.equal(p0,p1) and torch.equal(t0,t1)
        assert torch.equal(baseline([views[0].inputs()],p0),staged([views[0].inputs()],p1))
        assert torch.equal(baseline.objective(views),staged.objective(views))
    copy_model=model(modules[2],seed=43)
    copy_model.load_state_dict(copy.deepcopy(staged.state_dict()),strict=True)
    assert torch.equal(staged.condition([real_views[0].inputs()]),copy_model.condition([real_views[0].inputs()]))
    # Old live decoder rejects new domain; no architecture migration/relaxed loader.
    if modules[1] is not None:
        with pytest.raises(ValueError):baseline.condition([real_views[0].inputs()])
    bad=copy.deepcopy(staged.get_extra_state());bad["symbols"]=("A",)
    with pytest.raises(ValueError):staged.set_extra_state(bad)


def test_live_baselines_and_immutable_reader_package_remain_exact():
    if not STAGING:
        pytest.skip("live ownership/hash checks apply before safe integration only")
    receipt=json.loads((OUT/"baseline-receipt.json").read_bytes())
    for r in receipt["baselines"]:
        assert sha((ROOT/r["path"]).read_bytes())==r["sha256"]
        assert sha((OUT/"baseline"/r["path"]).read_bytes())==r["sha256"]
    patch=ROOT/receipt["source_reader_patch"]
    assert sha(patch.read_bytes())==receipt["source_reader_patch_sha256"]


@pytest.mark.parametrize("organism",["invented taxon",False,1,{}])
def test_RNAcentral_unknown_organism_cannot_be_invented_as_generation_condition(modules,real_views,organism):
    m=model(modules[2])
    c=dict(real_views[0].inputs(),source_organism=organism)
    with pytest.raises(ValueError):m.condition([c])
    with pytest.raises(ValueError):m.generate([c],max_new_symbols=3)
