"""Runtime accounting checks actual IDs/masks/graph packing; fixtures are synthetic."""
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

from mora3.data.nucleotide_prior import NucleotidePriorView, nucleotide_prior_view
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.nucleotide_decoder import NucleotideDecoder
from mora3.model.tokengt import TokenGT, orthogonal_identifiers

ROOT = Path(os.environ.get("MORA3_REPO_ROOT", Path.cwd()))
VOCAB = Path("outputs/implementation/I3-annotation-convextok-i43/vocabulary.json")


def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def load_api():
    path = os.environ.get("MORA3_ACCOUNTING_MODULE")
    if not path:
        try:
            return importlib.import_module("mora3.native_token_accounting")
        except ModuleNotFoundError:
            pytest.fail("source-bound runtime token accounting is not implemented")
    if not Path(path).is_file():
        pytest.fail("source-bound runtime token accounting is not implemented")
    existing = sys.modules.get("mora3.native_token_accounting")
    if existing is not None and Path(existing.__file__).resolve() == Path(path).resolve():
        return existing
    spec = importlib.util.spec_from_file_location("mora3.native_token_accounting",path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def nucleotide(seq):
    c = canonical(dict(prior_domain="source_DNA_sequence",source_organism=None))
    return NucleotidePriorView(seq,c,canonical(dict(sequence_sha256=sha(seq.encode()),condition_sha256=sha(c))))


def build():
    if not (ROOT/VOCAB).is_file():
        pytest.skip("existing installed vocabulary artifact is unavailable")
    with torch.random.fork_rng():
        torch.manual_seed(10131)
        vocab = AnnotationVocabulary((ROOT/VOCAB).read_text())
        shared = SharedAnnotationModel(vocab,("GO:0000001",),("1.1.1.1",),
            width=4,heads=2,layers=2)
        return NucleotideDecoder(shared)


def source(api,view,key="a",primary="nucleotide"):
    return api.SourceView.bind(source_family="synthetic-DNA-unit-fixture",source_revision="synthetic-v1",
        view_id=key,source_payload_sha256=sha(view.canonical_bytes()),manifest_sha256="a"*64,
        reader_sha256=sha((ROOT/"src/mora3/data/nucleotide_prior.py").read_bytes()),
        split="train",primary_family=primary,modalities=("nucleotide","sequence"),
        conditions=view.inputs(),targets={"source_native_sequence":view.sequence})


def binding(api,model,*,objective="nucleotide",packing=None):
    return api.RuntimeBinding.create(root=ROOT,reader_functions=(nucleotide_prior_view,),
        model=model,tokenizers={"payload":model.shared.payload.codec.tokenizer,"target":model.codec},
        vocabulary_paths=(VOCAB,),objective_key=objective,
        context={"objective_kind":"autoregressive","graph_required":True,
            "image_representation":None},
        packing=packing or {"policy":"native-right-padded-teacher-forcing-v1",
            "scoring":"target-not-PAD","pad":0,"bos":1,"eos":2})


def run_native(api,model,views):
    b = binding(api,model)
    ss = tuple(source(api,v,str(i)) for i,v in enumerate(views))
    prefix,target = model.teacher_batch(views)
    with api.RuntimeCapture(b,model=model,tokenizers={"payload":model.shared.payload.codec.tokenizer,"target":model.codec}) as capture:
        capture.autoregressive(prefix,target,model.codec)
        with torch.inference_mode():
            loss = model.objective(views)
    return b,capture.records(ss),prefix,target,loss


def test_actual_native_teacher_masks_controls_and_condition_streams():
    api = load_api()
    m = build()
    before = copy.deepcopy(m.state_dict())
    rng = torch.get_rng_state().clone()
    b,records,prefix,target,loss = run_native(api,m,(nucleotide("AcTUNx"),nucleotide("Tu")))
    a,z = records
    assert torch.isfinite(loss)
    assert a.measures["ar_scored_positions"].value == 7
    assert z.measures["ar_scored_positions"].value == 3
    assert a.measures["ar_lexical_scored_positions"].value == 6
    assert z.measures["ar_eos_scored_positions"].value == 1
    assert z.measures["ar_target_padding_positions"].value == 4
    assert z.measures["ar_prefix_padding_positions"].value == 4
    assert a.measures["ar_bos_prefix_positions"].value == 1
    assert a.measures["condition_graph_positions"].value == 2
    assert a.measures["graph_node_positions"].value == 1
    assert a.measures["graph_edge_positions"].value == 0
    assert a.measures["condition_payload_embedding_positions"].value > 1
    assert a.measures["image_patch_positions"].value is None
    assert a.measures["condition_payload_embedding_positions"].unit != a.measures["condition_graph_positions"].unit
    assert torch.equal(torch.get_rng_state(),rng)
    for k,v in before.items():
        if isinstance(v,torch.Tensor):
            assert torch.equal(v,m.state_dict()[k])
        else:
            assert v == m.state_dict()[k]


def test_target_length_changes_teacher_positions_but_not_observed_conditions():
    api = load_api()
    m = build()
    _,short,_,_,_ = run_native(api,m,(nucleotide("Tu"),))
    _,long,_,_,_ = run_native(api,m,(nucleotide("AcTUNx"*10),))
    keys=("condition_graph_positions","condition_payload_embedding_positions",
        "graph_node_positions","graph_edge_positions")
    assert all(short[0].measures[k].value == long[0].measures[k].value for k in keys)
    assert short[0].input_trace_sha256 == long[0].input_trace_sha256
    assert short[0].measures["ar_scored_positions"].value == 3
    assert long[0].measures["ar_scored_positions"].value == 61


def test_explicit_score_mask_counts_selected_actual_targets_not_allocated_width():
    api = load_api()
    m = build()
    prefix,target=m.teacher_batch((nucleotide("AcTUNx"),nucleotide("Tu")))
    mask=target.ne(m.codec.pad)
    mask[0,1]=False
    measured=api.measure_autoregressive(prefix,target,m.codec,scored_mask=mask)
    assert measured[0]["ar_scored_positions"].value == 6
    assert measured[1]["ar_scored_positions"].value == 3
    logits=torch.zeros((*target.shape,m.codec.size))
    losses=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),reduction="none").reshape_as(target)
    assert losses[mask].numel() == 9
    mask[1,-1]=True
    with pytest.raises(ValueError):
        api.measure_autoregressive(prefix,target,m.codec,scored_mask=mask)


def test_graph_rows_use_actual_roles_validity_and_packing():
    api = load_api()
    with torch.random.fork_rng():
        torch.manual_seed(121)
        model=TokenGT(4,2,width=4,heads=2,layers=1)
    def graph(nodes,edges):
        endpoints=[[i,i] for i in range(nodes)]+edges
        return dict(features=torch.ones((len(endpoints),4)),
            roles=torch.tensor([0]*nodes+[1]*len(edges)),endpoints=torch.tensor(endpoints),
            identifiers=orthogonal_identifiers(nodes,2,seed=9))
    graphs=(graph(1,[]),graph(2,[[0,1]]))
    b=api.RuntimeBinding.create(root=ROOT,reader_functions=(nucleotide_prior_view,),model=model,
        tokenizers={},objective_key="graph-reference",
        context={"objective_kind":"graph_only","graph_required":True,"image_representation":None},
        packing={"policy":"TokenGT-pad_sequence-right-with-graph-special-v1"})
    ss=(source(api,nucleotide("A"),"a"),source(api,nucleotide("T"),"b"))
    with api.RuntimeCapture(b,model=model,tokenizers={}) as capture:
        with torch.inference_mode():
            _,valid=model(graphs)
    records=capture.records(ss)
    assert valid.tolist() == [[True,True,False,False],[True,True,True,True]]
    assert records[0].measures["condition_graph_positions"].value == 2
    assert records[0].measures["condition_graph_padding_positions"].value == 2
    assert records[1].measures["graph_node_positions"].value == 2
    assert records[1].measures["graph_edge_positions"].value == 1
    assert records[1].measures["graph_special_positions"].value == 1
    assert records[0].measures["condition_payload_embedding_positions"].status == "inapplicable"


def test_native_masked_elements_are_separate_from_neural_tokens():
    api = load_api()
    values=torch.tensor([1.,2.,float("nan"),4.])
    mask=torch.tensor([True,True,False,True])
    q=api.native_tensor_measure(values,unit="expression_values",observed=mask)
    assert q.value == 3 and q.unit == "expression_values"
    with pytest.raises(ValueError):
        api.native_tensor_measure(values,unit="expression_values",observed=torch.ones(4))
    with pytest.raises(ValueError):
        api.native_tensor_measure(values,unit="expression_values")


def test_runtime_and_revision_config_mismatch_rejected():
    api = load_api()
    m=build()
    b=binding(api,m)
    data=b.data()
    data["code_files"][0]["sha256"]="0"*64
    bad=api.RuntimeBinding(canonical(data))
    with pytest.raises(ValueError):
        bad.verify(model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec})
    with pytest.raises(ValueError):
        api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec},
            expected_binding_sha256="0"*64)
    m.shared.payload.max_tokens += 1
    with pytest.raises(ValueError):
        b.verify(model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec})


def test_missing_forward_cannot_claim_actual_scored_model_positions():
    api = load_api()
    m=build()
    b=binding(api,m)
    v=nucleotide("AcT")
    with api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec}) as capture:
        p,t=m.teacher_batch((v,))
        capture.autoregressive(p,t,m.codec)
    with pytest.raises(ValueError):
        capture.records((source(api,v),))


def test_unsupported_objectives_remain_null_and_do_not_become_runtime_exposures():
    api = load_api()
    b=api.RuntimeBinding.create(root=ROOT,reader_functions=(nucleotide_prior_view,),
        model=None,tokenizers={},objective_key="PubChem-unsupported",
        context={"objective_kind":"unsupported","graph_required":False,"image_representation":None},
        packing={"policy":"not-implemented"})
    record=api.unsupported_record(source(api,nucleotide("A"*9000)),b,
        reason="PubChem native model objective is not registered",
        native_units={"native_SMILES_characters":api.Measure(9000,"native_characters","measured","source audit only")})
    assert all(v.value is None and v.status == "unsupported" for v in record.measures.values())
    ledger=api.AccountingLedger({"PubChem-unsupported":b.sha256})
    ledger.add("source-only",record,kind="source_inspection")
    snap=ledger.snapshot()
    assert snap["sampled_model_exposures"] == 0
    assert snap["whole_corpus_model_positions"] is None
    assert snap["position_metrics"]["ar_scored_positions"]["measured_sum"] is None
    with pytest.raises(ValueError):
        ledger.add("fake-training",record,kind="diagnostic_scoring")


def test_unique_views_exposures_partitions_and_modality_presence_are_distinct():
    api = load_api()
    m=build()
    b,records,_,_,_=run_native(api,m,(nucleotide("AcT"),nucleotide("Tu")))
    ledger=api.AccountingLedger({"nucleotide":b.sha256})
    ledger.add("event-0:row-0",records[0],kind="diagnostic_scoring")
    ledger.add("event-0:row-1",records[1],kind="diagnostic_scoring")
    ledger.add("event-1:row-0",records[0],kind="diagnostic_scoring")
    snap=ledger.snapshot()
    assert snap["unique_registered_views"] == 2
    assert snap["sampled_model_exposures"] == 3
    assert snap["optimizer_training_exposures"] == 0
    assert snap["position_metrics"]["ar_scored_positions"]["measured_sum"] == 11
    assert snap["primary_partitions"]["nucleotide"]["unique_views"] == 2
    assert snap["modality_presence"]["nucleotide"]["unique_views"] == 2
    assert snap["modality_presence"]["sequence"]["unique_views"] == 2
    assert snap["whole_corpus_model_positions"] is None
    with pytest.raises(ValueError):
        ledger.add("event-0:row-0",records[0],kind="diagnostic_scoring")
    changed=replace(records[0],source=replace(records[0].source,input_sha256="0"*64))
    with pytest.raises(ValueError):
        ledger.add("different-bytes-same-view",changed,kind="diagnostic_scoring")
    assert ledger.snapshot() == snap


@pytest.mark.parametrize("status,unit,reason", [
    ("invented","tokens","why"), ("unsupported","","why"), ("inapplicable","pixels",""),
])
def test_null_counts_keep_valid_status_unit_and_reason_types(status, unit, reason):
    api=load_api()
    with pytest.raises(ValueError):
        api.Measure(None,unit,status,reason)


def test_actual_forward_conditions_reject_wrong_source_input_binding():
    api=load_api()
    m=build()
    v=nucleotide("AcT")
    b=binding(api,m)
    p,t=m.teacher_batch((v,))
    with api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec}) as capture:
        capture.autoregressive(p,t,m.codec)
        with torch.inference_mode():
            m.objective((v,))
    wrong=replace(source(api,v),input_sha256="0"*64)
    with pytest.raises(ValueError):
        capture.records((wrong,))


def test_source_reader_revision_must_be_the_runtime_bound_reader():
    api=load_api()
    m=build()
    v=nucleotide("AcT")
    b=binding(api,m)
    p,t=m.teacher_batch((v,))
    with api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec}) as capture:
        capture.autoregressive(p,t,m.codec)
        with torch.inference_mode():
            m.objective((v,))
    wrong=replace(source(api,v),reader_sha256="0"*64)
    with pytest.raises(ValueError):
        capture.records((wrong,))


def test_native_cnn_pixels_masks_and_affinities_are_elements_not_patch_tokens():
    from dataclasses import asdict
    import numpy as np
    from mora3.data.raster_morphology import RasterMorphologyTask
    from mora3.model.morphology import MorphologyInstance
    from mora3.model.instance_affinity import affinity_targets
    api=load_api()
    image=np.arange(6,dtype=np.uint8).reshape(2,3)
    labels=np.array([[1,1,0],[0,0,0]],dtype=np.uint8)
    task=RasterMorphologyTask(2,3,"uint8","uint8",image.tobytes().hex(),labels.tobytes().hex(),
        "synthetic-image-unit-fixture","v1","image-0","a"*64,sha(image.tobytes()),sha(labels.tobytes()),
        "unit-test-only","parent-0","train","synthetic","synthetic","brightfield","synthetic",None)
    with torch.random.fork_rng():
        torch.manual_seed(112)
        model=MorphologyInstance(4)
    b=api.RuntimeBinding.create(root=ROOT,reader_functions=(RasterMorphologyTask.from_bytes,),
        model=model,tokenizers={},objective_key="native-image",
        context={"objective_kind":"continuous","graph_required":True,
            "image_representation":"native-CNN-single-image-graph-v1"},
        packing={"policy":"TokenGT-pad_sequence-right-with-graph-special-v1"})
    src=api.SourceView.bind(source_family="synthetic-image-unit-fixture",source_revision="v1",view_id="image-0",
        source_payload_sha256=sha(task.canonical_bytes()),manifest_sha256="a"*64,
        reader_sha256=sha((ROOT/"src/mora3/data/raster_morphology.py").read_bytes()),
        split="train",primary_family="image-segmentation",modalities=("image","segmentation"),
        conditions=api.condition_descriptor(task),targets={"labels_sha256":sha(labels.tobytes())})
    instances=task.instances()
    affinity,valid=affinity_targets(instances)
    native=dict(input_image_pixels=api.native_tensor_measure(task.input_image(),unit="image_pixels"),
        scored_foreground_pixels=api.native_tensor_measure(instances.any(0),unit="foreground_pixels"),
        scored_affinity_labels=api.native_tensor_measure(affinity,unit="neighbor_affinity_labels",observed=valid))
    with api.RuntimeCapture(b,model=model,tokenizers={}) as capture:
        with torch.inference_mode():
            out=model(task,for_training=True)
    assert out.shape == (3,2,3)
    record,=capture.records((src,),native_units=(native,))
    assert record.condition_descriptor_verified is True
    assert record.measures["condition_graph_positions"].value == 2
    assert record.measures["image_patch_positions"].value is None
    assert record.measures["image_mask_token_positions"].value is None
    assert record.measures["ar_scored_positions"].value is None
    ledger=api.AccountingLedger({"native-image":b.sha256})
    ledger.add("image-call",record,kind="diagnostic_scoring")
    metrics=ledger.snapshot()["native_unit_metrics"]
    assert metrics["input_image_pixels"]["sampled_runtime_measured_sum"] == 6
    assert metrics["scored_foreground_pixels"]["sampled_runtime_measured_sum"] == 6
    assert metrics["scored_affinity_labels"]["sampled_runtime_measured_sum"] == 7


def test_unsupported_inspection_still_requires_its_exact_reader_revision():
    api=load_api()
    b=api.RuntimeBinding.create(root=ROOT,reader_functions=(nucleotide_prior_view,),model=None,
        tokenizers={},objective_key="unsupported",context={"objective_kind":"unsupported","graph_required":False,
        "image_representation":None},packing={"policy":"not-implemented"})
    with pytest.raises(ValueError):
        api.unsupported_record(replace(source(api,nucleotide("A")),reader_sha256="0"*64),b,
            reason="no model objective")


def test_teacher_targets_must_match_the_source_target_descriptor():
    api=load_api()
    m=build()
    v=nucleotide("AcT")
    b=binding(api,m)
    p,t=m.teacher_batch((v,))
    with api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec}) as capture:
        capture.autoregressive(p,t,m.codec)
        with torch.inference_mode():
            m.objective((v,))
    with pytest.raises(ValueError):
        capture.records((replace(source(api,v),target_sha256="0"*64),))


def test_failed_capture_removes_hooks_and_cannot_report_scored_exposure():
    api=load_api()
    m=build()
    v=nucleotide("AcT")
    b=binding(api,m)
    p,t=m.teacher_batch((v,))
    before=sum(len(q._forward_hooks)+len(q._forward_pre_hooks) for q in m.modules())
    capture=api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec})
    with pytest.raises(RuntimeError):
        with capture:
            capture.autoregressive(p,t,m.codec)
            with torch.inference_mode():
                m.objective((v,))
            raise RuntimeError("aborted scoring event")
    assert sum(len(q._forward_hooks)+len(q._forward_pre_hooks) for q in m.modules()) == before
    with pytest.raises(ValueError):
        capture.records((source(api,v),))


def test_configuration_failure_at_capture_exit_cannot_be_reused_as_valid_evidence():
    api=load_api()
    m=build()
    v=nucleotide("AcT")
    b=binding(api,m)
    capture=api.RuntimeCapture(b,model=m,tokenizers={"payload":m.shared.payload.codec.tokenizer,"target":m.codec})
    p,t=m.teacher_batch((v,))
    with pytest.raises(ValueError):
        with capture:
            capture.autoregressive(p,t,m.codec)
            with torch.inference_mode():
                m.objective((v,))
            m.shared.payload.max_tokens += 1
    with pytest.raises(ValueError):
        capture.records((source(api,v),))
