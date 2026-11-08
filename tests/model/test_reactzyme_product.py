"""Complete enzyme/substrate graph inputs and exact source product recurrence."""

from dataclasses import replace
import hashlib

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import canonical
from mora3.model.pubchem_decoder import PubChemDecoder
from test_annotation_mup import make as annotation
from test_pubchem_decoder import ROOT, MASTER, SHA
from test_sciplex_mup import make as response


def axes():
    from mora3.data.reactzyme_current import ReactZymeSourceAxes
    return ReactZymeSourceAxes('a' * 64, 'b' * 64, (32, 32, 64),
                              ('ACKT', '().=CNO', '().=CNO'), (2, 0, 0))


def row(product='CC.O', annotation='audit'):
    from mora3.data.reactzyme_current import FAMILY, REVISION, SCHEMA, TASK, _project
    task = dict(schema=SCHEMA, task=TASK, conditions=dict(amino_acid_sequence='ACCKT', substrate_smiles='C(C)O'),
                targets=dict(product_smiles=product), missing=[],
                audit=dict(annotation_score=5, ec_number=annotation, entry='audit', rhea_id='audit', source_date='audit'))
    raw = canonical(task)
    return _project(dict(source_family=FAMILY, task_name=TASK, native_schema=SCHEMA, source_revision=REVISION,
                         split='train', admitted_for_task=True, native_payload=raw,
                         native_payload_sha256=hashlib.sha256(raw).hexdigest(), condition_context_json='{}'),
                    'original.parquet', 1, axes())


def make(width=8):
    from mora3.model.reactzyme_product import NativeReactZymeProductModel
    shared = annotation(width, backbone=response(width).backbone)
    decoder = PubChemDecoder(shared, source_root=ROOT, manifest=MASTER,
                             expected_manifest_sha256=SHA, parameterization_policy='mup-sequence-v1')
    return NativeReactZymeProductModel(decoder, axes()).double().eval()


def test_source_projection_separates_complete_known_condition_and_original_product():
    original, changed = row(), row('CC=O', 'changed audit')
    assert original.condition == changed.condition
    assert original.product_bytes == b'CC.O' and changed.product_bytes == b'CC=O'
    assert not hasattr(original.condition, 'rhea_id')
    with pytest.raises(ValueError):
        row('CC!')


def test_two_known_nodes_and_all_complete_target_tokens_match_reference():
    model = make(16)
    original = row()
    records = []
    encoder = model.decoder.shared.payload
    call = encoder.forward_task

    def inspect(record, *, task, identifiers):
        records.append((record, task, identifiers))
        return call(record, task=task, identifiers=identifiers)

    encoder.forward_task = inspect
    context = model.condition((original.condition,))
    graph, task, identifiers = records[0]
    assert [n.measurements[0].value for n in graph.nodes] == ['ACCKT', 'C(C)O']
    assert len(graph.edges) == 1 and torch.equal(identifiers, torch.eye(2, dtype=torch.float64))
    assert task.input_fields(graph) == {('node', 0, 'amino_acid_sequence'), ('node', 1, 'substrate_smiles')}
    torch.testing.assert_close(context, model.condition((row('CC=O', 'other').condition,)), rtol=0, atol=0)
    decoder = model.decoder
    ids = torch.tensor(decoder.codec.encode(original.product_bytes))
    hidden, _ = decoder.rnn(decoder.embedding(ids[:-1][None]), context[None].contiguous())
    expected = F.cross_entropy(decoder._readout(hidden)[0], ids[1:])
    actual = model._reference_objective((original,), chunk_size=2, recompute=True)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    actual.backward()
    assert torch.count_nonzero(model.decoder.rnn.weight_hh_l0.grad)
    assert torch.count_nonzero(model.decoder.shared.backbone.incidence.weight.grad)
    with pytest.raises(ValueError):
        model._reference_objective((replace(original, product_bytes=b'bad!'),), chunk_size=2, recompute=False)


def test_alias_only_family_and_generation_provenance():
    from mora3.model.reactzyme_product import reactzyme_parameter_roles, reactzyme_width_family
    members = tuple(make(width) for width in (8, 12, 16))
    axis = ('context', 'observation')
    family = reactzyme_width_family(*members, identifier_axis=axis)
    assert family.total_numel == tuple(sum(p.numel() for p in m.decoder.parameters()) for m in members)
    assert all(name.startswith('decoder.') for name in reactzyme_parameter_roles(members[-1], identifier_axis=axis))
    model = members[0]
    generated = model.generate((row().condition,), max_new_tokens=2)
    assert generated['evidence_class'] == 'model_produced'
    assert generated['target_fields'] == ('product_smiles',)
    assert generated['representation_validated'] is False
