from dataclasses import replace
import copy

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import canonical, digest
from mora3.data.biogrid_chemical_current import (
    FAMILY, REVISION, SCHEMA, TASK, BioGRIDChemicalAxes, project_biogrid_chemical_row,
)
from mora3.model.biogrid_chemical_model import NativeBioGRIDChemicalModel, biogrid_chemical_parameter_roles
from mora3.model.parameterization import parameter_ledger, width_family
from test_annotation_mup import make as annotation
from test_sciplex_mup import make as response


AXIS = (('degradation', 'recruited E3 ligase'), ('degradation', 'recruited e3 ligase'),
        ('degradation ', 'recruited e3 ligase'))


def document(action=AXIS[0], audit_id='123'):
    return dict(
        schema=SCHEMA, task=TASK, missing=['biophysical_binding'],
        conditions=dict(chemical_identifier='BIOGRID:-', chemical_name='Known molecule α',
                        gene_identifier='329', organism_taxon='9606'),
        targets=dict(action=action[0], interaction_type=action[1]),
        audit=dict(evidence='original BIOGRID curated', source_row={
            '#BioGRID Chemical Interaction ID': audit_id, 'Chemical Source': 'BIOGRID',
            'Chemical Source ID': '-', 'Chemical Name': 'Known molecule α',
            'Entrez Gene ID': '329', 'Organism ID': '9606',
            'Action': action[0], 'Interaction Type': action[1], 'Notes': 'audit-only',
        }),
    )


def packet(doc=None, ordinal=0):
    raw = canonical(document() if doc is None else doc)
    return project_biogrid_chemical_row(dict(
        source_family=FAMILY, source_revision=REVISION, native_schema=SCHEMA, task_name=TASK, split='train',
        condition_context_json='{}', admitted_for_task=True, model_adapter_supported=False,
        evidence_class='source_curated_publication_supported_annotation', native_payload=raw,
        native_payload_sha256=digest(raw),
    ), 'original.parquet', ordinal)


def make(width=8, *, physical=False):
    shared = annotation(width, physical=physical, backbone=response(width).backbone)
    axes = BioGRIDChemicalAxes('a'*64, 'b'*64, 4, AXIS, (19, 190, 8, 7))
    return NativeBioGRIDChemicalModel(shared, axes).double().eval()


def test_projection_preserves_exact_categories_source_occurrences_and_known_missing_id():
    observed = packet(document(AXIS[2], '123'))
    other = packet(document(AXIS[1], '124'), 1)
    assert observed.condition == other.condition
    assert observed.target == AXIS[2] and other.target == AXIS[1]
    assert observed.condition.chemical_identifier == 'BIOGRID:-'
    assert not hasattr(observed.condition, 'audit')
    assert observed.source_coordinate != other.source_coordinate


@pytest.mark.parametrize('change', ('target_leak', 'audit_target_drift', 'chemical_id_drift', 'gene_id_drift'))
def test_refuses_native_field_drift(change):
    doc = document()
    if change == 'target_leak':
        doc['conditions']['action'] = doc['targets']['action']
    elif change == 'audit_target_drift':
        doc['targets']['action'] += ' '
    elif change == 'chemical_id_drift':
        doc['conditions']['chemical_identifier'] = 'OTHER:12'
    else:
        doc['conditions']['gene_identifier'] = '333'
    with pytest.raises(ValueError):
        packet(doc)


def test_complete_known_graph_and_joint_ce_preserve_repeated_observed_labels():
    m = make(16)
    packets = (packet(document(AXIS[2], '123'), 0), packet(document(AXIS[2], '124'), 1),
               packet(document(AXIS[0], '125'), 2))
    calls = []
    hook = m.shared.payload.register_forward_pre_hook(
        lambda _model, args, kwargs: calls.append((args[0], kwargs)), with_kwargs=True,
    )
    out = m._reference_objective(packets)
    hook.remove()
    assert len(calls) == 3
    for graph, fields in calls:
        assert len(graph.nodes) == 2 and len(graph.edges) == 1
        assert graph.edges[0].kind == 'known-source-annotation-packet'
        assert {x.name:x.value for x in graph.nodes[0].measurements} == {
            'chemical_identifier': 'BIOGRID:-', 'chemical_name': 'Known molecule α',
        }
        assert {x.name:x.value for x in graph.nodes[1].measurements} == {
            'gene_identifier': '329', 'organism_taxon': '9606',
        }
        assert len(fields['input_fields']) == 4
        assert 'audit-only' not in graph.canonical_bytes().decode()
    assert torch.equal(out['logits'][0], out['logits'][1])
    assert torch.equal(out['logits'][1], out['logits'][2])
    torch.testing.assert_close(out['loss'], F.cross_entropy(out['logits'], torch.tensor([2, 2, 0])))
    out['loss'].backward()
    assert torch.count_nonzero(m.readout.weight.grad)
    assert torch.count_nonzero(m.shared.backbone.incidence.weight.grad)


def test_exact_finite_axis_roles_and_refusal_before_neural_and_checkpoint_mutation():
    models = tuple(make(width) for width in (8, 12, 16))
    roles = tuple(biogrid_chemical_parameter_roles(m, identifier_axis=('context','observation')) for m in models)
    family = width_family(*(parameter_ledger(m, r) for m, r in zip(models, roles, strict=True)))
    assert family.total_numel == tuple(sum(p.numel() for p in m.shared.parameters()) + len(AXIS)*(m.width + 1)
                                      for m in models)
    m = models[0]
    def no_encoder(*args, **kwargs):
        pytest.fail('unknown native target must refuse before encoder')
    m.shared.payload.forward = no_encoder
    with pytest.raises(ValueError, match='axis'):
        m._reference_objective((replace(packet(), target=('unknown', 'source')),))
    state = copy.deepcopy(m.state_dict())
    state['_extra_state']['objective'] = 'invalid'
    state['readout.weight'].add_(1)
    before = m.readout.weight.detach().clone()
    with pytest.raises(ValueError, match='checkpoint'):
        m.load_state_dict(state)
    assert torch.equal(before, m.readout.weight)


def test_exact_physical_annotation_owner_alias_and_complete_roles():
    from mora3.model.physical_trainer import SharedPhysicalModel

    model = make(physical=True)
    assert type(model.shared) is SharedPhysicalModel
    roles = biogrid_chemical_parameter_roles(model, identifier_axis=('context', 'observation'))
    assert set(roles) == {name for name, _ in model.named_parameters(remove_duplicate=False)}
    assert 'shared.physical.method.weight' in roles
