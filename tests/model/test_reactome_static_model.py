from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import canonical, digest
from mora3.data.reactome_static_pathways import (
    FAMILY, ROLE_AXIS, SCHEMA, TASK, ReactomeStaticAxes, project_reactome_row,
)
from mora3.model.reactome_static_model import NativeReactomeStaticModel, reactome_static_parameter_roles
from mora3.model.parameterization import parameter_ledger, width_family
from test_annotation_mup import make as annotation
from test_sciplex_mup import make as response


def document():
    return dict(
        schema=SCHEMA, task=TASK, missing=['kinetic_laws'],
        conditions=dict(
            known_participant_incidence=True,
            compartments=[dict(name='literal cell compartment', reference_qualifiers={'is': ['Compartment']})],
            species=[
                dict(name='Known entity α', sbo_term='SBO:0000252', compartment_index=0,
                     qualified_reference_context={'is': ['A'], 'hasPart': ['B', 'B']}),
                dict(name='Known entity β', sbo_term=None, compartment_index=None,
                     qualified_reference_context={'is': []}),
            ],
            reactions=[
                dict(reversible_source='false', participants=[
                    dict(species_index=1, role=None, stoichiometry_source=None),
                    dict(species_index=0, role=None, stoichiometry_source=None),
                    dict(species_index=1, role=None, stoichiometry_source=None),
                ]),
                dict(reversible_source='true', participants=[
                    dict(species_index=0, role='reactant', stoichiometry_source='2'),
                ]),
            ],
        ),
        targets=dict(masked_reaction_participants=[
            [dict(role='product', stoichiometry_source='3.0'), dict(role='reactant', stoichiometry_source='0.5'),
             dict(role='modifier', stoichiometry_source=None)],
            None,
        ]),
        audit=dict(selected_reaction_indices=[0], source_member='audit-only'),
    )


def packet(doc=None):
    raw = canonical(document() if doc is None else doc)
    row = dict(source_family=FAMILY, task_name=TASK, native_schema=SCHEMA, source_revision='Reactome98',
               split='train', native_payload=raw, native_payload_sha256=digest(raw),
               condition_context_json='{}', admitted_for_task=True, chemical_holdout_review_pass=True,
               source_annotation_optimization_admitted=True, final_full_mixture_admission=False,
               model_adapter_supported=False)
    return project_reactome_row(row, 'original.parquet', 7)


def axes():
    return ReactomeStaticAxes('a' * 64, 'b' * 64, 1, 5, 10, 300)


def make(width=8, *, physical=False, **kw):
    shared = annotation(width, physical=physical, backbone=response(width).backbone)
    return NativeReactomeStaticModel(shared, axes(), **kw).double().eval()


def test_exact_graph_and_native_occurrence_alignment_with_target_audit_exclusion():
    original = packet()
    changed = document()
    changed['audit']['source_member'] = 'unseen audit contents'
    changed['targets']['masked_reaction_participants'][0][0] = dict(role='reactant', stoichiometry_source='42')
    other = packet(changed)
    assert original.condition == other.condition
    assert original.targets != other.targets
    graph, slots = original.condition.graph()
    assert len(graph.nodes) == 5 and len(graph.edges) == 5
    assert slots == (6, 7, 8)
    assert [(edge.source, edge.target) for edge in graph.edges] == [(1, 0), (2, 3), (1, 3), (2, 3), (1, 4)]
    assert not any(edge.measurements for edge in graph.edges[1:4])
    assert 'hasPart' in graph.nodes[1].measurements[0].value
    assert 'audit-only' not in graph.canonical_bytes().decode()
    assert original.targets == (('product', '3.0'), ('reactant', '0.5'), ('modifier', None))


@pytest.mark.parametrize('change', ('leak-role', 'leak-stoichiometry', 'remove-occurrence', 'bad-mask',
                                    'bad-reference-shape', 'zero-stoichiometry'))
def test_refuses_malformed_native_observations_and_target_leaks(change):
    value = document()
    if change == 'leak-role':
        value['conditions']['reactions'][0]['participants'][0]['role'] = 'product'
    elif change == 'leak-stoichiometry':
        value['conditions']['reactions'][0]['participants'][0]['stoichiometry_source'] = '3.0'
    elif change == 'remove-occurrence':
        value['targets']['masked_reaction_participants'][0].pop()
    elif change == 'bad-mask':
        value['audit']['selected_reaction_indices'] = [1]
    elif change == 'bad-reference-shape':
        value['conditions']['species'][0]['qualified_reference_context']['extra'] = ['BAD']
    else:
        value['targets']['masked_reaction_participants'][0][0]['stoichiometry_source'] = '0'
    with pytest.raises(ValueError):
        packet(value)


def test_role_ce_and_supplied_stoichiometry_log_mse_match_reference_and_gradients():
    model = make(16)
    row = packet()
    result = model._reference_objective((row,))
    role = torch.tensor([ROLE_AXIS.index(x) for x in ('product', 'reactant', 'modifier')])
    expected_role = F.cross_entropy(result['role_logits'][0], role)
    expected_quantity = (result['log_stoichiometry'][0][:2] - torch.tensor(
        [math.log(3), math.log(.5)], dtype=torch.float64,
    )).square().mean()
    torch.testing.assert_close(result['loss'], expected_role + expected_quantity)
    result['log_stoichiometry'][0].retain_grad()
    result['loss'].backward()
    assert result['role_positions'] == 3 and result['supplied_stoichiometry_positions'] == 2
    assert torch.count_nonzero(result['log_stoichiometry'][0].grad[:2]) == 2
    assert result['log_stoichiometry'][0].grad[2] == 0
    assert torch.count_nonzero(model.graph.incidence.weight.grad)
    assert torch.count_nonzero(model.shared.payload.embedding.weight.grad)
    assert model.graph.layers is model.shared.backbone.layers
    assert model.graph.incidence is not model.shared.backbone.incidence


def test_preflight_refuses_complete_batch_before_encoder_and_state_load_recovery():
    model = make(max_attention_elements=500)
    original = packet()
    def no_encoder(*args, **kwargs):
        pytest.fail('model must reject before encoder')
    model.shared.payload.forward = no_encoder
    with pytest.raises(ValueError, match='attention'):
        model((original.condition,) * 2)
    with pytest.raises(ValueError, match='native'):
        model._reference_objective((replace(original, targets=original.targets[:-1]),))
    state = deepcopy(model.state_dict())
    assert all(isinstance(v, (torch.Tensor, dict, str)) for v in state.values())
    state['_extra_state']['topology'] = 'invalid'
    state['role_output.weight'].add_(5)
    before = model.role_output.weight.detach().clone()
    with pytest.raises(ValueError, match='checkpoint'):
        model.load_state_dict(state)
    assert torch.equal(model.role_output.weight, before)


def test_exact_width_family_covers_shared_aliases_and_new_finite_axes():
    members = [make(width) for width in (8, 12, 16)]
    identity = ('context', 'observation')
    ledgers = [parameter_ledger(m, reactome_static_parameter_roles(m, identifier_axis=identity)) for m in members]
    family = width_family(*ledgers)
    for count, model in zip(family.total_numel, members, strict=True):
        assert count == sum(p.numel() for p in model.shared.parameters()) + (2 * 5 + 4) * model.width + 4


def test_exact_physical_annotation_owner_alias_and_complete_roles():
    from mora3.model.physical_trainer import SharedPhysicalModel

    model = make(physical=True)
    assert type(model.shared) is SharedPhysicalModel
    roles = reactome_static_parameter_roles(model, identifier_axis=('context', 'observation'))
    assert set(roles) == {name for name, _ in model.named_parameters(remove_duplicate=False)}
    assert 'shared.physical.method.weight' in roles
