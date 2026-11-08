"""Three tiny CPU SharedPhysicalModel owner widths; zero optimizer updates.

Only three neural instances are constructed. Scalar likelihood checks use
explicit synthetic tensors and establish no dataset or complete-owner admission.
"""
import copy
from dataclasses import replace
import json
import math

import pytest
import torch
from torch.nn import functional as F

from mora3.data.annotation_axes import AnnotationAxes
from mora3.model.physical_trainer import SharedPhysicalModel
from mora3.model.tokengt import TokenGT
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.extensions.scri.contracts import NativeValue, SourceEvidence, StateRecord, TransitionPair
from mora3.extensions.scri.model import (
    SCRITargetAxis, SharedSCRIModel, masked_native_objective,
    scri_parameter_roles, scri_width_family,
)


FEATURES = (('RNA', 'GENE_A', 'native-expression'), ('voltage', 'Vmem', 'mV'))
IDENTIFIERS = ('cell-coordinate-0', 'cell-coordinate-1')


def pair(values=(3.0, None)):
    evidence = SourceEvidence('synthetic', 'r1', 'a'*64, 'fixture#baseline', 'CC0-1.0',
                              'synthetic_fixture', 'study-1', 'sample-1', 'donor-1',
                              'experiment-1', None)
    before = StateRecord('baseline', evidence, 'human', 'cell', 0.0, 'hour', 'treatment',
        (NativeValue(*FEATURES[0][:2], 1.0, FEATURES[0][2], True),
         NativeValue(*FEATURES[1][:2], -60.0, FEATURES[1][2], True)),
        None, None, None, (), None)
    after = replace(before, record_id='outcome', time_value=1.0,
        source=replace(evidence, source_locator='fixture#outcome'), intervention_ids=('perturbation-A',),
        measurements=tuple(NativeValue(modality, name, value, unit, value is not None)
                           for (modality, name, unit), value in zip(FEATURES, values)))
    return TransitionPair(before, after, 'matched_population', 'b'*64)


@pytest.fixture(scope='module')
def models():
    axis = SCRITargetAxis('c'*64, FEATURES, 'gaussian')
    vocabulary = AnnotationVocabulary(json.dumps(dict(learned_hex=[], controls=['pad', 'node', 'edge'],
        certificate={'scope': 'synthetic-unit-fixture-not-fitted-data'}), sort_keys=True))
    registry = AnnotationAxes(json.dumps(dict(
        rows={'GO': [{'axis_id': 'GO:0000001'}], 'EC': [{'axis_id': '1.1.1.1'}]},
        binding={'manifest_sha256': 'e'*64, 'registry_is_admission': False,
                 'scope': 'synthetic-unit-fixture-not-source-verified'}), sort_keys=True))
    result = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026110807)
        for width in (4, 6, 8):
            backbone = TokenGT(width, 2, width=width, heads=2, layers=1,
                scaling_policy='mup-v1', attn_mult=.5,
                initialization_policy='mup-transformer-v1', encoder_var=1.,
                parameterization_policy='mup-tokengt-v1', base_width=4, base_ffn_width=4)
            shared = SharedPhysicalModel.from_registry(vocabulary, registry,
                parameter_axis=('mass',), method_axis=('MALDI',),
                backbone=backbone,
                width=width, heads=2, layers=1, max_tokens=2048,
                parameterization_policy='mup-annotation-v1', base_width=4,
                readout_mult=1.5, attn_mult=0.5)
            before = {name: p.detach().clone() for name, p in shared.named_parameters()}
            model = SharedSCRIModel(shared, axis, max_batch_size=2).double()
            assert type(shared) is SharedPhysicalModel
            assert model.shared is shared and model.shared.backbone is shared.backbone
            assert model.shared.payload is shared.payload
            assert all(torch.equal(before[name].double(), p) for name, p in shared.named_parameters())
            assert sum(p.numel() for p in model.parameters()) <= 32768
            assert all(p.device.type == 'cpu' for p in model.parameters())
            result.append(model)
    return tuple(result)


def test_three_tiny_widths_have_complete_finite_roles(models):
    family = scri_width_family(*models, identifier_axis=IDENTIFIERS)
    print(json.dumps(dict(schema='Mora3.SCRIBoundedNeuralCheck/v1',
        widths=[m.get_extra_state()['width'] for m in models],
        total_unique_parameters=list(family.total_numel),
        identifier_capacity=[m.get_extra_state()['spatial']['identifier_capacity'] for m in models],
        device='cpu', optimizer_updates=0), sort_keys=True))
    assert family.total_numel == tuple(sum(p.numel() for p in m.parameters()) for m in models)
    assert family.total_numel[0] < family.total_numel[1] < family.total_numel[2]
    for model in models:
        roles = scri_parameter_roles(model, identifier_axis=IDENTIFIERS)
        assert set(roles) == set(dict(model.named_parameters(remove_duplicate=False)))
        assert roles['output.weight'].role == 'output'
        assert roles['output.weight'].axes[0].kind == 'finite'
        assert not any(role.role == 'custom' for role in roles.values())
        assert model.get_extra_state()['source_admission_issued'] is False


def test_shared_predictor_excludes_targets_and_audit_ids_and_observed_gradients_flow(models):
    model = models[1]
    first = pair()
    changed = pair((98765.0, -17.0))
    changed = replace(changed, outcome=replace(changed.outcome, record_id='heldout-secret-id',
        source=replace(changed.outcome.source, source_locator='other-outcome', artifact_sha256='d'*64)))
    output = model((first,))
    torch.testing.assert_close(output, model((changed,)), rtol=0, atol=0)
    result = model.objective(output, (first,))
    assert result['observed_targets'] == 1 and result['training_admitted'] is False
    params = tuple(model.parameters())
    gradients = torch.autograd.grad(result['loss'], params, allow_unused=True)
    mapped = dict(zip((name for name, _ in model.named_parameters()), gradients))
    assert mapped['output.weight'].abs().sum() > 0
    assert any(g is not None and g.abs().sum() > 0 for name, g in mapped.items()
               if name.startswith('shared.backbone.'))
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    # Missing second feature affects neither its mean nor its raw-scale row.
    assert torch.equal(mapped['output.weight'][1], torch.zeros_like(mapped['output.weight'][1]))
    assert torch.equal(mapped['output.weight'][3], torch.zeros_like(mapped['output.weight'][3]))


def test_native_readout_scaling_and_condition_bounds(models):
    model = models[-1]
    latent = torch.arange(8, dtype=torch.float64).reshape(1, 8)
    actual = model.output(model.shared._readout_features(latent))
    expected = F.linear(latent*(1.5/(8/4)), model.output.weight, model.output.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='bounded'):
        model((pair(), pair(), pair()))
    wrong = replace(pair().outcome, measurements=(NativeValue('RNA','GENE_A',2.0,'count',True),))
    with pytest.raises(ValueError, match='axis'):
        model((replace(pair(), outcome=wrong),))


def test_checkpoint_preflight_refuses_source_drift_and_nonfinite_before_shared_copy(models):
    model = models[0]
    pristine = copy.deepcopy(model.state_dict())
    model.load_state_dict(copy.deepcopy(pristine))
    for corruption in ('axis', 'nan', 'missing'):
        state = copy.deepcopy(pristine)
        if corruption == 'axis':
            state['_extra_state']['axis']['source_sha256'] = 'f'*64
            state['shared.backbone.feature.weight'].fill_(99)
        elif corruption == 'nan':
            state['output.weight'][0, 0] = float('nan')
        else:
            del state['output.bias']
        with pytest.raises(ValueError, match='checkpoint'):
            model.load_state_dict(state)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in pristine.items()
                   if isinstance(value, torch.Tensor))


def test_all_missing_has_connected_zero_loss_and_zero_denominator():
    axis = SCRITargetAxis('a'*64, FEATURES, 'gaussian')
    output = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    result = masked_native_objective(output, [[None, None]], [[False, False]], axis)
    assert result['observed_targets'] == 0 and result['loss'].item() == 0
    gradient, = torch.autograd.grad(result['loss'], (output,))
    assert torch.equal(gradient, torch.zeros_like(output))
    with pytest.raises(ValueError, match='missing'):
        masked_native_objective(output, [[0, None]], [[False, False]], axis)


def test_gaussian_preserves_native_target_and_unit_likelihood():
    axis = SCRITargetAxis('a'*64, (('voltage', 'Vmem', 'mV'),), 'gaussian')
    result = masked_native_objective(torch.tensor([[1., 0.]], dtype=torch.float64),
                                     [[-60.]], [[True]], axis)
    scale = math.log(2)+1e-6
    expected = 0.5*((-60-1)/scale)**2 + math.log(scale) + 0.5*math.log(2*math.pi)
    assert result['loss'].item() == pytest.approx(expected)


def test_poisson_accepts_only_native_nonnegative_integer_counts():
    axis = SCRITargetAxis('a'*64, (('RNA', 'GENE_A', 'count'),), 'poisson')
    output = torch.zeros((1, 1), dtype=torch.float64)
    result = masked_native_objective(output, [[2]], [[True]], axis)
    assert result['loss'].item() == pytest.approx(1+math.lgamma(3))
    for value in (-1, .5, True, 2**53+1):
        with pytest.raises(ValueError):
            masked_native_objective(output, [[value]], [[True]], axis)
    for unit in ('FPKM', 'TPM', 'log1p-count', 'normalized-count'):
        with pytest.raises(ValueError, match='count units'):
            SCRITargetAxis('a'*64, (('RNA', 'GENE_A', unit),), 'poisson')


def test_categorical_axes_and_masks_are_explicit():
    axis = SCRITargetAxis('a'*64, (('cell', 'fate', 'category'), ('cell', 'state', 'category')),
                          'categorical', (('A', 'B'), ('off', 'on', 'unknown-source-category')))
    output = torch.zeros((1, 5), dtype=torch.float64, requires_grad=True)
    result = masked_native_objective(output, [['A', None]], [[True, False]], axis)
    assert result['observed_targets'] == 1
    assert result['loss'].item() == pytest.approx(math.log(2))
    gradient, = torch.autograd.grad(result['loss'], (output,))
    assert torch.equal(gradient[0, 2:], torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match='unknown native category'):
        masked_native_objective(output, [['not-source-label', None]], [[True, False]], axis)


def test_spatial_graph_uses_neighbor_baseline_excludes_target_and_audit_and_bounds_rank(models):
    from mora3.extensions.scri.spatial import build_neighborhood_query
    model = models[-1]
    first = replace(pair().baseline, coordinates=(0., 0.), coordinate_unit='um',
                    coordinate_frame='native-frame')
    neighbor = replace(first, record_id='neighbor', coordinates=(.5, 0.),
        source=replace(first.source, source_locator='fixture#neighbor'),
        measurements=(replace(first.measurements[0], value=7.), first.measurements[1]))
    def query(rows, query_id):
        return build_neighborhood_query(rows, query_record_id=query_id,
                                         target_axis=FEATURES, radius=1.)
    native = query((first, neighbor), first.record_id)
    output = model.forward_spatial((native,))
    assert output.shape == (1, 4) and torch.isfinite(output).all()
    assert model.get_extra_state()['spatial']['identifier_capacity'] == 2
    audit_only = replace(native, audit={'private_id': 'secret', 'held_out_target': 12345.})
    torch.testing.assert_close(output, model.forward_spatial((audit_only,)), rtol=0, atol=0)
    changed_neighbor = replace(neighbor, measurements=(replace(neighbor.measurements[0], value=987.),
                                                       neighbor.measurements[1]))
    changed = model.forward_spatial((query((first, changed_neighbor), first.record_id),))
    assert not torch.equal(output, changed)
    # Only audit identities change: native baseline biology/geometry is identical.
    renamed = tuple(replace(row, record_id='private-'+str(i),
        source=replace(row.source, source_locator='private#'+str(i), artifact_sha256='d'*64))
                    for i, row in enumerate((first, neighbor)))
    torch.testing.assert_close(output, model.forward_spatial((query(renamed, renamed[0].record_id),)),
                               rtol=0, atol=0)
    # Supplying target truth in a spatial condition object is refused, not encoded.
    target = native.record.nodes[0]
    target = replace(target, measurements=tuple(
        replace(m, value=12345., status='observed', available=True) if m.name == 'target:0' else m
        for m in target.measurements))
    forbidden = replace(native, record=replace(native.record, nodes=(target, *native.record.nodes[1:])))
    with pytest.raises(ValueError):
        model.forward_spatial((forbidden,))
    third = replace(neighbor, record_id='third', coordinates=(0., .5),
                    source=replace(neighbor.source, source_locator='fixture#third'))
    with pytest.raises(ValueError, match='identifier capacity=2'):
        model.forward_spatial((query((first, neighbor, third), first.record_id),))
