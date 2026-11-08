"""Pinned scoped construction from a primary request to the exact 62-route binder."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from mora3.data.mcsa_role_list_admission import TASK as MCSA
from mora3.research import combined60_binding as legacy
from mora3.research import combined62_binding as combined
from mora3.research import finite59_assembly as finite
from mora3.research import source_text_campaign as text
from mora3.research.matched_validation import MatchedValidationCohort
from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign
from mora3.research.mup_sweep_selection import MAX_UPDATES
from mora3.research.primary_campaign_assembly import (
    ArtifactReference, PrimaryCampaignInputs, _document, _read, _reference,
    load_primary_campaign_request,
)


_SCHEMA = 'Mora3.PrimaryCampaignRouteConstruction62/v1'
_COHORT_FIELDS = {'validation', 'validation_adapter_sha256', 'train_routes', 'task_cycle'}
_TEXT_KEYS = {family: f'reviewed-native-text:{family}:{spec[0]}' for family, spec in text.SPECS.items()}
_FINITE_KEYS = set(finite.AXES)
_LEGACY_KEYS = _FINITE_KEYS | {MCSA}
_ROUTE_KEYS = _LEGACY_KEYS | set(_TEXT_KEYS.values())


@dataclass(frozen=True)
class ScopedPrimaryCampaign62:
    inputs: PrimaryCampaignInputs
    route_request: ArtifactReference
    routes: combined.Combined62

    @property
    def unresolved(self):
        return tuple(item for item in self.inputs.unresolved if item != 'executable_route_registry') + (
            'global_split_and_evaluation_admission', 'measured_runtime_window_capacity',
        )

    @property
    def allocation_authorized(self): return False

    @property
    def training_authorized(self): return False

    @property
    def complete_campaign(self): return False


def _object(value, keys, message):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(message)
    return value


def _budget(value, cls):
    _object(value, (field.name for field in fields(cls)), f'Exact {cls.__name__} fields required')
    return cls(**value)


def _group(value, keys, cls):
    _object(value, keys, f'Exact {cls.__name__} route keys required')
    return {key: _budget(item, cls) for key, item in value.items()}


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'Explicit positive {name} required')
    return value


def _load_prior(root, document):
    _object(document, ('campaign', 'validation'), 'Prior campaign and validation artifact pins required')
    cref = _reference(document['campaign'])
    vref = _reference(document['validation'])
    _read(root, cref)
    _read(root, vref)
    cohort = MatchedValidationCohort.load(root / vref.path, vref.sha256)
    return FrozenPrimaryProxyCampaign.load(root / cref.path, cref.sha256, validation=cohort), asdict(cref)


def _checked_chain(inputs, finite_campaign, legacy_campaign):
    final = inputs.campaign
    if inputs.source_view_policy != 'effective-study-v2':
        raise ValueError('Only the exact effective-study-v2 route policy is supported')
    for previous in (finite_campaign, legacy_campaign):
        if any(getattr(previous, field.name) != getattr(final, field.name)
               for field in fields(final) if field.name not in _COHORT_FIELDS):
            raise ValueError('Prior campaign snapshot, width, optimizer or evidence identity changed')
    for campaign, expected in ((finite_campaign, _FINITE_KEYS), (legacy_campaign, _LEGACY_KEYS),
                               (final, _ROUTE_KEYS)):
        if {plan.task_key for plan in campaign.train_routes} != expected:
            raise ValueError('Exact nested 59, 60 and 62 route axes required')
    for previous, next_campaign in ((finite_campaign, legacy_campaign), (legacy_campaign, final)):
        plans = {p.task_key: p for p in next_campaign.train_routes}
        if any(plans[p.task_key] != p for p in previous.train_routes):
            raise ValueError('Prior route plan identity or budget changed')
    expected_validation = {finite.INVERSE, legacy.FREE, MCSA}
    for campaign, expected in ((finite_campaign, {finite.INVERSE}),
                               (legacy_campaign, expected_validation),
                               (final, expected_validation | set(_TEXT_KEYS.values()))):
        if {task.task_key for task in campaign.validation.tasks} != expected:
            raise ValueError('Exact nested one, three and five validation scopes required')


def _checked_budgets(document, campaign):
    budget = _object(document, ('prior', 'reactzyme', 'assay', 'pdb', 'uniprot', 'annotation',
                                'rfam', 'goa', 'rosetta', 'remaining', 'mcsa', 'text'),
                     'Exact explicit source constructor budgets required')
    typed = {
        'prior': _group(budget['prior'], finite.prior.ROUTES, finite.prior.PriorBudget),
        'reactzyme': _budget(budget['reactzyme'], finite.reaction.ReactZymeBudget),
        'assay': _group(budget['assay'], finite.assay.ROUTES, finite.assay.AssayBudget),
        'pdb': _budget(budget['pdb'], finite.pdb.PDBRouteBudget),
        'uniprot': _group(budget['uniprot'], finite.uniprot.ROUTES, finite.uniprot.FullRecordBudget),
        'annotation': _group(budget['annotation'], finite.annotation.ROUTES, finite.annotation.AnnotationRouteBudget),
        'rfam': _group(budget['rfam'], finite.rg.RFAM_KEYS, finite.rg.RfamBudget),
        'goa': _budget(budget['goa'], finite.rg.GOABudget),
        'rosetta': _group(budget['rosetta'], finite.rosetta.KEYS, finite.rosetta.RosettaBudget),
        'remaining': _group(budget['remaining'], finite.remaining.KEYS, finite.remaining.RouteBudget),
        'mcsa': _budget(budget['mcsa'], legacy.mcsa.MCSAFinitePassPolicy),
        'text': _group(budget['text'], _TEXT_KEYS, text.TextBudget),
    }
    for group, module in (('prior', finite.prior), ('assay', finite.assay), ('uniprot', finite.uniprot),
                          ('annotation', finite.annotation), ('rosetta', finite.rosetta), ('remaining', finite.remaining)):
        module.checked_budgets(typed[group])
    finite.rg.checked_budgets(typed['rfam'], typed['goa'])
    windows = {f'native-prior:{key}': item.max_windows for key, item in typed['prior'].items()}
    for key in ('assay', 'uniprot', 'annotation', 'rfam', 'rosetta', 'remaining'):
        windows.update({route: item.max_windows for route, item in typed[key].items()})
    windows.update({finite.reaction.TASK: typed['reactzyme'].max_windows,
                    finite.pdb.TASK: typed['pdb'].max_windows, finite.rg.GOA_TASK: typed['goa'].max_windows})
    if windows != {p.task_key: p.max_windows for p in campaign.train_routes}:
        raise ValueError('Explicit source window budgets differ from the finite59 plan')
    return typed


def construct_scoped_primary_campaign62(root, path, expected_sha256):
    """Read exact metadata, reconstruct fixed source factories, and bind 62 routes.

    The companion request pins the existing primary request, two prior frozen
    campaign/cohort pairs, and explicit non-derived source budgets/resources.
    Existing binders own source-policy, runtime identity and target checks.
    """
    root = Path(root).resolve()
    route_ref = _reference(dict(path=path, sha256=expected_sha256))
    route_raw = _read(root, route_ref)
    doc = _object(_document(route_raw), ('schema', 'primary_request', 'finite59', 'combined60',
                                         'budgets', 'pdb_resource_binding_sha256',
                                         'rfam_goa_budget_binding_sha256', 'max_selected_text_bytes'),
                  'Exact scoped 62-route construction request required')
    if doc['schema'] != _SCHEMA:
        raise ValueError('Unknown scoped route construction request schema')
    ref = _reference(doc['primary_request'])
    primary_raw = _read(root, ref)
    primary = _document(primary_raw)
    inputs = load_primary_campaign_request(root, ref.path, ref.sha256)
    prior, prior_ref = _load_prior(root, doc['finite59'])
    sixty, sixty_ref = _load_prior(root, doc['combined60'])
    _checked_chain(inputs, prior, sixty)
    budgets = _checked_budgets(doc['budgets'], prior)
    text_bytes = _positive(doc['max_selected_text_bytes'], 'selected text byte bound')
    pdb_sha = doc['pdb_resource_binding_sha256']
    rfam_goa_sha = doc['rfam_goa_budget_binding_sha256']
    for expected in (pdb_sha, rfam_goa_sha):
        _reference(dict(path='resource-binding', sha256=expected))
    source = asdict(inputs.source_snapshot)
    base = dict(source_snapshot=source, expected_view_policy_sha256=finite.BASE_POLICY)
    pdb = finite.pdb.build_original_pdb_registry(root, budget=budgets['pdb'], **base)
    if pdb.resource_binding_sha256 != pdb_sha:
        raise ValueError('Constructed PDB resource binding differs from the pinned request')
    rfam_goa = finite.rg.build_rfam_goa_registry(root, source_snapshot=source,
                                                rfam_budgets=budgets['rfam'], goa_budget=budgets['goa'])
    if rfam_goa.budget_binding_sha256 != rfam_goa_sha:
        raise ValueError('Constructed Rfam/GOA budget binding differs from the pinned request')
    finite_inputs = dict(
        campaign=prior, campaign_reference=prior_ref, source_snapshot=source,
        prior_result=finite.prior.build_native_prior_registry(root, budgets=budgets['prior'], **base),
        reactzyme_result=finite.reaction.build_original_reactzyme_registry(root, budget=budgets['reactzyme'], **base),
        assay_result=finite.assay.build_original_assay_registry(root, budgets=budgets['assay'], **base),
        pdb_result=pdb, pdb_resource_binding_sha256=pdb_sha,
        uniprot_result=finite.uniprot.build_full_record_registry(root, source_snapshot=source,
            expected_view_policy_sha256=finite.INVERSE_POLICY, budgets=budgets['uniprot']),
        annotation_result=finite.annotation.build_original_annotation_registry(root, budgets=budgets['annotation'], **base),
        rfam_goa_result=rfam_goa, rfam_goa_budget_binding_sha256=rfam_goa_sha,
        rosetta_result=finite.rosetta.build_rosetta_registry(root, source_snapshot=source, budgets=budgets['rosetta']),
        remaining_result=finite.remaining.build_remaining_eight_registry(root, source_snapshot=source,
                                                                        budgets=budgets['remaining']),
    )
    plan_windows = {p.task_key: p.max_windows for p in inputs.campaign.train_routes}
    mcsa = legacy.mcsa.build_mcsa_bridge(root, source_snapshot=source, pass_policy=budgets['mcsa'],
        max_windows=plan_windows[MCSA], task_cycle=sixty.task_cycle, total_updates=MAX_UPDATES,
        max_validation_bytes=sixty.validation.max_selected_arrow_bytes)
    text_routes = text.build_reviewed_text_routes(root, source_snapshot=source, budgets=budgets['text'])
    result = combined.bind_combined62(
        root, combined60_inputs=dict(finite_inputs=finite_inputs, mcsa_bridge=mcsa,
                                     campaign=sixty, campaign_reference=sixty_ref),
        text_routes=text_routes, text_scan_limits={key: plan_windows[key] for key in _TEXT_KEYS.values()},
        max_selected_text_bytes=text_bytes, campaign=inputs.campaign, campaign_reference=primary['campaign'],
    )
    if _read(root, route_ref) != route_raw or _read(root, ref) != primary_raw:
        raise ValueError('Pinned primary or scoped route construction request changed')
    return ScopedPrimaryCampaign62(inputs, route_ref, result)
