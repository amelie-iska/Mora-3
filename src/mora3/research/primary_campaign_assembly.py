"""Pinned preallocation construction-input loading, never model admission.

Reuses exact existing campaign, cohort, PrimaryConfiguration and WidthFamily
validators. No source rows, tensors, models, optimizers, credentials or runtime
callbacks are loaded. Snapshot-bound evidence is opaque to semantic admission:
matching a digest or a snapshot assertion does not establish owner authenticity,
complete source coverage, operator qualification or measured runtime readiness.
"""
from dataclasses import dataclass,fields
from hashlib import sha256
from pathlib import Path
import re

from mora3.data.admet import canonical,_json
from mora3.model.full_owner import PrimaryConfiguration
from mora3.model.parameterization import Axis,ParameterEntry,ParameterLedger,WidthFamily
from mora3.model.mu_optimizer import _checked_family,_family_digest
from .matched_validation import MatchedValidationCohort
from .mup_campaign import FrozenPrimaryProxyCampaign
from .mup_sweep_selection import ProxyTrial,planned_proxy_trials

_EVIDENCE=('source_role_schedule','complete_owner_evidence','runtime_profile','existing_worker')
_MAX_BYTES=16*1024*1024


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str


@dataclass(frozen=True)
class SnapshotEvidence:
    key: str
    reference: ArtifactReference
    source_snapshot_sha256: str


@dataclass(frozen=True)
class PrimaryCampaignInputs:
    request: ArtifactReference
    source_snapshot: ArtifactReference
    source_view_policy: str
    trial: ProxyTrial
    configurations: tuple[tuple[str,PrimaryConfiguration],...]
    family: WidthFamily
    campaign: FrozenPrimaryProxyCampaign
    evidence: tuple[SnapshotEvidence,...]

    @property
    def allocation_authorized(self): return False

    @property
    def unresolved(self):
        return ('semantic_evidence_admission','complete_source_runtime_and_active_branches',
                'executable_route_registry','live_full_owner_family_verification')


def _reference(value):
    if type(value) is not dict or set(value)!={'path','sha256'}:
        raise ValueError('Exact artifact path and SHA256 required')
    name=value['path'];digest=value['sha256']
    if (type(name) is not str or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or str(Path(name))!=name
            or type(digest) is not str or re.fullmatch('[0-9a-f]{64}',digest) is None):
        raise ValueError('Exact normalized repository-relative artifact reference required')
    return ArtifactReference(name,digest)


def _read(root,reference):
    path=root/reference.path
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
        raise ValueError('Pinned regular nonsymlink metadata file required')
    path.resolve().relative_to(root)
    with path.open('rb') as stream: raw=stream.read(_MAX_BYTES+1)
    if len(raw)>_MAX_BYTES or sha256(raw).hexdigest()!=reference.sha256:
        raise ValueError('Pinned bounded artifact digest differs')
    return raw


def _document(raw,*,canonical_required=True):
    try: value=_json(raw)
    except (ValueError,UnicodeError) as error: raise ValueError('JSON metadata required') from error
    if type(value) is not dict or canonical_required and canonical(value)!=raw:
        raise ValueError('Exact canonical metadata object required')
    return value


def _width_family(document):
    def ledger(value):
        if type(value) is not dict or set(value)!={'schema','total_numel','trainable_numel','parameters'}:
            raise ValueError('Exact parameter ledger wire required')
        entries=[]
        for item in value['parameters']:
            if type(item) is not dict or set(item)!={'aliases','shape','axes','role','numel','trainable','dtype'}:
                raise ValueError('Exact parameter entry wire required')
            entries.append(ParameterEntry(tuple(item['aliases']),tuple(item['shape']),
                tuple(Axis(*axis) for axis in item['axes']),item['role'],item['numel'],item['trainable'],item['dtype']))
        return ParameterLedger(tuple(entries),value['total_numel'],value['trainable_numel'],value['schema'])
    if set(document)!={'schema','base','delta','target','widths','total_numel'}:
        raise ValueError('Exact width family wire required')
    try:
        result=WidthFamily(*(ledger(document[key]) for key in ('base','delta','target')),
            tuple(tuple(row) for row in document['widths']),tuple(document['total_numel']),document['schema'])
        return _checked_family(result)
    except (TypeError,KeyError) as error: raise ValueError('Malformed width family metadata') from error


def load_primary_campaign_request(root,path,expected_sha256):
    """Load and cross-bind declared inputs without granting allocation/training.

    All evidence requires a literal source_snapshot_sha256 matching the pinned
    source artifact. Unknown semantic evidence schemas are retained only as
    unresolved references, never accepted as COMPLETE admission. The caller must
    implement reviewed evidence/route/runtime construction before model creation.
    """
    root=Path(root).resolve()
    request_ref=_reference(dict(path=path,sha256=expected_sha256))
    request=_document(_read(root,request_ref))
    if (set(request)!={'schema','source_snapshot','trial_id','campaign','validation','width_family',
            'configurations','evidence','source_view_policy'}
            or request['schema']!='Mora3.PrimaryCampaignConstructionRequest/v1'
            or request['source_view_policy'] not in ('effective-study-v1','effective-study-v2')):
        raise ValueError('Exact versioned primary construction request required')
    source=_reference(request['source_snapshot']);_read(root,source)
    validation_ref=_reference(request['validation']);_read(root,validation_ref)
    validation=MatchedValidationCohort.load(root/validation_ref.path,validation_ref.sha256)
    campaign_ref=_reference(request['campaign']);_read(root,campaign_ref)
    campaign=FrozenPrimaryProxyCampaign.load(root/campaign_ref.path,campaign_ref.sha256,validation=validation)
    if campaign.source_snapshot_sha256!=source.sha256:
        raise ValueError('Campaign/cohort and pinned full source snapshot differ')
    trial=next((t for t in planned_proxy_trials() if t.trial_id==request['trial_id']),None)
    if trial is None: raise ValueError('Exact planned trial identity required')
    family_ref=_reference(request['width_family'])
    family=_width_family(_document(_read(root,family_ref)))
    if (_family_digest(family)!=family_ref.sha256 or family_ref.sha256!=campaign.width_family_sha256
            or family.base.total_numel!=campaign.proxy_parameter_count):
        raise ValueError('Complete declared width family/count/campaign binding differs')
    if type(request['configurations']) is not dict or set(request['configurations'])!={'base','delta','target'}:
        raise ValueError('All three exact width configurations required')
    configurations=[]
    for key in ('base','delta','target'):
        document=_document(_read(root,_reference(request['configurations'][key])))
        if set(document)!={field.name for field in fields(PrimaryConfiguration)}:
            raise ValueError('Complete explicit primary configuration fields required')
        configuration=PrimaryConfiguration(**document)
        if (configuration.source_snapshot_sha256!=source.sha256 or configuration.base_width!=campaign.proxy_width
                or configuration.readout_mult!=float(trial.readout_multiplier)
                or configuration.attn_mult!=float(trial.attention_key_multiplier)):
            raise ValueError('Source, base width or planned trial multiplier differs')
        configurations.append((key,configuration))
    widths=tuple(configuration.width for _,configuration in configurations)
    if (widths[0]!=campaign.proxy_width or not widths[0]<widths[1]<=widths[2]
            or not any(tuple(row[1:])==widths for row in family.widths)):
        raise ValueError('Configuration widths must bind an exact family width axis')
    # Everything except the member width is a fixed source/operator configuration.
    base=configurations[0][1]
    for _,configuration in configurations[1:]:
        if any(getattr(configuration,f.name)!=getattr(base,f.name) for f in fields(base) if f.name!='width'):
            raise ValueError('Non-width primary configuration drift')
    if type(request['evidence']) is not dict or set(request['evidence'])!=set(_EVIDENCE):
        raise ValueError('Complete exact four campaign evidence references required')
    evidence=[]
    for key in _EVIDENCE:
        reference=_reference(request['evidence'][key])
        document=_document(_read(root,reference),canonical_required=False)
        if (reference.sha256!=getattr(campaign,key+'_sha256')
                or document.get('source_snapshot_sha256')!=source.sha256):
            raise ValueError('Evidence digest/full-snapshot binding differs')
        evidence.append(SnapshotEvidence(key,reference,source.sha256))
    return PrimaryCampaignInputs(request_ref,source,request['source_view_policy'],trial,
        tuple(configurations),family,campaign,tuple(evidence))
