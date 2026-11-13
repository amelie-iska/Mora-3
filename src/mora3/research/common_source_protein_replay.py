"""Authenticate the special original UniProt protein-prior homology quarantine.

The six other native priors use the base study view. Protein has an additional
ORIGINAL verified train-only quarantine at shard ordinal 24. Omitting that
quarantine changes both the intended original sample selection and the
frozen native adapter/cursor identities. This module reconstructs, without
modifying source assets, exactly the candidate and common29-head scoped views.
No new biological equivalence, train admission, group closure or optimizer step.
"""
from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.finite_view_compatibility import BASE_POLICY
from .primary_campaign_source_rebind import InactiveCommonSourceRebind

_STAGE='outputs/native-common-chemical-quarantine-v6-20261108/'
_POLICY=_STAGE+'bound-route-00-quarantine.json'
_POLICY_SHA='b463ee189a3a39e916bf9ea46c3684496e498f81864ea9e914bfb8f3ff3ecd00'
_INDEX='outputs/v6-freesolv-tox21-train-chemistry-join-20261108-v4/v6-projectors.json'
_INDEX_SHA='dd364e33788e349f5f834618699e0835ef724d098adb4383dcdbee3226eff547'
_ORIGINAL='c47f7a2b55da718c64f9910167830e5667cf4806249889d57eae336747e34ec2'
_KEY='native-prior:protein'
_SHA=re.compile('[0-9a-f]{64}\\Z')


def _read_pin(root,path,expected_sha):
    if _SHA.fullmatch(expected_sha) is None:
        raise ValueError('Exact published source pin required')
    candidate=root/path
    if (not candidate.is_file() or candidate.is_symlink()
            or candidate.resolve()!=candidate or not candidate.is_relative_to(root)):
        raise ValueError('Exact nonsymlink original repository asset required')
    raw=candidate.read_bytes()
    if len(raw)>128_000 or sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Published quarantine source/policy bytes differ')
    try:return json.loads(raw)
    except (ValueError,UnicodeError) as exc:raise ValueError('Complete original quarantine JSON required')from exc


@dataclass(frozen=True)
class ProvenProteinQuarantine:
    candidate: FiniteUnionNativeView
    common: FiniteUnionNativeView
    quarantined_coordinates: frozenset
    original_source_sha256: str
    original_payload_sha256: str
    original_policy_sha256: str
    source_original_row_ordinal: int
    original_role: str
    original_native_task: str
    opaque_selected_identity_sha256: str
    training_admission: bool=False


def reconstruct_protein_quarantine_views(root,*,base_view,context,row):
    """Recreate original one-row context quarantine with exact source proof.

    The policy's opaque selected-identity digest is retained, never replaced by
    a guessed source ID or interpreted as an experimental biological identity.
    The original source shard, row role, record bytes and payload identity are
    separately checked against the actual immutable Parquet row.
    """
    import pyarrow.parquet as pq
    root=Path(root).resolve()
    if (type(base_view)is not FiniteUnionNativeView
            or type(context)is not InactiveCommonSourceRebind or type(row)is not dict
            or row.get('task_key')!=_KEY):
        raise ValueError('Exact original protein task, live base view and pinned common context required')
    from .primary_campaign_source_rebind import load_inactive_common_source_rebind
    if context!=load_inactive_common_source_rebind(root):
        raise ValueError('Original same-snapshot common route authority changed')
    from .primary_source_factory_preflight import document
    actual=document(root,context.evidence)
    matching=[entry for entry in actual['route_bindings']if entry['task_key']==_KEY]
    if len(matching)!=1 or row!=matching[0]:
        raise ValueError('Original protein source/quarantine/adapter commitment was altered')
    if finite_view_policy_sha256(base_view)!=BASE_POLICY:
        raise ValueError('Protein quarantine must extend the exact native base study view')
    index=_read_pin(root,_INDEX,_INDEX_SHA)
    rows=[x for x in index.get('bound_policies',())if x.get('task_key')==_KEY]
    if (len(rows)!=1 or rows[0]!={'task_key':_KEY,'file':Path(_POLICY).name,'sha256':_POLICY_SHA}
            or index.get('source_snapshot_sha256')!=_ORIGINAL):
        raise ValueError('Original protected protein homology policy index differs')
    policy=_read_pin(root,_POLICY,_POLICY_SHA)
    if (set(policy)!={'schema','excluded_selected_identity_sha256','inherited_freesolv_policy',
            'original_roles_preserved','original_source_masks_preserved','original_train_coordinates',
            'parent_native_view_policy_sha256','protected_chain_alignment_seal_sha256',
            'source_snapshot_sha256','task_key','training_authorized'}
            or policy['schema']!='Mora3.InactiveRouteScopedNativeContextQuarantine/v1'
            or policy['task_key']!=_KEY or policy['source_snapshot_sha256']!=_ORIGINAL
            or policy['parent_native_view_policy_sha256']!=BASE_POLICY
            or policy['inherited_freesolv_policy'] is not None
            or policy['original_roles_preserved'] is not True
            or policy['original_source_masks_preserved'] is not True
            or policy['training_authorized'] is not False
            or len(policy['original_train_coordinates'])!=1
            or len(policy['excluded_selected_identity_sha256'])!=1):
        raise ValueError('Nonadditional or unreviewed original protein source policy refused')
    selected=policy['excluded_selected_identity_sha256'][0]
    if any(_SHA.fullmatch(value)is None for value in (selected,policy['protected_chain_alignment_seal_sha256'])):
        raise ValueError('Opaque source identity/chain evidence requires exact digest')
    coordinate=policy['original_train_coordinates'][0]
    if (type(coordinate)is not list or len(coordinate)!=5 or type(coordinate[0])is not str
            or type(coordinate[2])is not int or coordinate[2]!=24
            or any(_SHA.fullmatch(x)is None for x in (coordinate[1],coordinate[3],coordinate[4]))):
        raise ValueError('Exact single original source homology-coordinate certificate required')
    path,source_digest,ordinal,identity_digest,payload_digest=coordinate
    refs={d['source_shard']['path']:d['source_shard'] for d in base_view.descriptors}
    if (path not in refs or refs[path]['sha256']!=source_digest
            or (path,ordinal) in base_view._ordinal_exclusions):
        raise ValueError('Quarantine coordinate must be inside the same original eligible unexcluded source')
    shard=root/path
    if (not shard.is_file() or shard.is_symlink() or shard.resolve()!=shard
            or not shard.is_relative_to(root) or shard.stat().st_size>2_000_000):
        raise ValueError('Exact small original UniProt Parquet shard required')
    if sha256(shard.read_bytes()).hexdigest()!=source_digest:
        raise ValueError('Original native protein shard changed')
    pf=pq.ParquetFile(shard)
    if pf.metadata.num_rows!=refs[path]['rows'] or pf.metadata.num_rows>2048:
        raise ValueError('Bounded original protein shard row axis differs')
    columns=('split','source_family','source_revision','task_name','native_schema',
             'native_payload_sha256','admitted_for_task')
    table=pf.read(columns=list(columns),use_threads=False)
    actual=table.slice(ordinal,1).to_pylist()[0]
    if (actual['split']!='train' or actual['source_family']!='UniProt'
            or actual['task_name']!='protein_sequence_modeling'
            or actual['source_revision']!='UniProt2026_03'
            or actual['native_schema']!='mora3.CorpusTask/v1'
            or actual['admitted_for_task'] is not True
            or actual['native_payload_sha256']!=payload_digest):
        raise ValueError('Original selected protein row identity/role/payload differs from protected source')
    extra=frozenset(((path,ordinal),))
    if len(extra)!=1 or extra & base_view._ordinal_exclusions:
        raise ValueError('Quarantine must add exactly one distinct excluded original train ordinal')
    candidate_snapshot='1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733'
    if (context.common_snapshot.sha256!='79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'
            or set(row)!={'candidate','common','original','task_key','scheduled_updates',
                           'common_projection','common_target_free_trace','common_view_policy_sha256'}
            or row['scheduled_updates']!=82 or row['common_projection'] is not None
            or row['common_target_free_trace'] is not None):
        raise ValueError('Exact reviewed protein candidate/common source-snapshot chain required')
    views={}
    for label,snapshot,digest in (('candidate',candidate_snapshot,
                                     '4f144c456c46a393e678d640cc96c99e4d4b7af793174b50b49971088fb8f227'),
                                  ('common',context.common_snapshot.sha256,
                                     '2888d9f11038f70f9b25a049c8a006d87ca9918de6e94adcb71ec5ee1c9fd34b')):
        expected=row.get(label)
        if (type(expected)is not dict or set(expected)!={'adapter_sha256','initial_cursor_sha256',
                 'max_windows','task_key'} or expected['task_key']!=_KEY or expected['max_windows']!=2
                or any(_SHA.fullmatch(expected[k])is None for k in ('adapter_sha256','initial_cursor_sha256'))):
            raise ValueError('Bound common protein train cursor/adapter metadata required')
        original=copy(base_view)
        original.effective_filter=dict(policy,source_snapshot_sha256=snapshot)
        original._ordinal_exclusions=base_view._ordinal_exclusions|extra
        if (original.manifest is not base_view.manifest
                or original.descriptors is not base_view.descriptors
                or original._sealed_by_path is not base_view._sealed_by_path
                or finite_view_policy_sha256(original)!=digest):
            raise ValueError('Reconstructed original quarantine policy does not match the frozen scoped digest')
        views[label]=original
    if (row['common_view_policy_sha256']!=finite_view_policy_sha256(views['common'])
            or finite_view_policy_sha256(base_view)!=BASE_POLICY):
        raise ValueError('Protein common view cannot be substituted with the unquarantined base view')
    return ProvenProteinQuarantine(views['candidate'],views['common'],extra,
        source_digest,payload_digest,_POLICY_SHA,ordinal,'train','protein_sequence_modeling',identity_digest)


def audit_common_source_protein_two_windows(root,*,task_key=_KEY,max_windows=2):
    """Bounded genuine two-original-window replay, never training admission."""
    if type(task_key)is not str or task_key!=_KEY or type(max_windows)is not int or max_windows!=2:
        raise ValueError('Exact protected protein prior and two-window scan budget required')
    from .common_source_native_replay import _audit_common_source_two_prior_windows
    return _audit_common_source_two_prior_windows(root,task_key=task_key,max_windows=max_windows)
