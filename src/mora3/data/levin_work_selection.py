"""One preferred eligible source version per independently verified work."""
from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NativeWorkVersionCandidate:
    work_group: str
    version_id: str
    revision_lineage: str
    revision_index: int
    version_kind: str
    qualification: str
    source_body_sha256: str | None
    qualification_evidence_sha256: str

    def validate(self):
        if (any(type(v) is not str or not v for v in (self.work_group,self.version_id,self.revision_lineage))
                or type(self.revision_index) is not int or self.revision_index<1
                or self.version_kind not in ('preprint','published_article')
                or self.qualification not in ('eligible_complete_source','ineligible','unresolved')):
            raise ValueError('exact reviewed work, version, lineage and qualification required')
        for value in (self.source_body_sha256,self.qualification_evidence_sha256):
            if value is None and self.qualification!='eligible_complete_source':continue
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
                raise ValueError('original body and qualification hashes required')
        if self.qualification_evidence_sha256 is None:raise ValueError('explicit proof or unresolved disposition receipt required')


@dataclass(frozen=True)
class NativeWorkVersionSelection:
    work_group: str
    selected_version_id: str | None
    disposition: str
    preserved_candidates: tuple


def select_preferred_work_versions(candidates, *, closed_work_groups):
    """Choose within a closed, externally reviewed version candidate set.

    Later source snapshots can add candidates; no original bytes, source roles,
    or previous manifests are modified. No candidate set is closed by this API.
    """
    rows=tuple(candidates)
    if (not rows or type(closed_work_groups) is not frozenset
            or any(type(v) is not str or not v for v in closed_work_groups)):
        raise ValueError('explicit scoped work candidate set and closed reviewed groups required')
    by_work=defaultdict(list);seen=set();coordinates=set()
    for row in rows:
        if type(row) is not NativeWorkVersionCandidate:raise ValueError('exact typed source version candidate required')
        row.validate();coordinate=(row.work_group,row.version_kind,row.revision_lineage,row.revision_index)
        if row.version_id in seen or coordinate in coordinates:
            raise ValueError('duplicate or conflicting exact source version coordinate')
        seen.add(row.version_id);coordinates.add(coordinate);by_work[row.work_group].append(row)
    if not closed_work_groups<=set(by_work):raise ValueError('closed group has no retained native candidates')
    results=[]
    for group,items in sorted(by_work.items()):
        kept=tuple(sorted((row.version_id for row in items)))
        if group not in closed_work_groups:
            results.append(NativeWorkVersionSelection(group,None,'version_candidate_scope_not_closed',kept));continue
        for kind in ('preprint','published_article'):
            versions=[x for x in items if x.version_kind==kind]
            eligible=[x for x in versions if x.qualification=='eligible_complete_source']
            unresolved=[x for x in versions if x.qualification=='unresolved']
            if not eligible and not unresolved:continue
            if len({x.revision_lineage for x in (*eligible,*unresolved)})>1:
                results.append(NativeWorkVersionSelection(group,None,'multiple_version_lineages_require_review',kept));break
            chosen=max(eligible,key=lambda x:x.revision_index) if eligible else None
            if chosen is None or any(x.revision_index>chosen.revision_index for x in unresolved):
                results.append(NativeWorkVersionSelection(group,None,'preferred_version_qualification_unresolved',kept));break
            results.append(NativeWorkVersionSelection(group,chosen.version_id,
                'preferred_eligible_preprint' if kind=='preprint' else 'eligible_published_article_fallback',kept));break
        else:results.append(NativeWorkVersionSelection(group,None,'no_eligible_complete_body_in_reviewed_scope',kept))
    return tuple(results)
