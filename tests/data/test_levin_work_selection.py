import pytest
from mora3.data.levin_work_selection import NativeWorkVersionCandidate,select_preferred_work_versions


def row(version,kind,lineage,index,qualification='eligible_complete_source'):
    return NativeWorkVersionCandidate('reviewed-work',version,lineage,index,kind,qualification,
        'a'*64 if qualification=='eligible_complete_source' else None,'b'*64)


def test_exact_preprint_preference_and_nonqualified_fallback():
    journal=row('published','published_article','publisher',1)
    p1=row('preprint-v1','preprint','preprint-server',1)
    p2=row('preprint-v2','preprint','preprint-server',2)
    closed=frozenset({'reviewed-work'})
    result,=select_preferred_work_versions((journal,p1,p2),closed_work_groups=closed)
    assert result.selected_version_id=='preprint-v2' and len(result.preserved_candidates)==3
    blocked=row('preprint-v1','preprint','preprint-server',1,'ineligible')
    result,=select_preferred_work_versions((journal,blocked),closed_work_groups=closed)
    assert result.selected_version_id=='published' and result.disposition=='eligible_published_article_fallback'


def test_pending_source_or_conflicting_versions_not_selected():
    journal=row('published','published_article','publisher',1)
    uncertain=row('preprint-v2','preprint','preprint-server',2,'unresolved')
    closed=frozenset({'reviewed-work'})
    for items,groups in [((journal,uncertain),closed),((journal,),frozenset()),
                         ((row('a','preprint','a',1),row('b','preprint','b',3)),closed)]:
        result,=select_preferred_work_versions(items,closed_work_groups=groups)
        assert result.selected_version_id is None
    with pytest.raises(ValueError,match='duplicate'):
        select_preferred_work_versions((journal,journal),closed_work_groups=closed)
