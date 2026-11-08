"""Authenticated planarian fragment count marginals from two original workbook sheets."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

NATIVE_PATH='outputs/data-sources/readiness-20261106-levin-planarian-native-counts-and-gene-annotation-v1/manifest.json'
NATIVE_SHA='ee4daae09b5afdfe3973e9cd59747f587373ff10eac22b96ffcc55ad935cd9ae'
ACQUISITION_SHA='3f878eda75fb31bc85bcd13b2dfaee6b9510af2f6366b76c4b6cb148576452db'
WORKBOOK_VERSION='PMC6485777.1'
OUTCOMES=('1H','2H')
SCOPES=('Figure4_empirical_condition_fragment_marginal','Figure7_empirical_round_fragment_marginal')
_CAP=8*2**20


def _stamp(path):
    s=path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


@dataclass(frozen=True)
class PlanarianFragmentCount:
    assay_scope: str
    original_arm: str
    original_fragment: str
    counts: tuple[int,int]
    native_round_label: str | None
    sheet: str
    native_count_coordinates: tuple[str,str]

    def conditions(self):
        # The round label is provenance, as are the coordinates and counts.
        return dict(assay_scope=self.assay_scope,original_arm=self.original_arm,
                    original_fragment=self.original_fragment,endpoint='reported regenerated head classes')

    @property
    def training_authorized(self):
        return False


class PlanarianFragmentCountSource:
    """Whole exact source-only reader. No experimental-unit or split assumption."""

    def __init__(self,root):
        root=Path(root).resolve();self._files={}
        def read(ref):
            p=(root/ref['path']).resolve();p.relative_to(root)
            stamp=_stamp(p)
            with p.open('rb')as f:raw=f.read(_CAP+1)
            if (len(raw)>_CAP or len(raw)!=ref['bytes'] or sha256(raw).hexdigest()!=ref['sha256'] or _stamp(p)!=stamp):
                raise ValueError('exact immutable original planarian source required')
            self._files[p]=stamp
            return raw
        p=(root/NATIVE_PATH).resolve();p.relative_to(root)
        ledger=json.loads(read(dict(path=NATIVE_PATH,bytes=p.stat().st_size,sha256=NATIVE_SHA)))
        if (ledger['schema']!='Mora3.LevinPlanarianOriginalNativeTables/v1'
                or ledger['acquisition_manifest']['sha256']!=ACQUISITION_SHA
                or ledger['no_model_training_or_source_role_assigned']is not True
                or ledger['original_formula_execution']is not False
                or ledger['independent_animals_or_fragment_parent_ids']is not False):
            raise ValueError('exact scoped unassigned original source required')
        archive=json.loads(read(ledger['acquisition_manifest']))
        file,=(f for f in archive['records']if f['version']==WORKBOOK_VERSION)
        read(dict(path=file['original_path'],bytes=file['bytes'],sha256=file['sha256']))
        cell={s['title']:{c['coordinate']:c for c in s['exact_cells']}for s in ledger['original_workbook_sheets']}
        rows=[]
        for g in ledger['raw_observed_fragment_marginal_groups']:
            s=g['sheet'];scope=g['kind'];kind=g['known_condition_coordinates'];xy=tuple(g['coordinates'])
            d=cell[s]
            x,y=(d[c]for c in xy);vals=(x['parsed_value'],y['parsed_value'])
            want={'arm','fragment'} if s=='Figure 4' else {'arm','fragment','round_label'}
            if (scope!=SCOPES[0 if s=='Figure 4' else 1] or set(kind)!=want or tuple(g['outcomes'])!=OUTCOMES
                    or x['formula_literal']is not None or y['formula_literal']is not None
                    or not all(type(n)is int and n>=0 for n in vals) or sum(vals)<=0
                    or vals!=tuple(g['counts']) or g['status']!='observed'
                    or not all(type(d[c]['parsed_value'])is str and d[c]['parsed_value'] for c in kind.values())):
                raise ValueError('complete exact original reported count pair and condition required')
            rows.append(PlanarianFragmentCount(scope,d[kind['arm']]['parsed_value'],d[kind['fragment']]['parsed_value'],
                vals,None if 'round_label'not in kind else d[kind['round_label']]['parsed_value'],s,xy))
        if (len(rows)!=53 or sum(r.assay_scope==SCOPES[0] for r in rows)!=20
                or len({(r.sheet,r.native_count_coordinates)for r in rows})!=53):
            raise ValueError('exact original observed group scope required')
        self._rows=tuple(rows)
        self._issued={id(r):r for r in rows}

    @property
    def rows(self):
        return self._rows

    def require_rows(self,rows):
        rows=tuple(rows)
        if (not rows or len({id(r)for r in rows})!=len(rows)
                or any(type(r)is not PlanarianFragmentCount or self._issued.get(id(r))is not r for r in rows)
                or any(_stamp(p)!=st for p,st in self._files.items())):
            raise ValueError('exact original issued group objects required')
        return tuple(r.counts for r in rows)
