"""Exact ordered LinCx Table S1 docking scores; not electrical conductance or intervention advice."""
from dataclasses import dataclass
from hashlib import md5,sha256
import json
from pathlib import Path
import re
import struct

BASE='outputs/data-sources/readiness-20261106-lincx-original-published-supplements-v1/'
M=BASE+'native-FETCH-S1-v1/manifest.json'
SOURCE_SHA='294b0d62d948e55b63030d59efd7554d7bc7b0f9e8092810af2b5afc49d84e1c'
PANELS=('Cx34.7','Cx35','Cx34.7 - Cx35')
COUNTS=(71,67,56)
_ALLELE=re.compile(r'(?:[A-Z][1-9][0-9]*[A-Z]|WT)\Z')
_NUMBER=re.compile(r'[0-9]+\.[0-9][0-9]\Z')


def _stamp(p):
    s=p.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


@dataclass(frozen=True)
class NativeFetchGroup:
    page: int
    native_row: int
    panel: str
    known_ordered_pair: tuple
    reported_mean_lexeme: str
    reported_se_lexeme: str

    def conditions(self):
        return self.known_ordered_pair,self.panel


class LinCxNativeFetchSource:
    def __init__(self,root):
        root=Path(root).resolve();self._files={}
        def read(path,digest,size=None):
            p=(root/path).resolve();p.relative_to(root);st=_stamp(p)
            with p.open('rb') as f:raw=f.read(8*2**20+1)
            if len(raw)>8*2**20 or (size is not None and len(raw)!=size) or sha256(raw).hexdigest()!=digest or st!=_stamp(p):
                raise ValueError('exact immutable native supplement source required')
            self._files[p]=st;return raw
        m=json.loads(read(M,SOURCE_SHA));orig=m['exact_original_PDF']
        if (m['reported_observation_groups']!=sum(COUNTS) or m['complete_body_pages']!=[2,3]
                or m['one_study_no_split_roles'] is not True or m['training_authorized'] is not False
                or orig['version']!='PMC13372691.1' or orig['article_license']!='CC-BY-4.0'
                or m['native_panel_counts']!=dict(zip(PANELS,COUNTS))):
            raise ValueError('exact original native Table S1 coverage, license and role required')
        b=read(orig['original_path'],orig['sha256'],orig['bytes'])
        if md5(b).hexdigest()!=orig['provider_md5'] or not b.startswith(b'%PDF-'):
            raise ValueError('whole original publisher PDF transport mismatch')
        for name,r in m['proof_files'].items():
            read(BASE+'native-FETCH-S1-v1/'+name,r['sha256'],r['bytes'])
        rows=[]
        for d in m['native_groups']:
            panel=d['native_panel_label'];pi=PANELS.index(panel)
            pair=tuple(tuple(x)for x in d['native_ordered_pair'])
            if (d['panel_ordinal']!=pi or len(pair)!=2 or any(len(q)!=2 for q in pair)
                    or tuple(x[0]for x in pair)!=(('Cx34.7','Cx35')if pi==2 else (panel,panel))
                    or any(not _ALLELE.fullmatch(x[1])for x in pair) or pi!=2 and pair[0]!=pair[1]
                    or d['original_page_1based'] not in (2,3) or type(d['visual_row_ordinal'])is not int
                    or d['visual_row_ordinal']<0 or d['original_or_prospective_role'] is not None
                    or d['training_authorized'] is not False):
                raise ValueError('known exact ordered assay pair and source role required')
            mean,se=d['original_mean_lexeme'],d['original_standard_error_lexeme']
            if any(not _NUMBER.fullmatch(s) or struct.pack('>d',float(s)).hex()!=d[key]
                    for s,key in ((mean,'mean_float64_bits'),(se,'standard_error_float64_bits'))):
                raise ValueError('native reported mean or standard error bits differ')
            rows.append(NativeFetchGroup(d['original_page_1based'],d['visual_row_ordinal'],panel,pair,mean,se))
        if (len(rows)!=sum(COUNTS) or len({r.conditions() for r in rows})!=len(rows)
                or len({(r.page,r.native_row,r.panel)for r in rows})!=len(rows)):
            raise ValueError('complete unique original observed FETCH group table required')
        self.rows=tuple(rows)

    def require_rows(self,rows):
        rows=tuple(rows)
        if (not rows or any(type(r) is not NativeFetchGroup or r not in self.rows for r in rows)
                or len({(r.page,r.native_row,r.panel)for r in rows})!=len(rows)
                or any(_stamp(p)!=st for p,st in self._files.items())):
            raise ValueError('exact unique complete native source rows required')
        return tuple(float(r.reported_mean_lexeme) for r in rows)
