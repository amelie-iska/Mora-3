"""Exact publisher figure table counts, with no inferred organism-level independence."""
from dataclasses import dataclass
from hashlib import md5,sha256
import json
from pathlib import Path
import re

BASE='outputs/data-sources/readiness-20261106-levin-HCN4-original-table-images-v1/'
SOURCE_SHA='8764f61e9c9350e92c0ba673c11de03138fe696e9c6fe79dce4dc724c430b465'
LEDGER_SHA='0b043328055168adfc5bb4a2ea55ca9afe93792edb53be12fe0d219153b26678'
MARKERS=('Xnr-1 (Nodal)','Lefty','Pitx2')
CATEGORIES=('Left','Right','Bilateral','None')
STUDY='doi:10.1242/bio.025957'


def _stamp(p):
    s=p.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


@dataclass(frozen=True)
class Hcn4SourceCountGroup:
    source_table: int
    original_group_ordinal: int
    original_group_label: str
    marker: str
    counts: tuple[int,...]
    source_image_sha256: str

    def conditions(self):
        return dict(species='Xenopus laevis',assay='embryo in situ hybridization expression laterality',
            queried_marker=self.marker,original_experimental_group=self.original_group_label)


class Hcn4SourceCountGroups:
    """Authenticated complete source groups; these rows have no split role."""
    def __init__(self,root):
        root=Path(root).resolve();self._files={}
        def read(path,digest,n=None):
            p=(root/path).resolve();p.relative_to(root)
            before=_stamp(p)
            with p.open('rb') as f:b=f.read(8*2**20+1)
            if len(b)>8*2**20 or (n is not None and len(b)!=n) or sha256(b).hexdigest()!=digest or before!=_stamp(p):
                raise ValueError('exact original source coordinate required')
            self._files[p]=before
            return b
        def ref(d):return read(d['path'],d['sha256'],d['bytes'])
        meta=json.loads(read(BASE+'manifest.json',SOURCE_SHA))
        counts=json.loads(read(BASE+'native-counts-v1/native-observed-laterality-groups.json',LEDGER_SHA))
        if (counts['source_manifest_sha256']!=SOURCE_SHA or counts['original_reported_group_count']!=9
                or counts['literal_count_positions']!=36 or counts['source_roles_assigned'] is not False
                or counts['source_license']!='CC-BY-3.0' or counts['version']!='PMC5665463.1'
                or counts['exact_original_images_reviewed_visually'] is not True
                or counts['independent_OCR_numeric_cells_agree'] is not True):
            raise ValueError('complete visually and independently checked nine original groups required')
        rows=[]
        for i,(r,o) in enumerate(zip(meta['records'],counts['ocr'],strict=True),1):
            if r['version']!='PMC5665463.1' or r['article_license']!='CC-BY-3.0':
                raise ValueError('original source study and grant differ')
            image=read(r['original_path'],r['sha256'],r['bytes'])
            if md5(image).hexdigest()!=r['provider_md5']:raise ValueError('publisher original image checksum mismatch')
            origmeta=json.loads(ref(r['metadata']))
            if origmeta['doi'].lower()!=STUDY[4:] or origmeta['pmcid']!='PMC5665463':
                raise ValueError('exact article metadata identity differs')
            jats=json.loads(ref(r['source_projection']))
            q=jats['qualification']
            if q['expected_PMCID']!='PMC5665463' or q['selected_original_grant_identifiers']!=['CC-BY-3.0'] or q['source_candidate'] is not True:
                raise ValueError('exact original article grant differs')
            original=jats['complete_XML_logical_tree']
            for ordinal in r['article_native_media_coordinate']:
                original=original['children'][ordinal]
            if (original['tag']!='graphic' or original['attributes']['{http://www.w3.org/1999/xlink}href']!=Path(r['original_path']).name):
                raise ValueError('original graphic coordinate differs')
            text=read(BASE+'native-counts-v1/'+o['file'],o['sha256'],o['bytes']).decode('utf-8')
            if not any(re.fullmatch(r'Treatment group\s+Left\s+Right\s+Bilateral\s+None',l) for l in text.splitlines()):
                raise ValueError('complete original category order required')
            labels=('Controls','HCN4-DN 2/2','ZD7288 St 1-10' if i==2 else 'ZD7288 (St 1-10)')
            for j,label in enumerate(labels):
                d=counts['counts'][(i-1)*3+j]
                values=d['original_counts']
                match=[l for l in text.splitlines() if l.startswith(label)]
                if (d['source_table']!=i or d['original_group_ordinal']!=j or d['original_group_label']!=label
                        or tuple(d['ordered_categories'])!=CATEGORIES or d['marker']!=MARKERS[i-1]
                        or d['source_image_sha256']!=r['sha256'] or d['caption_stated_N_literal']!='N=3'
                        or d['native_JATS_graphic_path']!=r['article_native_media_coordinate']
                        or d['original_or_prospective_role'] is not None or d['training_authorized'] is not False
                        or len(values)!=4 or any(type(x) is not int or x<0 for x in values)
                        or len(match)!=1 or list(map(int,match[0][len(label):].split()))!=values):
                    raise ValueError('exact native source group or original counts differ')
                rows.append(Hcn4SourceCountGroup(i,j,label,MARKERS[i-1],tuple(values),r['sha256']))
        if len(rows)!=9 or sum(sum(r.counts) for r in rows)!=371:
            raise ValueError('complete original count source coverage differs')
        self.rows=tuple(rows)

    def require_rows(self,rows):
        rows=tuple(rows)
        if (not rows or any(type(r) is not Hcn4SourceCountGroup or r not in self.rows for r in rows)
                or len({(r.source_table,r.original_group_ordinal) for r in rows})!=len(rows)
                or any(_stamp(p)!=st for p,st in self._files.items())):
            raise ValueError('exact unique authenticated observed count groups required')
        return tuple(r.counts for r in rows)
