"""Original-grant-checked Levin JATS candidates, without version or role admission."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

BASE='outputs/data-sources/'
FIRST=(BASE+'readiness-20261106-levin-exact-author-native-jats-v2/manifest.json',
 'a2d7bb851949b2605de4d6717123927a6a7cb1ff0bc33a8424c27d7202fd460e')
SECOND=(BASE+'readiness-20261106-levin-lab-byline-native-jats-v1/manifest.json',
 'fed91a14a237d4854037b65f5c493188bfe05ddecc4964a34a906907ffbf0afd')
ORACLE=(BASE+'readiness-20261106-levin-native-jats-independent-text-and-screen-v1/manifest.json',
 'cf5ba4080501c2ef91075ec308fdc34db27e6813aaa04240ff38f4db09db1346')
GROUPS=('outputs/data-intake/readiness-20261106-corpus-continuation/levin-exact-work-data-exposure-components-v2/manifest.json',
 '4686c60de60fcdeb22d69101b06ac10768177d8d9669bfb76a16e8693f2f5425')
GRANTS={'CC-BY-3.0','CC-BY-4.0','CC0-1.0'}


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _stamp(p):
    st=p.stat()
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns


@dataclass(frozen=True)
class LevinJatsTextCandidate:
    provider_version: str
    original_sha256: str
    target_utf8: bytes
    selected_original_grants: tuple
    complete_native_character_slots: int

    def condition_inputs(self):
        return dict(task='source_language_prior',language='en',domain='biomedical')

    @property
    def training_authorized(self):
        return False


class LevinJatsTextCandidateSource:
    """Read complete exact ordered native character data, never choose a preprint/paper."""
    def __init__(self,root):
        self._root=Path(root).resolve();self._refs={};self._issued={}
        first,second,oracle,groups=(json.loads(self._read(*pin)) for pin in (FIRST,SECOND,ORACLE,GROUPS))
        self._oracle={r['version']:r for r in oracle['records']}
        extra=json.loads(self._read_ref(groups['new_native_body_candidate_joins']))
        self._extra={r['native_version']:r for r in extra}
        self._items=tuple(r for m in (first,second) for r in m['records'] if r['status']=='full_native_body_candidate')
        versions={r['version'] for r in self._items}
        if len(self._items)!=60 or len(self._oracle)!=55 or len(self._extra)!=5 or len(versions)!=60:
            raise ValueError('complete exact frozen original Levin JATS candidates required')
        if versions!=self._oracle.keys()|self._extra.keys() or self._oracle.keys()&self._extra.keys():
            raise ValueError('each original JATS candidate must have an independent scope')
        if any(r['prospective_role'] is not None or r['training_authorized'] is not False for r in extra):
            raise ValueError('unassigned source role is required in candidate view')

    def _read(self,path,digest,size=None):
        p=(self._root/path).resolve();p.relative_to(self._root)
        st=_stamp(p)
        if p in self._refs and st!=self._refs[p]:raise ValueError('source file identity drift')
        with p.open('rb') as f:raw=f.read(32*2**20+1)
        if len(raw)>32*2**20 or (size is not None and len(raw)!=size) or _sha(raw)!=digest or _stamp(p)!=st:
            raise ValueError('exact bounded immutable source artifact required')
        self._refs[p]=st
        return raw

    def _read_ref(self,ref):
        return self._read(ref['path'],ref['sha256'],ref['bytes'])

    def iter_candidates(self):
        for item in self._items:
            version=item['version'];pmc=item['PMCID']
            if re.fullmatch(pmc+r'\.[1-9]\d*',version) is None:
                raise ValueError('exact versioned native PMCID required')
            raw=self._read_ref(item['original'])
            if hashlib.md5(raw).hexdigest()!=item['publisher_md5_verified']:
                raise ValueError('native publisher checksum mismatch')
            proj=json.loads(self._read_ref(item['native_projection']))
            qual=proj['qualification']
            rights=tuple(qual['selected_original_grant_identifiers'])
            if (qual['expected_PMCID']!=pmc or qual['source_candidate'] is not True or qual['language_qualified'] is not True
                    or qual['rights_qualified'] is not True or not rights or not set(rights)<=GRANTS
                    or proj['original_source_ref']!=item['original']):
                raise ValueError('exact original body version, language and grant required')
            slots=proj['native_character_data_segments']
            target=b''.join(x['text_exact_native'].encode('utf-8') for x in slots)
            if version in self._oracle:
                proof=self._oracle[version]
                if (proof['complete_native_character_slots']!=len(slots)
                        or proof['full_text_sha256']!=_sha(target) or proof['target_utf8_bytes']!=len(target)
                        or proof['original']!=item['original'] or proof['native_projection']!=item['native_projection']):
                    raise ValueError('native character-data target differs from independent exact original oracle')
            else:
                proof=self._extra[version]
                if proof['exact_original_body_candidate']!=item['original'] or proof['native_projection']!=item['native_projection']:
                    raise ValueError('native original and projection differ from independent exact lab/byline proof')
            if not target:raise ValueError('source candidate has no complete native character target')
            row=LevinJatsTextCandidate(version,item['original']['sha256'],target,rights,len(slots))
            signature=(row.original_sha256,_sha(target),rights,len(slots))
            if self._issued.setdefault(version,signature)!=signature:
                raise ValueError('issued source target changed')
            yield row

    def require_candidate(self,row):
        if (type(row) is not LevinJatsTextCandidate or type(row.target_utf8) is not bytes
                or self._issued.get(row.provider_version)!=(row.original_sha256,_sha(row.target_utf8),
                    row.selected_original_grants,row.complete_native_character_slots)):
            raise ValueError('exact unaltered issued full source candidate required')
        if any(_stamp(p)!=st for p,st in self._refs.items()):
            raise ValueError('original source identity drift')
