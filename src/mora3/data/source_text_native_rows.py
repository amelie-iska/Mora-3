"""Exact source-text views; conditions, full targets and native audit stay separate.

No source role is invented. These readers never confer corpus/training approval.
Source code, HTML, XML links and SQL are inert text. No normalization, separator
insertion, root inference, source-title guessing or cross-provider verse zip.
"""
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

FAMILIES = ('INV-081','INV-080','INV-082/OSHB','INV-084','INV-083')
WLC_LINEAGE = 'source-lineage:http://www.tanach.us/Tanach.xml'


def _wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')


def _text(value):
    if type(value) is not str:
        raise ValueError('exact native source string required')
    try:return value.encode('utf8')
    except UnicodeError as exc:raise ValueError('native UTF8 codepoints required') from exc


def _identity(path):
    s=Path(path).stat()
    return tuple(getattr(s,k) for k in ('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns'))


def _json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate source JSON key')
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite source JSON')))


def _hash(reference):
    if (type(reference) is not dict or type(reference.get('path')) is not str
            or type(reference.get('bytes')) is not int or reference['bytes']<1
            or type(reference.get('sha256')) is not str or re.fullmatch('[0-9a-f]{64}',reference['sha256']) is None):
        raise ValueError('exact source path/byte length/SHA256 required')


@dataclass(frozen=True)
class NativeTextRow:
    source_family: str
    source_coordinate_json: bytes
    conditions_json: bytes
    full_target_utf8: bytes
    source_audit_json: bytes
    group_keys: tuple
    original_role: object
    evaluation_only: bool
    rights_disposition: str

    def condition_inputs(self):
        return _json(self.conditions_json)

    def source_audit(self):
        return _json(self.source_audit_json)

    @property
    def training_authorized(self):
        return False


def _row(family,coordinate,conditions,target,audit,groups,role=None,*,evaluation_only=False,rights='source-native-candidate'):
    task=conditions.get('task')
    if task=='source_language_prior':
        if (set(conditions)!={'task','language','domain'} or conditions['language'] not in ('he','en')
                or conditions['domain'] not in ('biomedical','biblical','general','mathematics','code')):
            raise ValueError('literal source-language-only condition whitelist required')
    elif task=='oshb_source_root_list':
        if set(conditions)!={'task','word'} or type(conditions['word']) is not str or not conditions['word']:
            raise ValueError('complete native word-only root condition required')
    elif task in ('gsm8k_answer','humaneval_solution'):
        field='question' if task=='gsm8k_answer' else 'prompt'
        if set(conditions)!={'task',field} or type(conditions[field]) is not str:
            raise ValueError('complete question/prompt-only condition required')
    else:raise ValueError('unsupported native source-text task')
    if role not in (None,'train','test','validation','evaluation_only'):
        raise ValueError('literal original role or explicit unknown required')
    if task=='humaneval_solution' and (role!='evaluation_only' or evaluation_only is not True):
        raise ValueError('HumanEval source can only support evaluation')
    if type(target) is not bytes or not groups or any(type(k) is not str or not k for k in groups):
        raise ValueError('full source target bytes and explicit native groups required')
    return NativeTextRow(family,_wire(coordinate),_wire(conditions),target,_wire(audit),tuple(groups),
        role,evaluation_only,rights)


def wiki_rows(native,coordinate):
    page=native['native_page_id']
    if type(page) is not int or page<1 or native.get('text_normalization_performed') is not False:
        raise ValueError('exact original page identity/no-normalization evidence required')
    for ordinal,revision in enumerate(native['native_revisions']):
        target=_text(revision['native_text_exact']);rid=revision['native_revision_id']
        if (type(rid) is not int or rid<1 or revision.get('native_text_available') is not True
                or revision['native_text_UTF8_bytes']!=len(target)
                or revision.get('native_declared_byte_count_matches') is not True
                or revision.get('native_declared_base36_SHA1_matches') is not True):
            raise ValueError('complete exact original native revision text required')
        yield _row('INV-081',dict(coordinate,revision_ordinal=ordinal,native_page_id=page,native_revision_id=rid),
            dict(task='source_language_prior',language='he',domain='biomedical'),target,native,
            ('hewiki:page:'+str(page),),rights='CC-BY-SA-4.0-native-notices-and-history-retained')


def sefaria_row(native,coordinate):
    target=_text(native['text_exact_native'])
    if (hashlib.sha256(target).hexdigest()!=native['text_SHA256']
            or native.get('text_normalization_performed') is not False or native['native_license']!='Public Domain'):
        raise ValueError('exact native verse/codepoints/public-domain edition required')
    return _row('INV-080',coordinate,dict(task='source_language_prior',language='he',domain='biblical'),
        target,native,(WLC_LINEAGE,native['source_book_group_key']),rights='native-selected-edition-Public-Domain')


def logical_character_slots(tree, *, include_nontext=True, include_root_tail=True):
    """In-order exact text/tail slots with native child coordinates, no separators."""
    stack=[(tree,(),False)]
    while stack:
        node,path,tail=stack.pop()
        if type(node) is not dict or set(node)!={'tag','attributes','text','tail','children'}:
            raise ValueError('complete literal native XML logical node required')
        if tail:
            if node['tail'] is not None and (path or include_root_tail):yield dict(source_child_ordinal_path=list(path),slot='tail',text_exact_native=node['tail'])
            continue
        if node['text'] is not None and (include_nontext or node['tag'] not in ('#comment','#processing-instruction')):yield dict(source_child_ordinal_path=list(path),slot='text',text_exact_native=node['text'])
        stack.append((node,path,True))
        if type(node['children']) is not list:raise ValueError('ordered native children required')
        for i in range(len(node['children'])-1,-1,-1):stack.append((node['children'][i],path+(i,),False))


def oshb_book_row(tree,coordinate,book):
    slots=tuple(logical_character_slots(tree))
    target=b''.join(_text(s['text_exact_native']) for s in slots)
    audit=dict(native_book_OSIS_ID=book,complete_native_XML_logical_tree=tree,
        source_character_slots=slots,role='complete-original-XML-character-stream-not-a-new-edition',
        morphology_and_lexical_root_attributes_are_not_model_conditions=True,
        word_root_assertions_require_separate_exact_native_sidecar=True)
    return _row('INV-082/OSHB',coordinate,dict(task='source_language_prior',language='he',domain='biblical'),
        target,audit,(WLC_LINEAGE,'OSHB:book:'+book),rights='CC-BY-4.0-native-source-annotations-WLC-lineage')


def reasoning_row(native,coordinate):
    fields=native['native_fields'];repo=native['source_repository'];role=native['official_role']
    if repo=='openai/grade-school-math':
        if set(fields)!={'question','answer'} or role not in ('train','test'):
            raise ValueError('complete original GSM8K fields/official role required')
        q=_text(fields['question']);target=_text(fields['answer'])
        return _row('INV-084/GSM8K',coordinate,dict(task='gsm8k_answer',question=fields['question']),target,native,
            ('GSM8K:question-sha256:'+hashlib.sha256(q).hexdigest(),),role,rights='MIT-original-source')
    if repo=='openai/human-eval':
        if (set(fields)!={'task_id','prompt','canonical_solution','test','entry_point'}
                or role!='evaluation_only' or native.get('generated_code_or_tests_executed') is not False):
            raise ValueError('complete evaluation-only HumanEval fields; code remains inert')
        return _row('INV-084/HumanEval',coordinate,dict(task='humaneval_solution',prompt=fields['prompt']),
            _text(fields['canonical_solution']),native,('HumanEval:task:'+fields['task_id'],),role,
            evaluation_only=True,rights='MIT-original-source-evaluation-only')
    raise ValueError('unrecognized original reasoning repository')


def jats_row(native,coordinate,*,rights_disposition='unreviewed-original-grant-context'):
    # The source descriptor's exact_text_segments includes XML element text
    # and all child tails. XML comment/PI contents are nontext markup preserved
    # in the complete audit tree, rather than visible article character slots.
    slots=tuple(logical_character_slots(native['complete_XML_logical_tree'],
        include_nontext=False,include_root_tail=False))
    declared=native['native_character_data_segments']
    reduced=[{k:x[k] for k in ('source_child_ordinal_path','slot','text_exact_native')} for x in declared]
    if _wire(reduced)!=_wire(slots) or native.get('XML_codepoints_or_source_text_normalized') is not False:
        raise ValueError('complete native article text/tail source order and boundaries required')
    qualification=native['qualification'];groups=[]
    for identity in qualification['native_article_ID_fields']:
        kind,value=identity['native_type'],identity['native_value']
        if kind in ('pmcid','pmid','doi') and type(value) is str and value:
            groups.append('article-self-'+kind+':'+value)
    if not groups:raise ValueError('exact self-article IDs required; bibliography is not identity')
    return _row('INV-083',coordinate,dict(task='source_language_prior',language='en',domain='biomedical'),
        b''.join(_text(s['text_exact_native']) for s in slots),native,tuple(groups),rights=rights_disposition)


class SourceTextNativeBinding:
    """Pinned source-projection artifacts; original split unknown stays unknown."""
    def __init__(self,handoff_path,expected_sha256,*,root):
        self.root=Path(root);p=Path(handoff_path);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('trusted source handoff SHA differs')
        self._handoff_json=raw;self._handoff_identity=_identity(p);self._handoff_path=p
        doc=_json(raw)
        if len(doc['families'])!=len(FAMILIES) or {x['id'] for x in doc['families']}!=set(FAMILIES):raise ValueError('exact finite selected source families required')
        english=next(x for x in doc['families'] if x['id']=='INV-083')
        if (doc.get('historical7_candidate_count_superseded_by6_consistent_declarations_and1_conflict') is not True
                or 'audit_only_grant_conflicts' not in english):
            raise ValueError('explicit additive original-grant conflict requalification required')
        self._family_json={x['id']:_wire(x) for x in doc['families']}
        self._source_identities={}
        for family in doc['families']:self._read(family['manifest'])

    def _read(self,reference):
        _hash(reference);p=self.root/reference['path'];before=_identity(p);raw=p.read_bytes()
        if (_identity(p)!=before or len(raw)!=reference['bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']):
            raise ValueError('native source projection identity/SHA/bytes changed')
        known=self._source_identities.get(str(p))
        if known is not None and known!=before:raise ValueError('bound source identity changed')
        self._source_identities[str(p)]=before
        return raw

    def require_sources_unchanged(self):
        if _identity(self._handoff_path)!=self._handoff_identity:raise ValueError('source handoff changed')
        if any(_identity(Path(p))!=i for p,i in self._source_identities.items()):raise ValueError('bound native projection changed')

    def _records(self,artifact):
        raw=self._read(artifact['ref']);fmt=artifact['format']
        if fmt=='JSONL':stream=io.BytesIO(raw)
        elif fmt=='JSONL.gz':stream=gzip.GzipFile(fileobj=io.BytesIO(raw))
        elif fmt in ('JSON','JSON.gz'):
            yield 0,_json(gzip.decompress(raw) if fmt=='JSON.gz' else raw);return
        else:raise ValueError('explicit supported inert native JSON format required')
        for ordinal,line in enumerate(stream):
            if not line.strip():raise ValueError('unexpected empty native source record')
            yield ordinal,_json(line)

    def iter_rows(self,*,families=FAMILIES):
        if (type(families) not in (tuple,list) or not families or len(set(families))!=len(families)
                or any(x not in self._family_json for x in families)):
            raise ValueError('explicit unique selected source families required')
        for fid in families:
            family=_json(self._family_json[fid])
            if fid=='INV-081':
                for artifact in family['artifacts']:
                    for ordinal,native in self._records(artifact):
                        coordinate=dict(native_projection=artifact['ref'],native_projection_row_ordinal=ordinal)
                        yield from wiki_rows(native,coordinate)
            elif fid=='INV-080':
                artifact=next(a for a in family['artifacts'] if a['format']=='JSONL')
                for ordinal,native in self._records(artifact):
                    yield sefaria_row(native,dict(native_projection=artifact['ref'],native_projection_row_ordinal=ordinal))
            elif fid=='INV-082/OSHB':
                for book in family['books']:
                    artifact=book['full_book_XML_logical_tree'];items=list(self._records(artifact))
                    if len(items)!=1:raise ValueError('one complete native book tree required')
                    yield oshb_book_row(items[0][1],dict(native_projection=artifact['ref'],source_object_ordinal=book['source_object_ordinal']),book['native_book_OSIS_ID'])
            elif fid=='INV-084':
                for item in family['objects']:
                    artifact=item['artifact']
                    for ordinal,native in self._records(artifact):
                        if native['official_role']!=item['official_role']:raise ValueError('original source role differs')
                        yield reasoning_row(native,dict(native_projection=artifact['ref'],native_projection_row_ordinal=ordinal))
            else:
                for article in family['articles']:
                    artifact=article['artifact'];items=list(self._records(artifact))
                    if len(items)!=1:raise ValueError('one complete article projection required')
                    native=items[0][1]
                    qualification=native['qualification']
                    if (_wire(qualification)!=_wire(article['grant_context'])
                            or qualification.get('rights_qualified') is not True
                            or qualification.get('language_qualified') is not True
                            or qualification.get('identity_or_scope_conflicts')):
                        raise ValueError('consistent exact original-grant/language source context required')
                    yield jats_row(native,dict(native_projection=artifact['ref'],PMCID=article['PMCID']),
                        rights_disposition='source-grant-consistent-candidate-corpus-admission-pending')
            self.require_sources_unchanged()


    def iter_oshb_word_contexts(self):
        """Separate direct native root/morphology sidecar, no root inference.

        Empty direct-root lists mean unlabeled root context. Lemma/morphology,
        lexical IDs and gold roots stay outside unknown-root observations.
        """
        import pyarrow.parquet as pq
        family=_json(self._family_json['INV-082/OSHB'])
        for book in family['books']:
            artifact=book['word_index'];raw=self._read(artifact['ref'])
            table=pq.read_table(io.BytesIO(raw))
            tree_artifact=book['full_book_XML_logical_tree']
            tree=_json(gzip.decompress(self._read(tree_artifact['ref'])))
            for row in table.to_pylist():
                if (row['source_book_OSIS_ID']!=book['native_book_OSIS_ID']
                        or row['source_object_ordinal']!=book['source_object_ordinal']):
                    raise ValueError('native root sidecar book/source identity differs')
                subtree=tree
                for index in row['source_child_ordinal_path']:
                    if type(index) is not int or index<0 or index>=len(subtree['children']):
                        raise ValueError('exact native word XML path differs')
                    subtree=subtree['children'][index]
                yield native_word_root_context(row,dict(native_projection=artifact['ref'],
                    source_word_ordinal=row['source_word_ordinal'],original_XML_logical_tree=tree_artifact['ref']),
                    source_word_subtree=subtree)
            self.require_sources_unchanged()


@dataclass(frozen=True)
class NativeWordRootContext:
    observed_word_utf8: bytes
    direct_root_assertions_json: bytes
    gold_morphology_json: bytes
    source_audit_json: bytes
    source_coordinate_json: bytes
    group_keys: tuple
    root_target_status: str
    word_input_status: str

    def unknown_root_observations(self):
        return {'observed_word':self.observed_word_utf8.decode('utf8')}

    def observed_root_list_view(self):
        if self.root_target_status=='unlabeled':return None
        if self.word_input_status not in ('complete-native-leaf-word','complete-native-word-inline-formatting'):
            raise ValueError('complete original word character evidence required; no direct-slot truncation')
        roots=[item['native_root'] for item in _json(self.direct_root_assertions_json)]
        if not roots or any(type(root) is not str or not root for root in roots):
            raise ValueError('nonempty ordered literal source root strings required')
        return _row('INV-082/OSHB-observed-root-list',_json(self.source_coordinate_json),
            dict(task='oshb_source_root_list',word=self.observed_word_utf8.decode('utf8')),
            _wire(roots),dict(source_native_word=_json(self.source_audit_json),
                target_codec='Mora3.OSHBOrderedNativeRootListUTF8JSON/v1',
                native_root_order_and_repetitions_preserved=True,
                no_empty_or_unknown_root_target=True),self.group_keys,
            rights='CC-BY-4.0-source-annotation-list-not-biological-root-closure')

    @property
    def training_authorized(self):return False


def native_word_root_context(native,coordinate,*,source_word_subtree=None):
    roots=native['direct_original_lexicon_root_bindings']
    if type(roots) is not list:
        raise ValueError('explicit direct native root assertion list required')
    for root in roots:
        if (type(root) is not dict or not {'component_ordinal','native_aug_code','lexical_entry_id',
                'native_root','native_lexical_entry_ordinal'}<=set(root)
                or type(root['native_root']) is not str or not root['native_root']):
            raise ValueError('literal nonempty native lexical root assertion required')
    if (type(native['native_word_ID']) is not str or not native['native_word_ID']
            or type(native['source_child_ordinal_path']) is not list
            or any(type(x) is not int or x<0 for x in native['source_child_ordinal_path'])):
        raise ValueError('exact native word identity and original XML coordinate required')
    status='source-direct-slot-only-unqualified-as-complete-word'
    word_text=native['native_word_direct_text']
    audit=dict(native)
    if source_word_subtree is not None:
        node=source_word_subtree
        if (type(node) is not dict or node.get('tag','').rsplit('}',1)[-1]!='w'
                or node.get('attributes')!=_json(native['native_word_attributes_JSON'])
                or (node.get('text') or '')!=native['native_word_direct_text']):
            raise ValueError('exact original word subtree/attributes/direct slot differs')
        status=('native-word-has-child-content' if node['children'] else
            'complete-native-leaf-word' if native['native_word_direct_text'] else 'native-word-text-missing')
        # These native leaf seg nodes alter glyph presentation only. Include
        # their exact characters/tails in source order; never flatten notes,
        # reading alternatives or unknown child grammars into a word.
        seg_tag=(node['tag'].rsplit('}',1)[0]+'}seg') if '}' in node['tag'] else 'seg'
        if node['children'] and all(type(child) is dict and child.get('tag')==seg_tag
                and not child.get('children') and set(child.get('attributes',{}))=={'type'}
                and child['attributes']['type'] in ('x-large','x-small','x-suspended')
                for child in node['children']):
            word_text=(node.get('text') or '')+''.join((child.get('text') or '')+(child.get('tail') or '')
                for child in node['children'])
            status='complete-native-word-inline-formatting' if word_text else 'native-word-text-missing'
        audit.update(original_native_word_XML_subtree=node,word_input_status=status,
            complete_native_word_text_UTF8_SHA256=hashlib.sha256(_text(word_text)).hexdigest())
    return NativeWordRootContext(_text(word_text),_wire(roots),
        _wire(dict(native_morph=native['native_morph'],native_lemma=native['native_lemma'])),
        _wire(audit),_wire(coordinate),(WLC_LINEAGE,'OSHB:book:'+native['source_book_OSIS_ID']),
        'observed-direct-source-assertions' if roots else 'unlabeled',status)
