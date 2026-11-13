"""Explicit appended control axis and byte-exact, typed complete-forest grammar.

Existing byte/learned/control IDs are preserved. The extended AnnotationVocabulary
is a NEW artifact; no old metadata offset, embedding or checkpoint is migrated.
Token syntax is not evidence validation. All generated node content stays
predicted regardless of text such as 'observed' appearing in payload bytes.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import json,re
from .shortest_path import ByteTokenizer

_NAMES=('FOREST_BEGIN','FOREST_END','TREE_BEGIN','TREE_END','NODE_BEGIN','NODE_END','EDGE_BEGIN','EDGE_END',
 'PARENT','BRANCH','JOIN','SUBGOAL','PROPOSE','VERIFY','REVISE','FINISH',
 'CONDITION','DESIRED','PREDICTED','SOURCE_OBSERVED','SOURCE_COMPUTED','SIMULATED','HYPOTHESIS','UNKNOWN',
 'ANALOGY','MAPPING','RELATION','CONSTRAINT','UNIT','FRAME','MISSING','ABSTAIN',
 'SUPPORTED','UNSUPPORTED','PASS','FAIL','UNCERTAIN','CONFLICT','INFEASIBLE','BUDGET_EXHAUSTED')
FOREST_CONTROLS=tuple('<MORA:'+name+'>'for name in _NAMES)


def _wire(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate vocabulary JSON field')
        result[key]=value
    return result


@dataclass(frozen=True)
class ForestTokenAxis:
    base_specification_json: str
    base_sha256: str
    learned_hex: tuple[str,...]
    base_controls: tuple[str,...]
    controls: tuple[str,...]=FOREST_CONTROLS

    def __post_init__(self):
        if(type(self.base_specification_json)is not str or len(self.base_specification_json.encode())>8*1024**2
            or type(self.base_sha256)is not str or re.fullmatch('[0-9a-f]{64}',self.base_sha256)is None):
            raise ValueError('Bounded exact source vocabulary and digest required')
        if sha256(self.base_specification_json.encode()).hexdigest()!=self.base_sha256:
            raise ValueError('Base vocabulary digest mismatch')
        base=json.loads(self.base_specification_json,object_pairs_hook=_unique)
        if ('forest_extension'in base or type(self.learned_hex)is not tuple or type(self.base_controls)is not tuple
            or self.controls!=FOREST_CONTROLS or type(self.controls)is not tuple
            or tuple(base['learned_hex'])!=self.learned_hex or tuple(base['controls'])!=self.base_controls
            or set(self.controls)&set(self.base_controls)):
            raise ValueError('One explicit fixed control extension of the exact ordered base axis required')
        if any(type(x)is not str or re.fullmatch('(?:[0-9a-f]{2}){2,}',x)is None for x in self.learned_hex):
            raise ValueError('Canonical multi-byte learned payload tokens required')
        if not {'pad','node','edge'}<=set(self.base_controls):raise ValueError('Native base graph controls required')
        tokenizer=ByteTokenizer(tuple(bytes.fromhex(x)for x in self.learned_hex),controls=self.base_controls+self.controls)
        if tokenizer.size>65536:raise ValueError('Forest vocabulary exceeds declared finite axis limit')

    @classmethod
    def from_annotation(cls,vocabulary):
        from mora3.model.annotation_vocabulary import AnnotationVocabulary
        if type(vocabulary)is not AnnotationVocabulary:raise ValueError('Exact native fitted vocabulary artifact required')
        text=vocabulary.specification_json;base=json.loads(text,object_pairs_hook=_unique)
        return cls(text,sha256(text.encode()).hexdigest(),tuple(base['learned_hex']),tuple(base['controls']))

    @property
    def sha256(self):
        self.__post_init__();return sha256(_wire(dict(schema='Mora3.ForestTokenAxis/v1',**asdict(self)))).hexdigest()

    @property
    def control_ids(self):
        t=self.tokenizer();return tuple(t.control_id(x)for x in self.controls)

    def tokenizer(self):
        self.__post_init__()
        return ByteTokenizer(tuple(bytes.fromhex(x)for x in self.learned_hex),controls=self.base_controls+self.controls)

    def annotation_vocabulary(self):
        from mora3.model.annotation_vocabulary import AnnotationVocabulary
        t=self.tokenizer();base=json.loads(self.base_specification_json,object_pairs_hook=_unique)
        base['controls']=list(self.base_controls+self.controls)
        base['forest_extension']=dict(schema='Mora3.ForestControlExtension/v1',base_sha256=self.base_sha256,
            axis_sha256=self.sha256,control_ids={name:t.control_id(name)for name in self.controls},
            base_fit_certificate_unchanged=True,new_lp_fit=False,metadata_offset=t.size)
        return AnnotationVocabulary(_wire(base).decode())


def serialize_token_forest(axis,forest,payloads,*,max_tokens=1_000_000,max_bytes=4*1024**2):
    """Serialize all complete trees, preserving bytes, roles and directed parents.

    Source node IDs are replaced with explicit canonical wire indices. Their
    provenance remains in the separate forest record. Parent references refer
    only to prior nodes in the same tree. This is not serialization of embeddings.
    """
    from mora3.reasoning.flow_forest_refinement import ForestState
    if type(axis)is not ForestTokenAxis or type(forest)is not ForestState:
        raise ValueError('Exact forest axis and complete native forest required')
    axis.__post_init__();forest.__post_init__()
    if(not forest.is_complete or type(payloads)is not tuple or len(payloads)!=len(forest.parents)
        or any(type(x)is not bytes for x in payloads)
        or type(max_tokens)is not int or max_tokens<1 or type(max_bytes)is not int or max_bytes<1):
        raise ValueError('Complete forest, all byte payloads and exact resource budgets required')
    if sum(map(len,payloads))>max_bytes:raise ValueError('Full forest content exceeds byte budget; no truncation')
    t=axis.tokenizer();c=lambda name:t.control_id('<MORA:'+name+'>')
    roots=[i for i,p in enumerate(forest.parents)if p==-1];children={i:[]for i in range(len(forest.parents))}
    for i,p in enumerate(forest.parents):
        if p!=-1:children[p].append(i)
    out=[c('FOREST_BEGIN')];mapping={}
    for root in roots:
        out.append(c('TREE_BEGIN'));stack=[root]
        while stack:
            i=stack.pop();mapping[i]=len(mapping);p=forest.parents[i]
            out.extend((c('NODE_BEGIN'),c('PARENT')))
            out.extend(t.encode(str(-1 if p==-1 else mapping[p]).encode()))
            out.append(c('RELATION'));out.extend(t.encode(forest.directions[i].encode()))
            out.extend((c('FINISH')if forest.statuses[i]=='terminal'else c('SUBGOAL'),
                        c('CONDITION')if i in forest.clamped else c('PREDICTED')))
            out.extend(t.encode(payloads[i]));out.append(c('NODE_END'))
            stack.extend(reversed(children[i]))
            if len(out)>max_tokens:raise ValueError('Full forest serialization exceeds token budget; no truncation')
        out.append(c('TREE_END'))
    out.append(c('FOREST_END'))
    if len(out)>max_tokens:raise ValueError('Full forest serialization exceeds token budget')
    return tuple(out)


def parse_token_forest(axis,tokens,*,max_nodes=512,max_tokens=1_000_000,max_bytes=4*1024**2):
    """Strict inverse of the typed wire grammar, not an inverse latent decoder."""
    if type(axis)is not ForestTokenAxis:raise ValueError('Exact forest vocabulary required')
    axis.__post_init__();t=axis.tokenizer();c=lambda name:t.control_id('<MORA:'+name+'>')
    if (type(tokens)is not tuple or type(max_tokens)is not int or not 1<=len(tokens)<=max_tokens
        or type(max_nodes)is not int or not 1<=max_nodes<=512 or type(max_bytes)is not int or max_bytes<1
        or any(type(x)is not int or not 0<=x<t.size for x in tokens)):
        raise ValueError('Bounded exact finite token tuple required')
    i=0;rows=[];used=0
    def expect(token):
        nonlocal i
        if i>=len(tokens)or tokens[i]!=token:raise ValueError('Malformed complete forest token grammar')
        i+=1
    def payload():
        nonlocal i,used
        values=[]
        while i<len(tokens)and tokens[i]<len(t.vocabulary):values.append(tokens[i]);i+=1
        raw=t.decode(values);used+=len(raw)
        if used>max_bytes:raise ValueError('Decoded forest exceeds byte budget')
        return raw
    expect(c('FOREST_BEGIN'));trees=0
    while i<len(tokens)and tokens[i]==c('TREE_BEGIN'):
        i+=1;trees+=1;start=len(rows)
        while i<len(tokens)and tokens[i]==c('NODE_BEGIN'):
            if len(rows)>=max_nodes:raise ValueError('Decoded complete forest exceeds node budget')
            i+=1;expect(c('PARENT'));raw=payload()
            if re.fullmatch(b'-1|0|[1-9][0-9]*',raw)is None:raise ValueError('Canonical parent reference required')
            parent=int(raw);expect(c('RELATION'));direction=payload()
            if direction not in(b'root',b'computational',b'temporal',b'observational',b'interventional'):
                raise ValueError('Explicit recognized direction required')
            if i>=len(tokens)or tokens[i]not in(c('SUBGOAL'),c('FINISH')):raise ValueError('Explicit complete node status required')
            status='terminal'if tokens[i]==c('FINISH')else'open';i+=1
            if i>=len(tokens)or tokens[i]not in(c('CONDITION'),c('PREDICTED')):raise ValueError('Explicit non-upgraded evidence role required')
            role='condition'if tokens[i]==c('CONDITION')else'predicted';i+=1
            raw=payload();expect(c('NODE_END'))
            if len(rows)==start:
                if parent!=-1 or direction!=b'root' or role!='condition':raise ValueError('A tree starts with one fixed condition root')
            elif not start<=parent<len(rows)or rows[parent]['status']!='open' or direction==b'root':
                raise ValueError('Forward, same-tree open-parent reference required')
            rows.append(dict(parent=parent,direction=direction.decode(),status=status,role=role,payload=raw))
        if len(rows)==start:raise ValueError('Empty tree is not a complete reasoning tree')
        expect(c('TREE_END'))
    expect(c('FOREST_END'))
    if i!=len(tokens)or not trees:raise ValueError('Trailing tokens or absent complete forest')
    parents={row['parent']for row in rows if row['parent']!=-1}
    if any((n not in parents)!=(row['status']=='terminal')for n,row in enumerate(rows)):
        raise ValueError('Every declared terminal must be a leaf; open leaves are censored, not complete')
    return tuple(rows)
