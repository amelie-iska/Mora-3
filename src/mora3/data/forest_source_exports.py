"""Complete source traces -> fixed native prefix coordinates -> forest supervision.

This is a source-known-tree reconstruction task, not unknown-topology learning.
The separately frozen encoder is lossy. Its actual weights/codebook are bound,
not asserted to be trained or scientifically qualified. Source roles never change.
Plain JSON with exact tensor bytes avoids pickle and execution of source programs.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import copy
import json
import re
import sys
import torch
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.annotation_vocabulary import AnnotationPayloadEncoder
from mora3.model.forest_flow import forest_context_sha256
from mora3.model.forest_token_decoder import ForestTokenCodebook, _tensor_sha
from mora3.tokenization.forest_tokens import ForestTokenAxis
from mora3.reasoning.trajectory_contracts import (ReasoningTrace, TraceNode, TraceStep,
    TraceFact, prefix_record, prefix_native_graph)
from mora3.reasoning.directed_topology import Square
from mora3.reasoning.flow_forest_refinement import ForestState

SCHEMA = 'Mora3.CompleteSourceForestExport/v1'
MAX_BYTES = 64 * 1024**2
MAX_TOTAL_PREFIX_TOKENS = 64_000_000
SKELETON_POLICY = sha256(b'Mora3.source-known-complete-tree-all-prefixes-v1').hexdigest()
_DTYPES = {'torch.float32': torch.float32, 'torch.float64': torch.float64}


def _wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('Exact SHA256 commitment required')


def _text(value):
    if type(value) is not str or not 0 < len(value.encode('utf8')) <= 4096:
        raise ValueError('Bounded nonempty source text required')


def _parse(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate export JSON key')
            result[key]=value
        return result
    def constant(_): raise ValueError('Nonfinite export JSON constant')
    try:
        result=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
        _wire(result)
        return result
    except (TypeError,UnicodeError,RecursionError,OverflowError) as exc:
        raise ValueError('Finite bounded source JSON required') from exc


def _trace(value):
    try:
        d=dict(value)
        d['groups']=tuple(d['groups'])
        d['nodes']=tuple(TraceNode(**(n|{'visible_facts':tuple(n['visible_facts'])}))for n in d['nodes'])
        d['steps']=tuple(TraceStep(**(s|{'requires':tuple(s['requires'])}))for s in d['steps'])
        d['facts']=tuple(TraceFact(**(f|{'depends_on':tuple(f['depends_on'])}))for f in d['facts'])
        d['squares']=tuple(Square(tuple(s['left']),tuple(s['right']),s['evidence_sha256'])for s in d['squares'])
        return ReasoningTrace(**d)
    except (TypeError,KeyError) as exc:
        raise ValueError('Exact complete typed source trace required') from exc


def _layout(trace):
    if type(trace) is not ReasoningTrace:
        raise ValueError('Exact original ReasoningTrace required')
    trace.__post_init__()
    if trace.squares or any(n.status not in ('open','terminal')for n in trace.nodes):
        raise ValueError('Only complete trees supported; no censored branch or square may be dropped')
    incoming={n.key:[]for n in trace.nodes}
    for step in trace.steps: incoming[step.target].append(step)
    if any(len(v)>1 for v in incoming.values()):
        raise ValueError('Merged DAG histories require a separate representation, not silent tree flattening')
    nodes=tuple(n for _,n in sorted(enumerate(trace.nodes),key=lambda pair:(pair[1].layer,pair[0])))
    positions={n.key:i for i,n in enumerate(nodes)}; paths={}; parents=[]; directions=[]; payloads=[]
    facts={f.name:f for f in trace.facts}
    for n in nodes:
        edge=incoming[n.key]
        if edge:
            edge=edge[0];paths[n.key]=paths[edge.source]+(edge.key,)
            parents.append(positions[edge.source]);directions.append(edge.direction_kind)
        else:
            paths[n.key]=();parents.append(-1);directions.append('root')
        state=prefix_record(trace,paths[n.key])['state']
        # Preserve the original lexical JSON string rather than reformat numbers.
        for item in state['facts']:
            item.pop('value');item['value_json']=facts[item['name']].value_json
        payloads.append(_wire(state))
    return nodes,tuple(parents),tuple(directions),tuple(paths[n.key]for n in nodes),tuple(payloads)


def _stream_lengths(geometry,record,task):
    selected=task.input_fields(record);groups={}
    for role,entities in (('node',record.nodes),('edge',record.edges)):
        groups[role+'s']=tuple(replace(e,measurements=tuple(m for m in e.measurements
            if (role,i,m.name)in selected))for i,e in enumerate(entities))
    payload=geometry.payload;packed=payload.codec.encode(replace(record,**groups));lengths=[]
    for entity in packed['entities']:
        count=len(payload._header({'kind':entity['kind'],'directed':entity['directed']}))
        for field in sorted(entity['measurements'],key=lambda f:f['name']):
            count+=2+len(payload._header({k:field[k]for k in ('name','units','status','value_type')}))+len(field['token_ids'])
        lengths.append(count)
    return tuple(lengths)


def _module_wire(geometry,payload,axis):
    axis.__post_init__()
    if payload._vocabulary_json != axis.annotation_vocabulary().specification_json:
        raise ValueError('Exact extended native payload vocabulary required; old metadata rows are not controls')
    tokenizer=axis.tokenizer();actual=payload.codec.tokenizer
    if (payload.offset!=tokenizer.size or actual.vocabulary!=tokenizer.vocabulary
            or actual._controls!=tokenizer._controls
            or payload.embedding.num_embeddings!=tokenizer.size+258
            or payload.embedding.weight.shape[0]!=tokenizer.size+258):
        raise ValueError('Actual extended payload tokenizer/embedding differs from committed axis')
    values=[]
    for prefix,module in (('geometry',geometry),('payload',payload)):
        for name,m in module.named_modules():
            if m.training: raise ValueError('Frozen source encoder cannot change to training mode')
        for name,p in module.named_parameters():
            if p.requires_grad or p.grad is not None or p.device.type!='cpu' or p.dtype not in _DTYPES.values():
                raise ValueError('Detached CPU FP32/FP64 frozen source parameters required')
            if not torch.isfinite(p).all(): raise ValueError('Finite frozen source weights required')
            values.append([prefix+'.'+name,_tensor_sha(p)])
        for name,b in module.named_buffers():
            if b.device.type!='cpu' or b.is_floating_point()and not torch.isfinite(b).all():
                raise ValueError('Finite CPU frozen source buffers required')
            values.append([prefix+'.buffer:'+name,_tensor_sha(b)])
    ref=geometry.graph.feature.weight
    if payload.embedding.weight.dtype!=ref.dtype:
        raise ValueError('Source prefix and token encoder dtypes must agree')
    gp=geometry.payload;tok=gp.codec.tokenizer
    source_modules={type(m).__module__ for root in (geometry,payload)for m in root.modules()
                    if type(m).__module__.startswith('mora3.')}
    source_modules.update((__name__,'mora3.reasoning.trajectory_contracts',
        'mora3.tokenization.graph_payload','mora3.tokenization.shortest_path',
        'mora3.tokenization.forest_tokens','mora3.data.schema','mora3.data.tasks',
        'mora3.data.provenance','mora3.model.forest_flow','mora3.model.forest_token_decoder'))
    implementation={}
    for name in sorted(source_modules):
        module=sys.modules.get(name);file=getattr(module,'__file__',None)
        if not file or Path(file).suffix!='.py':
            raise ValueError('Exact native source implementation file required')
        implementation[name]=sha256(Path(file).read_bytes()).hexdigest()
    return _wire(dict(geometry=geometry.get_extra_state(),payload=payload.get_extra_state(),
        implementation_sha256=implementation,
        runtime=dict(torch=str(torch.__version__),torch_git_revision=torch.version.git_version,
            python=list(sys.version_info[:3]),byteorder=sys.byteorder,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
        geometry_payload=dict(max_tokens=gp.max_tokens,pooling=gp.pooling,offset=gp.offset,
            vocabulary=[x.hex()for x in tok.vocabulary],controls=tok._controls),
        modules=[(prefix,name,type(m).__module__+'.'+type(m).__qualname__)
            for prefix,root in (('geometry',geometry),('payload',payload))for name,m in root.named_modules()],
        source_native=geometry._source_backbone_reference[0].configuration(),
        axis_sha256=axis.sha256,weights_and_buffers=values))


class FrozenForestSourceEncoder:
    """Independent frozen snapshot; existing live training modules are never frozen.

    CPU export is deterministic under the recorded native kernels. No new model
    fitting, vocabulary fitting or training admission is performed here.
    """
    skeleton_policy_sha256=SKELETON_POLICY

    def __init__(self,geometry,payload,axis,*,source_snapshot_sha256,
                 encoder_evidence_sha256,encoder_status):
        if (type(geometry)is not GeometryConditioner or type(payload)is not AnnotationPayloadEncoder
                or type(axis)is not ForestTokenAxis):
            raise ValueError('Exact native geometry, extended payload and forest axis required')
        for value in (source_snapshot_sha256,encoder_evidence_sha256):_sha(value)
        if encoder_status not in ('untrained-diagnostic','checkpoint-export'):
            raise ValueError('Explicit diagnostic versus source-checkpoint encoder status required')
        axis.__post_init__()
        if payload._vocabulary_json!=axis.annotation_vocabulary().specification_json:
            raise ValueError('Exact extended vocabulary required; cannot reinterpret old metadata rows')
        if geometry.config['width']>256 or any(p.device.type!='cpu'for m in(geometry,payload)for p in m.parameters()):
            raise ValueError('Bounded CPU native encoder required; no implicit device transfer')
        self.geometry,self.payload=copy.deepcopy((geometry,payload));self.axis=axis
        for m in (self.geometry,self.payload):m.eval();m.requires_grad_(False)
        self.source_snapshot_sha256=source_snapshot_sha256
        self.encoder_evidence_sha256=encoder_evidence_sha256;self.encoder_status=encoder_status
        self.dimension=self.geometry.config['width'];self.dtype=self.geometry.graph.feature.weight.dtype
        self._frozen=_module_wire(self.geometry,self.payload,self.axis)
        self.encoder_sha256=sha256(_wire(dict(schema='Mora3.FrozenNativeForestEncoder/v1',
            source_snapshot_sha256=source_snapshot_sha256,encoder_evidence_sha256=encoder_evidence_sha256,
            encoder_status=encoder_status,configuration_sha256=sha256(self._frozen).hexdigest()))).hexdigest()
        self.coordinate_sha256=sha256(_wire(['Mora3.FrozenNativePrefixCoordinates/v1',self.encoder_sha256,
            self.dimension,'all-ancestor-graph-token-no-truncation'])).hexdigest()
        self._binding=_wire(self._description())

    def _description(self):
        return dict(schema='Mora3.FrozenNativeForestEncoder/v1',encoder_sha256=self.encoder_sha256,
            coordinate_sha256=self.coordinate_sha256,source_snapshot_sha256=self.source_snapshot_sha256,
            encoder_evidence_sha256=self.encoder_evidence_sha256,encoder_status=self.encoder_status,
            configuration_sha256=sha256(self._frozen).hexdigest(),axis_sha256=self.axis.sha256,
            dimension=self.dimension,dtype=str(self.dtype),lossless_latent=False,training_admission=False)

    def validate(self):
        if (_module_wire(self.geometry,self.payload,self.axis)!=self._frozen
                or _wire(self._description())!=self._binding):
            raise ValueError('Frozen source encoder weights/operator/axis/configuration changed')

    def binding(self):
        self.validate();return _parse(self._binding)

    def codebook(self):
        self.validate()
        values=self.payload.embedding.weight[:self.axis.tokenizer().size].detach().clone()
        export=sha256(_wire(['exact-extended-native-token-embedding-rows-v1',self.encoder_sha256,self.axis.sha256])).hexdigest()
        return ForestTokenCodebook(self.axis,values,self.source_snapshot_sha256,self.coordinate_sha256,
            self.encoder_sha256,export,self.encoder_evidence_sha256)


    def save_codebook(self,path):
        """Persist the exact native decoder asset as JSON/tensor bytes, no pickle.

        This records embedding values and provenance, not the complete source
        encoder checkpoint. Its independently retained checkpoint remains an
        upstream requirement for regenerating source latents.
        """
        self.validate();book=self.codebook()
        if sys.byteorder!='little':raise ValueError('Little-endian codebook wire required')
        data=dict(schema='Mora3.FrozenSourceCodebookExport/v1',axis=asdict(self.axis),
            encoder=self.binding(),binding=book.binding(),
            values_hex=book.values.contiguous().view(torch.uint8).numpy().tobytes().hex())
        raw=_wire(data)
        if len(raw)>MAX_BYTES:raise ValueError('Complete codebook export exceeds byte budget')
        path=Path(path).absolute()
        if any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('No symlink codebook destination')
        with path.open('xb')as stream:stream.write(raw)
        return sha256(raw).hexdigest()


def _evidence(trace):
    kinds={f.evidence_kind for f in trace.facts}
    return 'synthetic_fixture'if 'synthetic_fixture'in kinds else 'model_generated'if 'model_generated'in kinds else 'computed'


def _document(encoder,trace,layout,*,source_revision,original_ordinal,source_evidence_sha256,latent_hex,prefix_tokens):
    nodes,parents,directions,_,payloads=layout
    graph,task,_=prefix_native_graph(trace,())
    return dict(schema=SCHEMA,source_trace=asdict(trace),source_trace_sha256=sha256(_wire(asdict(trace))).hexdigest(),
        source_revision=source_revision,original_ordinal=original_ordinal,source_evidence_sha256=source_evidence_sha256,
        split=trace.split,groups=trace.groups,group_policy_sha256=trace.group_policy_sha256,
        evidence_kind=_evidence(trace),encoder=encoder.binding(),codebook=encoder.codebook().binding(),
        skeleton_policy_sha256=SKELETON_POLICY,node_ids=tuple(n.key for n in nodes),parents=parents,
        directions=directions,statuses=tuple(n.status for n in nodes),payload_hex=tuple(p.hex()for p in payloads),
        context_sha256=forest_context_sha256(graph,task,source_snapshot_sha256=encoder.source_snapshot_sha256),
        latent_shape=(len(nodes),encoder.dimension),latent_dtype=str(encoder.dtype),latent_hex=latent_hex,
        total_native_prefix_tokens=prefix_tokens,training_admission=False)


@torch.no_grad()
def compile_source_forest(encoder,trace,*,source_revision,original_ordinal,source_evidence_sha256,
                          max_nodes=128,max_total_prefix_tokens=2_000_000,max_export_bytes=MAX_BYTES):
    """Compile every branch with prefix-only neural inputs and exact source retention."""
    if type(encoder)is not FrozenForestSourceEncoder:raise ValueError('Exact frozen native source encoder required')
    encoder.validate();_text(source_revision);_sha(source_evidence_sha256)
    for value,limit in ((max_nodes,128),(max_total_prefix_tokens,MAX_TOTAL_PREFIX_TOKENS),(max_export_bytes,MAX_BYTES)):
        if type(value)is not int or not 1<=value<=limit:raise ValueError('Exact positive bounded export resources required')
    if type(original_ordinal)is not int or original_ordinal<0:raise ValueError('Exact original source ordinal required')
    layout=_layout(trace);nodes,_,_,paths,_=layout
    if len(nodes)>max_nodes or len(nodes)>encoder.geometry.config['id_width']:
        raise ValueError('Complete source tree exceeds node/identifier capacity; no truncation')
    inputs=[];total=0
    for path in paths:
        graph,task,_=prefix_native_graph(trace,path)
        lengths=_stream_lengths(encoder.geometry,graph,task)
        if any(n>encoder.geometry.payload.max_tokens for n in lengths):
            raise ValueError('Complete native entity exceeds source token capacity; no truncation')
        total+=sum(lengths)
        if total>max_total_prefix_tokens:raise ValueError('Complete source prefixes exceed token budget')
        inputs.append((graph,task))
    if sys.byteorder!='little':raise ValueError('This exact tensor wire version requires little-endian host')
    itemsize=torch.empty((),dtype=encoder.dtype).element_size()
    document=_document(encoder,trace,layout,source_revision=source_revision,original_ordinal=original_ordinal,
        source_evidence_sha256=source_evidence_sha256,latent_hex='00'*(len(nodes)*encoder.dimension*itemsize),prefix_tokens=total)
    if len(_wire(document))>max_export_bytes:raise ValueError('Complete source export exceeds byte budget before neural evaluation')
    # The complete source/prefix/byte budget is validated before ANY forward call.
    values=torch.cat([encoder.geometry(graph,task).detach()for graph,task in inputs],dim=0)
    if values.shape!=(len(nodes),encoder.dimension)or not torch.isfinite(values).all():
        raise ValueError('Complete finite native prefix coordinates required')
    encoder.validate()
    document['latent_hex']=values.contiguous().view(torch.uint8).numpy().tobytes().hex()
    raw=_wire(document)
    if len(raw)>max_export_bytes:raise ValueError('Complete source export exceeds byte budget')
    return ForestSourceExport(raw,sha256(raw).hexdigest())


def _check_encoder_codebook(enc,book):
    encoder_keys={'schema','encoder_sha256','coordinate_sha256','source_snapshot_sha256',
        'encoder_evidence_sha256','encoder_status','configuration_sha256','axis_sha256',
        'dimension','dtype','lossless_latent','training_admission'}
    if (type(enc)is not dict or set(enc)!=encoder_keys or enc.get('schema')!='Mora3.FrozenNativeForestEncoder/v1'
            or enc.get('encoder_status')not in('untrained-diagnostic','checkpoint-export')
            or enc.get('training_admission')is not False or enc.get('lossless_latent')is not False
            or type(enc.get('dimension'))is not int or not 1<=enc['dimension']<=256):
        raise ValueError('Exact declared frozen native encoder binding required')
    for name in('encoder_sha256','coordinate_sha256','source_snapshot_sha256','encoder_evidence_sha256','configuration_sha256','axis_sha256'):_sha(enc[name])
    encoder_identity=dict(schema='Mora3.FrozenNativeForestEncoder/v1',
        source_snapshot_sha256=enc['source_snapshot_sha256'],
        encoder_evidence_sha256=enc['encoder_evidence_sha256'],encoder_status=enc['encoder_status'],
        configuration_sha256=enc['configuration_sha256'])
    if enc['encoder_sha256']!=sha256(_wire(encoder_identity)).hexdigest():
        raise ValueError('Frozen source encoder commitment is internally inconsistent')
    coordinate=sha256(_wire(['Mora3.FrozenNativePrefixCoordinates/v1',enc['encoder_sha256'],
        enc['dimension'],'all-ancestor-graph-token-no-truncation'])).hexdigest()
    if enc['coordinate_sha256']!=coordinate:
        raise ValueError('Source coordinate commitment does not bind its frozen encoder')
    book_keys={'schema','axis_sha256','source_snapshot_sha256','coordinate_sha256','encoder_sha256',
        'export_sha256','evidence_sha256','values_sha256','shape','dtype','source_admission'}
    if (type(book)is not dict or set(book)!=book_keys or book['schema']!='Mora3.ForestTokenCodebook/v1'
            or book['source_admission']is not False or book['dtype']!=enc['dtype']
            or type(book['shape'])is not list or len(book['shape'])!=2
            or any(type(v)is not int for v in book['shape'])
            or not 1<=book['shape'][0]<=65536 or not 1<=book['shape'][1]<=1024):
        raise ValueError('Exact complete native source codebook schema/shape/dtype required')
    for name in('axis_sha256','source_snapshot_sha256','coordinate_sha256','encoder_sha256',
                'export_sha256','evidence_sha256','values_sha256'):_sha(book[name])
    for name in('encoder_sha256','coordinate_sha256','source_snapshot_sha256','axis_sha256'):
        if book[name]!=enc[name]:raise ValueError('Source codebook/encoder binding mismatch')
    export=sha256(_wire(['exact-extended-native-token-embedding-rows-v1',
        enc['encoder_sha256'],enc['axis_sha256']])).hexdigest()
    if book['export_sha256']!=export or book['evidence_sha256']!=enc['encoder_evidence_sha256']:
        raise ValueError('Source codebook export/evidence commitment mismatch')


@dataclass(frozen=True)
class ForestSourceExport:
    raw: bytes
    sha256: str

    def __post_init__(self):
        _sha(self.sha256)
        if type(self.raw)is not bytes or not 0<len(self.raw)<=MAX_BYTES or sha256(self.raw).hexdigest()!=self.sha256:
            raise ValueError('Exact immutable source export bytes required')
        d=_parse(self.raw)
        expected={'schema','source_trace','source_trace_sha256','source_revision','original_ordinal',
            'source_evidence_sha256','split','groups','group_policy_sha256','evidence_kind','encoder',
            'codebook','skeleton_policy_sha256','node_ids','parents','directions','statuses',
            'payload_hex','context_sha256','latent_shape','latent_dtype','latent_hex',
            'total_native_prefix_tokens','training_admission'}
        if (type(d)is not dict or set(d)!=expected or d.get('schema')!=SCHEMA
                or d.get('training_admission')is not False):
            raise ValueError('Exact complete non-admitting source export schema required')
        if type(d['total_native_prefix_tokens'])is not int or not 1<=d['total_native_prefix_tokens']<=MAX_TOTAL_PREFIX_TOKENS:
            raise ValueError('Positive exact bounded native prefix token count required')
        if _wire(d)!=self.raw:raise ValueError('Canonical source export bytes required')
        if type(d['parents'])is not list or any(type(p)is not int for p in d['parents']):
            raise ValueError('Exact integer source parent references required')
        trace=_trace(d['source_trace']);nodes,parents,directions,_,payloads=_layout(trace)
        if (d['split']!=trace.split or d['groups']!=list(trace.groups)or d['group_policy_sha256']!=trace.group_policy_sha256
                or d['source_trace_sha256']!=sha256(_wire(asdict(trace))).hexdigest()
                or d['skeleton_policy_sha256']!=SKELETON_POLICY or d['evidence_kind']!=_evidence(trace)
                or d['node_ids']!=[n.key for n in nodes]or d['parents']!=list(parents)
                or d['directions']!=list(directions)or d['statuses']!=[n.status for n in nodes]
                or d['payload_hex']!=[p.hex()for p in payloads]):
            raise ValueError('Source trace/split/groups/prefix/skeleton content binding mismatch')
        _text(d['source_revision']);_sha(d['source_evidence_sha256'])
        if type(d['original_ordinal'])is not int or d['original_ordinal']<0:
            raise ValueError('Exact original source ordinal required')
        enc=d['encoder'];book=d['codebook']
        _check_encoder_codebook(enc,book)
        graph,task,_=prefix_native_graph(trace,())
        if d['context_sha256']!=forest_context_sha256(graph,task,source_snapshot_sha256=enc['source_snapshot_sha256']):
            raise ValueError('Source root-only context binding mismatch')
        self._latents(d)

    @staticmethod
    def _latents(d):
        shape=d['latent_shape'];dtype=d['latent_dtype'];raw=d['latent_hex']
        if (type(shape)is not list or shape!=[len(d['node_ids']),d['encoder']['dimension']]
                or any(type(n)is not int for n in shape)or dtype not in _DTYPES or dtype!=d['encoder']['dtype']
                or type(raw)is not str or sys.byteorder!='little'):
            raise ValueError('Exact finite source coordinate shape/dtype required')
        size=shape[0]*shape[1]*(4 if dtype=='torch.float32'else 8)
        if len(raw)!=2*size or re.fullmatch('[0-9a-f]*',raw)is None:
            raise ValueError('Exact complete little-endian tensor bytes required')
        value=torch.frombuffer(bytearray.fromhex(raw),dtype=_DTYPES[dtype]).reshape(shape).clone()
        if not torch.isfinite(value).all():raise ValueError('Finite source coordinates required')
        return value

    def document(self):
        self.__post_init__();return _parse(self.raw)

    def latents(self):
        return self._latents(self.document())

    def payloads(self):
        return tuple(bytes.fromhex(p)for p in self.document()['payload_hex'])

    def save(self,path):
        self.__post_init__();path=Path(path).absolute()
        if any(p.is_symlink()for p in (path,*path.parents)):raise ValueError('No symlink export destination')
        with path.open('xb')as stream:stream.write(self.raw)

    @classmethod
    def load(cls,path,*,expected_sha256):
        _sha(expected_sha256);path=Path(path).absolute()
        if any(p.is_symlink()for p in(path,*path.parents))or not path.is_file():
            raise ValueError('Original regular export file required')
        if path.stat().st_size>MAX_BYTES:raise ValueError('Source export exceeds read budget')
        with path.open('rb')as stream:raw=stream.read(MAX_BYTES+1)
        return cls(raw,expected_sha256)

    def training_batch(self,heads,spec):
        """Prepare only original train data, without granting full-owner admission."""
        from mora3.model.forest_training import (ForestTrainingHeads,ForestTrainingSpec,ForestBatchIdentity,
            ForestTrainingBatch,FlowSupervision,TokenSupervision,ForestTopologySupervision,preflight_forest_batch)
        from mora3.model.forest_token_decoder import prepare_forest_token_targets
        d=self.document();enc=d['encoder']
        if d['split']!='train':raise ValueError('Original heldout source export cannot become train supervision')
        if enc['encoder_status']=='untrained-diagnostic'and d['evidence_kind']!='synthetic_fixture':
            raise ValueError('Untrained diagnostic encoder is not a qualified real training export')
        if type(heads)is not ForestTrainingHeads or type(spec)is not ForestTrainingSpec:
            raise ValueError('Exact native heads and frozen objective required')
        if (_wire(heads.decoder._codebook_binding)!=_wire(d['codebook'])
                or spec.encoder_sha256!=enc['encoder_sha256']or spec.coordinate_sha256!=enc['coordinate_sha256']
                or spec.skeleton_policy_sha256!=d['skeleton_policy_sha256']
                or spec.source_snapshot_sha256!=enc['source_snapshot_sha256']or spec.group_policy_sha256!=d['group_policy_sha256']):
            raise ValueError('Exact source codebook/encoder/coordinate/group/objective binding required')
        ref=heads.flow.readout.weight;values=self.latents()
        if values.dtype!=ref.dtype:raise ValueError('Source dtype cannot be silently converted')
        values=values.to(ref.device);trace=_trace(d['source_trace']);graph,task,_=prefix_native_graph(trace,())
        roots=tuple(i for i,p in enumerate(d['parents'])if p==-1)
        known=torch.zeros_like(values);known[list(roots)]=values[list(roots)]
        template=ForestState(tuple(d['node_ids']),tuple(d['parents']),tuple(d['statuses']),known,roots,
            tuple(d['directions']),d['context_sha256'],enc['coordinate_sha256'],'declared')
        payloads=tuple(None if i in roots else p for i,p in enumerate(self.payloads()))
        target,mask=prepare_forest_token_targets(heads.decoder.axis,(template,),(payloads,),max_slots=heads.decoder.spec.max_slots)
        available=torch.ones_like(values[None],dtype=torch.bool);available[:,list(roots)]=False
        identity=ForestBatchIdentity(enc['source_snapshot_sha256'],enc['coordinate_sha256'],enc['encoder_sha256'],
            d['group_policy_sha256'],d['skeleton_policy_sha256'],d['source_revision'],(trace.identity,),
            (d['original_ordinal'],),(trace.groups,),trace.split,d['evidence_kind'],self.sha256)
        batch=ForestTrainingBatch(identity,graph,task,template,FlowSupervision(values[None],available),
            TokenSupervision(values[None].clone(),target,mask,sha256(_wire(d['payload_hex'])).hexdigest()))
        if spec.topology_weight:
            from mora3.model.forest_topology import NativeForestTopologyHead
            if type(heads.topology)is not NativeForestTopologyHead:
                raise ValueError('Registered native topology policy required for source supervision')
            batch=replace(batch,topology=ForestTopologySupervision(len(roots),(tuple(d['parents']),),
                heads.topology.spec.construction_policy_sha256,self.sha256))
        preflight_forest_batch(heads,spec,batch)
        return batch


def load_source_codebook(path,*,expected_sha256):
    """Reconstruct the exact frozen native decoder asset in a fresh process."""
    _sha(expected_sha256);path=Path(path).absolute()
    if any(p.is_symlink()for p in(path,*path.parents))or not path.is_file():
        raise ValueError('Original regular codebook export file required')
    if path.stat().st_size>MAX_BYTES:raise ValueError('Codebook export exceeds read budget')
    with path.open('rb')as stream:raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES or sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Exact complete codebook export bytes required')
    d=_parse(raw)
    if (type(d)is not dict or set(d)!={'schema','axis','encoder','binding','values_hex'}
            or d['schema']!='Mora3.FrozenSourceCodebookExport/v1' or _wire(d)!=raw):
        raise ValueError('Exact canonical codebook export schema required')
    _check_encoder_codebook(d['encoder'],d['binding'])
    try:
        axis=dict(d['axis'])
        for name in ('learned_hex','base_controls','controls'):axis[name]=tuple(axis[name])
        axis=ForestTokenAxis(**axis)
    except (KeyError,TypeError)as exc:
        raise ValueError('Complete native codebook token axis required')from exc
    b=d['binding'];text=d['values_hex'];shape=b['shape'];dtype=b['dtype']
    if dtype not in _DTYPES or sys.byteorder!='little':raise ValueError('Exact little-endian floating codebook dtype required')
    n=shape[0]*shape[1]*(4 if dtype=='torch.float32'else 8)
    if type(text)is not str or len(text)!=2*n or re.fullmatch('[0-9a-f]*',text)is None:
        raise ValueError('Exact complete source embedding bytes required')
    values=torch.frombuffer(bytearray.fromhex(text),dtype=_DTYPES[dtype]).reshape(shape).clone()
    book=ForestTokenCodebook(axis,values,b['source_snapshot_sha256'],b['coordinate_sha256'],
        b['encoder_sha256'],b['export_sha256'],b['evidence_sha256'])
    if _wire(book.binding())!=_wire(b):raise ValueError('Actual codebook values or axis differ from committed source')
    return book
