"""Complete guarded UniProt annotation/sequence directions on the registered byte decoder."""
import copy
import json
from torch import nn

from mora3.data.provenance import Provenance
from mora3.data.schema import GraphRecord,Measurement,Node
from mora3.data.tasks import ConditionalTask
from .mixed_payload_projection import exact_wire
from .pubchem_decoder import PubChemDecoder
from .source_byte_objective import source_byte_objective,validate_byte_targets

_GUARD='c9ef860d0695222c03fce33568f29e243c2efbb19d4c80d3c215f4b5496f685d'
_FIELDS=('source_curated_function','source_curated_catalytic_activity','source_protein_name','source_GO_identifiers')
_FORWARD=('amino_acid_sequence','organism_taxon')
_INVERSE=('source_experimentally_supported_function_texts','source_experimental_associated_GO_ids',
 'source_explicit_experimental_NOT_GO_ids','source_experimentally_supported_EC_ids','source_kinetic_conditions','organism_taxon_id')
_TARGET_BYTES_BOUND=35688
_OUTPUTS=('native_canonical_amino_acid_sequence',)+_FIELDS


def _wire(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _known_graph(fields,output):
    if type(fields) is not dict or output not in _OUTPUTS:
        raise ValueError('exact native known fields and observed output direction required')
    required=_INVERSE if output==_OUTPUTS[0] else _FORWARD
    if set(fields)!=set(required):
        raise ValueError('target-free known UniProt fields must match the exact native source whitelist')
    if output==_OUTPUTS[0]:
        taxon=fields['organism_taxon_id']
        if type(taxon) is not int or taxon<1 or any(type(fields[k]) is not tuple for k in required[:-1]):
            raise ValueError('exact typed inverse known conditions required')
    else:
        seq,taxon=fields['amino_acid_sequence'],fields['organism_taxon']
        if (type(taxon) is not int or taxon<1 or type(seq) is not str or not seq or not seq.isascii()):
            raise ValueError('complete known sequence and native integer taxon required')
    known=tuple(Measurement(k,fields[k] if type(fields[k])is str else _wire(fields[k]),None,'observed',True) for k in required)
    node=Node('known-UniProt-annotation-context',known+(
        Measurement('requested_source_output',output,None,'observed',True),
        Measurement('requested_native_target',None,None,'missing',False)))
    rec=GraphRecord('source-uniprot-annotation-task-input',(node,),(),
        (Provenance('native-model-input','uniprot-exact-annotation-v1','input','0'*64,'source-admission-external'),))
    task=ConditionalTask(tuple((('node',0,m),'condition')for m in (*required,'requested_source_output'))+
        ((('node',0,'requested_native_target'),'target'),),topology_known=True)
    return rec,task


class SharedUniProtNativeAnnotationModel(nn.Module):
    """No added weights; known source inputs and complete byte targets use one shared decoder."""
    def __init__(self,decoder,*,max_batch_size=128,max_total_target_bytes=64*1024*1024):
        super().__init__()
        if (type(decoder) is not PubChemDecoder or decoder.get_extra_state()['version']!=2
                or type(max_batch_size)is not int or max_batch_size<1
                or type(max_total_target_bytes)is not int or max_total_target_bytes<_TARGET_BYTES_BOUND):
            raise ValueError('complete exact native byte decoder and declared resource bounds required')
        self.decoder=decoder;self._owner=(decoder,)
        self.max_batch_size=max_batch_size;self.max_total_target_bytes=max_total_target_bytes
        self._configuration=self._local()

    def _local(self):
        return dict(schema='Mora3.SharedUniProtNativeAnnotationModel/v1',source_prospective_guard_sha256=_GUARD,
            forward_target_fields=_FIELDS,inverse_target_field=_OUTPUTS[0],
            max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes,
            forward_targets='separate exact native UTF8 fields; ordered GO IDs fixed 10 ASCII bytes each; append one EOS',
            inverse_target='complete canonical source amino-acid sequence bytes and one EOS',
            conditions='only typed native known fields and requested output; omit all target/audit/accession/path inputs',
            missing='no loss or artificial empty label; an entirely missing output batch is explicitly refused',
            raw_source_roles_preserved=True,training_admission=False)

    def get_extra_state(self):
        if (self.decoder is not self._owner[0] or set(self._modules)!={'decoder'} or self._parameters or self._buffers
                or not exact_wire(self._local(),self._configuration)):
            raise ValueError('exact native UniProt shared owner or role policy drift')
        return dict(**copy.deepcopy(self._configuration),decoder=self.decoder.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('exact native UniProt checkpoint source binding required')

    def _score(self,rows,output,*,chunk_size,recompute):
        self.get_extra_state();rows=tuple(rows)
        if not 1<=len(rows)<=self.max_batch_size:raise ValueError('bounded complete native task batch required')
        visible=[];targets=[]
        for row in rows:
            raw=row.canonical_sequence.encode('ascii') if output==_OUTPUTS[0] else row.target_bytes(output)
            if raw is None:continue
            visible.append(row.model_condition_fields());targets.append(raw)
        if not visible:raise ValueError('this native output is unobserved in the whole batch')
        targets=validate_byte_targets(targets,max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes,
            chunk_size=chunk_size,recompute=recompute)
        # Validate all known fields and source targets before any model operation.
        graphs=tuple(_known_graph(v,output) for v in visible)
        shared=self.decoder.shared
        state=shared.backbone([shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers())for r,t in graphs])[0][:,0]
        return dict(loss=source_byte_objective(self.decoder,state,targets,chunk_size=chunk_size,recompute=recompute,
            max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes),observed=len(targets),
            missing=len(rows)-len(targets),exact_target_UTF8_bytes=sum(map(len,targets)),native_output=output)

    def inverse_objective(self,view,batch,*,split='train',chunk_size=1024,recompute=False):
        from mora3.data.uniprot_native_annotations import uniprot_native_inverse_rows
        rows=uniprot_native_inverse_rows(view,batch,split=split)
        return self._score(rows,_OUTPUTS[0],chunk_size=chunk_size,recompute=recompute)

    def forward_objective(self,view,batch,field,*,split='train',chunk_size=1024,recompute=False):
        from mora3.data.uniprot_native_annotations import UNIPROT_FORWARD_TARGET_FIELDS,uniprot_native_forward_rows
        if type(field)is not str or field not in _FIELDS or UNIPROT_FORWARD_TARGET_FIELDS!=_FIELDS:
            raise ValueError('exact one requested native forward field required')
        rows=uniprot_native_forward_rows(view,batch,split=split)
        return self._score(rows,field,chunk_size=chunk_size,recompute=recompute)


def uniprot_native_annotation_parameter_roles(model,*,identifier_axis):
    from .mu_optimizer import _storage_signature
    from .parameterization import parameter_ledger
    from .sequence_mup import sequence_mup_parameter_roles
    if type(model) is not SharedUniProtNativeAnnotationModel:raise ValueError('exact native UniProt annotation owner required')
    model.get_extra_state()
    roles={'decoder.'+name:role for name,role in sequence_mup_parameter_roles(model.decoder,identifier_axis=identifier_axis).items()}
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
