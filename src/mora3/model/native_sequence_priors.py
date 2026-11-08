"""Actual current-view native sequence priors, sharing existing decoder weights.

The public path invokes the source-owned exact-row extractor first; it does not
forge pilot-native views or reinterpret historical flags. Complete source strings
and EOS enter only the recurrent targets. Native input domains retain their
source task semantics; IDs, lengths, hashes and audit fields are excluded.
"""
import copy,json
import torch
from torch import nn
from mora3.data.schema import GraphRecord,Node,Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.data.nucleotide_prior import SOURCES
from mora3.data.rnacentral_prior import DOMAIN as RNACENTRAL_DOMAIN
from .nucleotide_decoder import NucleotideDecoder
from .pubchem_decoder import PubChemDecoder,pubchem_condition_graph
from .source_byte_objective import validate_byte_targets,_source_token_objective
from .mixed_payload_projection import exact_wire

_NUCLEOTIDES={'dna':SOURCES['DNA-native-reference-prior'][0],
 'rfam_rna':SOURCES['Rfam15.1-native-reference-prior'][0],'rnacentral_rna':RNACENTRAL_DOMAIN}
_BYTES={'protein','native_smiles','smiles_to_selfies','selfies_to_smiles'}


def _protein_graph(c):
    if c!={'prior_domain':'protein_reference_native_amino_acids'}or type(c)is not dict:
        raise ValueError('exact source-task protein prior condition required')
    raw=json.dumps(c,sort_keys=True,separators=(',',':'))
    record=GraphRecord('input',(Node('protein-reference-prior',(
        Measurement('condition',raw,None,'observed',True),
        Measurement('requested_output',None,None,'missing',False))),),(),
        (Provenance('native-model-input','protein-source-prior-v1','input','0'*64,'source-admission-external'),))
    task=ConditionalTask(((('node',0,'condition'),'condition'),(('node',0,'requested_output'),'target')),topology_known=True)
    return record,task


class NativeSequencePriorModel(nn.Module):
    def __init__(self,nucleotide,decoder,*,max_batch_size=64,max_total_target_bytes=64*1024*1024):
        super().__init__()
        if (type(nucleotide)is not NucleotideDecoder or type(decoder)is not PubChemDecoder
                or nucleotide.get_extra_state()['version']!=2 or decoder.get_extra_state()['version']!=2
                or nucleotide.shared is not decoder.shared):
            raise ValueError('exact same shared primary native-symbol and byte decoder owners required')
        if any(type(n)is not int or n<1 for n in(max_batch_size,max_total_target_bytes)):
            raise ValueError('positive complete native prior resource bounds required')
        self.nucleotide=nucleotide;self.decoder=decoder;self._owners=(nucleotide,decoder)
        self.max_batch_size=max_batch_size;self.max_total_target_bytes=max_total_target_bytes
        self._configuration=self._local_state()

    def _local_state(self):
        return dict(schema='Mora3.CurrentNativeSequencePriorModel/v1',
            routes=tuple(_NUCLEOTIDES)+tuple(sorted(_BYTES)),
            architecture='exact-existing-native-symbol-or-PubChem-byte-GRU;sharedannotationencoder;no-newweights',
            objective='full-source-symbol-or-byte-tokens+EOS;globaltarget-tokenmean;completehistory;optionalcheckpointchunks',
            condition='source-task-semantic-whitelist-only;no-family-ID-audit-or-targetderivedlength',
            source_binding='source_native_prior_views(require_native_rows);no-pilotview-or-flag-relabel',
            training_admission='external-global-model-data-campaign;none-issued-here',
            resource_bounds=dict(max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes))

    def get_extra_state(self):
        if (self.nucleotide is not self._owners[0]or self.decoder is not self._owners[1]
                or self.nucleotide.shared is not self.decoder.shared or not exact_wire(self._local_state(),self._configuration)):
            raise ValueError('current native prior source/owner/resource drift')
        return dict(**copy.deepcopy(self._configuration),nucleotide=self.nucleotide.get_extra_state(),decoder=self.decoder.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('current native prior checkpoint mismatch')

    def _validate_record(self,r):
        from mora3.data.source_native_priors import NativePriorView
        if type(r)is not NativePriorView:raise ValueError('exact typed current native prior required')
        expected_datatype={'dna':'DNA','rfam_rna':'RNA','rnacentral_rna':'RNA','protein':'protein',
            'native_smiles':'native_SMILES','smiles_to_selfies':'standard_SELFIES','selfies_to_smiles':'native_SMILES'}
        if r.datatype!=expected_datatype.get(r.route):raise ValueError('exact native prior datatype/route required')
        c=r.condition_inputs()
        if r.route in _NUCLEOTIDES:
            if (type(c)is not dict or set(c)!={'prior_domain','source_organism'}
                    or c['prior_domain']!=_NUCLEOTIDES[r.route]
                    or(c['source_organism']is not None and(type(c['source_organism'])is not str or not c['source_organism']))
                    or(r.route=='rnacentral_rna'and c['source_organism']is not None)):
                raise ValueError('exact current native nucleotide condition scope required')
            return self.nucleotide,self.nucleotide.codec.encode(r.target_bytes.decode('utf8'))
        if r.route not in _BYTES:raise ValueError('unsupported exact current native prior route')
        if r.route=='protein':_protein_graph(c)
        else:pubchem_condition_graph(c)
        if not r.target_bytes:raise ValueError('complete nonempty native source target required')
        d=self.decoder;return d,[d.codec.bos,*d.codec.tokenizer.encode(r.target_bytes),d.codec.eos]

    def _condition(self,r):
        if r.route in _NUCLEOTIDES:return self.nucleotide.condition((r.condition_inputs(),))
        if r.route!='protein':return self.decoder.condition((r.condition_inputs(),))
        record,task=_protein_graph(r.condition_inputs());shared=self.decoder.shared
        graph=shared.payload.forward_task(record,task=task,identifiers=shared.sequence_identifiers())
        return shared.backbone([graph])[0][:,0]

    def _reference_objective(self,records,*,chunk_size=1024,recompute=False):
        """Private equation reference; public source extraction authenticates rows."""
        self.get_extra_state();records=tuple(records)
        validate_byte_targets((r.target_bytes for r in records),max_batch_size=self.max_batch_size,
            max_total_target_bytes=self.max_total_target_bytes,chunk_size=chunk_size,recompute=recompute)
        encoded=tuple(self._validate_record(r)for r in records) # All records before any NN.
        total=None;count=0
        for r,(d,ids)in zip(records,encoded):
            context=self._condition(r)
            loss=_source_token_objective(d,context,(ids,),chunk_size=chunk_size,recompute=recompute)
            n=len(ids)-1;count+=n;total=loss*n if total is None else total+loss*n
        return total/count

    def objective(self,source_view,batch,*,split='train',chunk_size=1024,recompute=False):
        from mora3.data.source_native_priors import source_native_prior_views
        records=source_native_prior_views(source_view,batch,split=split)
        return self._reference_objective(records,chunk_size=chunk_size,recompute=recompute)


def native_sequence_prior_parameter_roles(model,*,identifier_axis):
    from .sequence_mup import sequence_mup_parameter_roles
    from .parameterization import parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeSequencePriorModel:raise ValueError('exact current native prior owner required')
    model.get_extra_state()
    if set(model._modules)!={'nucleotide','decoder'}or model._parameters or model._buffers:
        raise ValueError('exact current native prior alias-only wrapper required')
    roles={prefix+'.'+n:r for prefix,owner in(('nucleotide',model.nucleotide),('decoder',model.decoder))
        for n,r in sequence_mup_parameter_roles(owner,identifier_axis=identifier_axis).items()}
    names=dict(model.named_parameters(remove_duplicate=False));by_id={}
    for n,r in roles.items():
        key=id(names[n])
        if key in by_id and by_id[key]!=r:raise ValueError('current native prior shared finite axis/role mismatch')
        by_id[key]=r
    _storage_signature(model);parameter_ledger(model,roles);return roles
