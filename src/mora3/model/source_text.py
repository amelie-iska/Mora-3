"""Target-free native text conditions on the existing shared byte decoder.

This supplies reference losses, not source/split authorization. HumanEval is an
evaluation-only direction. Full UTF8 targets and EOS are retained; optional
activation recomputation carries recurrent state without detaching history.
"""
import copy
import json

import torch
from torch import nn
from .source_byte_objective import source_byte_objective,validate_byte_targets

from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .pubchem_decoder import PubChemDecoder
from .mixed_payload_projection import exact_wire


def text_condition_graph(condition):
    if type(condition) is not dict:
        raise ValueError('explicit target-free source text condition required')
    task=condition.get('task')
    if task=='source_language_prior':
        if (set(condition)!={'task','language','domain'}
                or condition['language'] not in ('he','en')
                or condition['domain'] not in ('biomedical','biblical','general','mathematics','code')):
            raise ValueError('exact native language/domain prior condition required')
    elif task=='oshb_source_root_list':
        if set(condition)!={'task','word'} or type(condition['word'])is not str or not condition['word']:
            raise ValueError('complete observed native word only; no root/lemma/morphology/audit conditions')
    elif task in ('gsm8k_answer','humaneval_solution'):
        field='question' if task=='gsm8k_answer' else 'prompt'
        if set(condition)!={'task',field} or type(condition[field]) is not str or not condition[field]:
            raise ValueError('complete known native question/prompt required without answer/audit')
    else:
        raise ValueError('unsupported source text task')
    raw=json.dumps(condition,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    raw.encode('utf8')
    record=GraphRecord('input',(Node('source-text',(
        Measurement('condition',raw,None,'observed',True),
        Measurement('requested_output',None,None,'missing',False))),),(),
        (Provenance('native-model-input','source-text-v1','input','0'*64,'source-admission-external'),))
    roles=ConditionalTask(((("node",0,"condition"),'condition'),
                           (("node",0,"requested_output"),'target')),topology_known=True)
    return record,roles


class SharedSourceTextModel(nn.Module):
    """No new weights: exact same encoder and byte embedding/GRU/readout owner."""
    def __init__(self,decoder,*,max_batch_size=64,max_total_target_bytes=64*1024*1024,root_list_policy='disabled'):
        super().__init__()
        if type(decoder) is not PubChemDecoder or decoder.get_extra_state()['version']!=2:
            raise ValueError('exact native primary byte decoder owner required')
        if any(type(v) is not int or v<1 for v in (max_batch_size,max_total_target_bytes)):
            raise ValueError('positive explicit resource refusal bounds required')
        if root_list_policy not in('disabled','observed-native-root-list-v1'):
            raise ValueError('explicit known source root-list policy required')
        self.root_list_policy=root_list_policy
        self.decoder=decoder
        self._owner=(decoder,)
        self.max_batch_size=max_batch_size
        self.max_total_target_bytes=max_total_target_bytes
        self._configuration=self._local_state()

    def _local_state(self):
        if self.root_list_policy not in('disabled','observed-native-root-list-v1'):
            raise ValueError('source root-list policy drift')
        state=dict(schema='Mora3.SharedSourceText/v1',
            architecture='exact-shared-native-PubChem-byte-embedding-GRU-readout-and-annotation-encoder',
            conditions='strict-task-whitelist;complete-question-prompt-or-language-domain-only',
            targets='full-native-UTF8-bytes-plus-EOS;no-normalization-or-truncation',
            objective='all-next-token-and-EOS-cross-entropy;global-token-mean',
            recurrent_history='carried-without-detach;optional-nonreentrant-checkpoint-recomputation',
            humaneval='evaluation-only;untrusted-source-code-never-executed',
            source_admission='external-immutable-reader-global-split-and-campaign;none-issued-here',
            resource_bounds=dict(max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes))
        if self.root_list_policy!='disabled':
            state.update(schema='Mora3.SharedSourceText/v2',root_list_policy=self.root_list_policy,
                root_conditions='complete-observed-native-word-only;lemma-morphology-lexicalIDs-roots-audit-excluded',
                root_targets='nonempty-native-ordered-literal-root-string-list;canonicalUTF8JSON+EOS;repeats-preserved',
                unlisted_roots='unlabeled-no-loss;not-empty-list-or-biological-negatives')
        return state

    def get_extra_state(self):
        if self.decoder is not self._owner[0] or not exact_wire(self._local_state(),self._configuration):
            raise ValueError('shared text owner/resource policy drift')
        return dict(**copy.deepcopy(self._configuration),decoder=self.decoder.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('shared text checkpoint mismatch')

    def condition(self,conditions):
        conditions=tuple(conditions)
        if not conditions or len(conditions)>self.max_batch_size:
            raise ValueError('explicit nonempty bounded native text batch required')
        if any(type(c)is dict and c.get('task')=='oshb_source_root_list' for c in conditions) and self.root_list_policy=='disabled':
            raise ValueError('observed native root lists require explicit opt-in policy')
        records=tuple(text_condition_graph(c) for c in conditions)
        shared=self.decoder.shared
        graphs=[shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers()) for r,t in records]
        return shared.backbone(graphs)[0][:,0]

    def _objective(self,conditions,targets,*,chunk_size,recompute,allow_evaluation_only):
        self.get_extra_state()
        conditions=tuple(conditions);targets=tuple(targets)
        if (not conditions or len(conditions)!=len(targets) or len(conditions)>self.max_batch_size
                or any(type(raw) is not bytes for raw in targets)
                or sum(map(len,targets))>self.max_total_target_bytes
                or type(chunk_size) is not int or chunk_size<1 or type(recompute) is not bool):
            raise ValueError('complete native target bytes and explicit compute/resource bounds required')
        # Validate every record before any neural work or target allocation.
        for condition,raw in zip(conditions,targets):
            text_condition_graph(condition);raw.decode('utf8')
            if condition['task']=='oshb_source_root_list':
                if self.root_list_policy!='observed-native-root-list-v1':
                    raise ValueError('observed native root lists require explicit opt-in policy')
                roots=json.loads(raw)
                if (type(roots)is not list or not roots or any(type(root)is not str or not root for root in roots)
                        or json.dumps(roots,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')!=raw):
                    raise ValueError('nonempty complete native ordered root-string list required; unknown roots have no loss')
            if condition['task']=='humaneval_solution' and not allow_evaluation_only:
                raise ValueError('HumanEval canonical solutions are evaluation-only')
        context=self.condition(conditions)
        return source_byte_objective(self.decoder,context,targets,chunk_size=chunk_size,recompute=recompute,
            max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes)

    def objective(self,conditions,targets,*,chunk_size=1024,recompute=False):
        return self._objective(conditions,targets,chunk_size=chunk_size,recompute=recompute,allow_evaluation_only=False)

    def evaluation_objective(self,conditions,targets,*,chunk_size=1024,recompute=False):
        return self._objective(conditions,targets,chunk_size=chunk_size,recompute=recompute,allow_evaluation_only=True)
