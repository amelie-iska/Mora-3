"""Train-condition-only ConvexTok fitting and checkpoint-bound payload encoding.

Uses the existing bounded LP/Det reference; no annotation label is a fit input.
The caller must supply source-validated prediction views from the native reader.
This does not certify family-heldout splits or generation eligibility.
"""
from dataclasses import dataclass
import hashlib
import json
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model.payload import PayloadEncoder
from mora3.tokenization.convex_vocab import fit_vocabulary, substring_candidates
from mora3.tokenization.shortest_path import ByteTokenizer


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


@dataclass(frozen=True)
class AnnotationVocabulary:
    """Immutable serialized fit/binding; tensor checkpoints bind exact ordered IDs."""
    specification_json: str

    @property
    def certificate(self):
        return json.loads(self.specification_json)['certificate']

    def tokenizer(self):
        spec=json.loads(self.specification_json)
        return ByteTokenizer(tuple(bytes.fromhex(x) for x in spec['learned_hex']),
                             controls=spec['controls'])


def fit_annotation_vocabulary(views, *, learned_budget=32, max_length=3,
                              max_candidates=4096, max_bytes=100_000,
                              max_edges=500_000, time_limit=120.):
    """Fit complete canonical sequence conditions only, preserving record frequency.

    Candidate universe is ALL within-sequence substrings of lengths 2..max_length.
    Caps reject instead of truncating. Targets/annotation metadata are never read.
    Source identity binds raw page hashes plus accession and full sequence hashes.
    """
    views=tuple(views)
    if not views or any(not isinstance(v,AnnotationPredictionView) for v in views):
        raise ValueError('nonempty source-validated prediction views required')
    if type(learned_budget) is not int or learned_budget<0:
        raise ValueError('nonnegative integer learned budget required')
    corpus=[];sources=[]
    for v in views:
        p=v.provenance()
        if p.get('split')!='train' or p.get('license')!='CC-BY-4.0':
            raise ValueError('source train-only rights scope required')
        raw=v.inputs()['canonical_sequence'].encode('utf-8')
        corpus.append(raw)
        sources.append(dict(accession=v.accession,source_ref=p['source_ref'],
                            sequence_sha256=hashlib.sha256(raw).hexdigest()))
    if sum(map(len,corpus))>max_bytes:raise ValueError('byte limit exceeded')
    candidates=substring_candidates(corpus,max_length=max_length,max_candidates=max_candidates)
    controls=('pad','node','edge')
    fit=fit_vocabulary(corpus,candidates,vocab_size=256+len(controls)+learned_budget,
                       controls=controls,rounding='det',max_bytes=max_bytes,
                       max_edges=max_edges,time_limit=time_limit)
    spec=dict(version=1,method='ConvexTok-2605.22821v1-LP-Det',
              learned_hex=[x.hex() for x in fit.learned],controls=controls,
              sources=sorted(sources,key=_canonical),
              configuration=dict(learned_budget=learned_budget,max_length=max_length,
                  max_candidates=max_candidates,max_bytes=max_bytes,max_edges=max_edges,
                  time_limit=time_limit),
              certificate=dict(lp_objective=fit.lp_objective,dual_objective=fit.dual_objective,
                  rounded_objective=fit.rounded_objective,
                  max_constraint_violation=fit.max_constraint_violation,
                  dual_residual=fit.dual_residual,edge_count=fit.edge_count,
                  vertex_count=fit.vertex_count,candidates=len(candidates)))
    return AnnotationVocabulary(_canonical(spec))


class AnnotationPayloadEncoder(PayloadEncoder):
    """Existing shared lossy payload reference with learned, immutable ID binding.

    Sequence bytes roundtrip losslessly through the tokenizer; positional pooling
    remains lossy. This is not residue-token expansion or a complete sequence LM.
    """
    def __init__(self,vocabulary,width,*,max_tokens=8192,pooling='positional',
                 parameterization_policy='legacy',base_width=None):
        if not isinstance(vocabulary,AnnotationVocabulary):
            raise ValueError('fitted annotation vocabulary required')
        self._vocabulary_json=vocabulary.specification_json
        super().__init__(vocabulary.tokenizer(),width,max_tokens=max_tokens,pooling=pooling,
                         parameterization_policy=parameterization_policy,base_width=base_width)

    def get_extra_state(self):
        result=dict(version=1,vocabulary=self._vocabulary_json,
                    max_tokens=self.max_tokens,pooling=self.pooling)
        if self._payload_parameterization is not None:
            result.update(version=2,parameterization=self._payload_mup_configuration())
        return result

    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('annotation vocabulary/payload checkpoint mismatch')
