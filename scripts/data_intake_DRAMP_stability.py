"""Peptide native measurement projection, no task or physical property admission."""
import hashlib,json,re
from decimal import Decimal
IDENTITY_FIELDS=('Sequence','N-terminal_Modification','C-terminal_Modification','Other_Modifications','Stereochemistry','Linear/Cyclic/Branched')
SCALAR=re.compile(r'\s*(?:the\s+)?half[- ](?:life|live)\s*(?:is\s*)?(?P<relation>>=|<=|>|<|≥|≤|=)?\s*(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>seconds|minutes|hours|second|minute|hour|min|h|s)\.?\s*',re.I)

def strict_native_half_life(text):
    """Parse only a complete single scalar/censor statement in its literal unit.

    Fractions remaining, functional activity, approximations, uncertainties,
    matrix-dependent multiple outcomes and extra narrative remain unprojected.
    A syntax candidate is not an independently verified measured endpoint.
    """
    if not isinstance(text,str):return None
    match=SCALAR.fullmatch(text)
    if match is None:return None
    relation=match['relation'] or '='
    return dict(schema='mora3.NativeReportedHalfLifeScalarCandidate/v1',source_literal=text,value_literal=match['value'],unit_literal=match['unit'],relation=relation,exact_value_observed=relation=='=',positive_value_candidate=Decimal(match['value'])>0,normalized_value=None,source_evidence_role='database_curated_report;primary_assay_verification_pending',task_admitted=False,model_use=False)

def material_descriptor(native_fields):
    missing=[k for k in IDENTITY_FIELDS if k not in native_fields]
    if missing:raise ValueError('missing native identity fields; do not assume free modifications: '+','.join(missing))
    descriptor={k:native_fields[k] for k in IDENTITY_FIELDS}
    raw=json.dumps(descriptor,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return dict(descriptor=descriptor,native_descriptor_sha256=hashlib.sha256(raw).hexdigest(),atomic_graph_verified=False,source_names_or_accessions_not_identity_proof=True)

def hidden_half_life_predictors(native_fields):
    """Source identity+assay labels only; outcome/names/IDs/citations stay audit."""
    identity=material_descriptor(native_fields)['descriptor'];context={k:native_fields.get(k) for k in ['Type','Assay']}
    unknown={'','not mentioned','not found','no entry found'}
    available={k:isinstance(v,str) and v.strip().casefold() not in unknown for k,v in context.items()}
    return json.dumps(dict(native_peptide_identity=identity,source_assay_context=context,source_assay_context_applicability=available),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
