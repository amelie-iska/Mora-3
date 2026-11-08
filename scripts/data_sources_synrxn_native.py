"""Lossless native SynRXN row projection; no task admission or predictor enablement."""
import json

def compact(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def project_native_row(columns,cells,source_sha256,ordinal):
    if len(columns)!=len(cells):
        raise ValueError("native row width mismatch")
    if len(set(columns))!=len(columns):
        raise ValueError("duplicate native column names")
    if not all(isinstance(x,str) for x in columns+cells):
        raise ValueError("native literal cell strings required")
    d=dict(zip(columns,cells))
    split=d.get("split")
    if split=="train":
        role="SOURCE_TRAIN_CANDIDATE_PENDING_EXPOSURE"
    elif split in ("val","valid","validation","test"):
        role="EVALUATION_EXPOSURE_CONTROL_ONLY"
    else:
        role="UNKNOWN_SOURCE_PROTOCOL_QUARANTINE"
    targets=[x for x in columns if x in ("rc","reagent","ground_truth","label","class","reaction_center")]
    reasons=[]
    if role=="UNKNOWN_SOURCE_PROTOCOL_QUARANTINE":reasons.append("unknown_or_missing_native_split")
    if not d.get("r_id"):reasons.append("missing_native_row_identifier")
    return {"source_file_sha256":source_sha256,"source_row":int(ordinal),"reaction_id":d.get("r_id"),
            "native_columns_json":compact(columns),"native_cells_json":compact(cells),
            "native_split":split,"source_protocol_role":role,"gold_target_columns_json":compact(targets),
            "default_predictor_columns_json":"[]","quality_reasons_json":compact(reasons),
            "task_admitted":False,"adapter_supported":False,"model_use_allowed":False}
