"""Pinned PaRoutes computed lookup candidates, with bounded original JSON replay.

This is a source inspection interface.  Computed SMARTS contexts are neither
molecular product inputs nor measured reactions, and never authorize model use.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

BATCH = "outputs/data-sources/20261101-retrosynthesis-source-v1"
HANDOFF_PATH = BATCH + "/handoff-complete-ORD-PaRoutes-native-reaction-sources-v1.json"
HANDOFF_SHA256 = "62ea114eeab722e8363884b4f3704143b7f0d38928791b01d2fa39694dad72d7"
MANIFEST_PATH = BATCH + "/curated/PaRoutes-whole15-native-QC-v1/manifest.json"
MANIFEST_SHA256 = "695f7ff874585f5c5f28053f7a5ed3ba36c5530466f101e0d5b966ef76b7099a"
PLAN_SHA256 = "5695a0edc2e72712d965b87b3f74d211f8b91d57078fb089eec69f0283870e30"
AVAILABLE_PATH = BATCH + "/PaRoutesV2-complete-data-release-v1/whole-15-original-data-available-manifest-v1.json"
AVAILABLE_SHA256 = "86181d60475a958fc779462bd5204bfb4347db16894fd34dead783f474aacb2e"
PROOF_SHA256 = "c4f0bfe52bab987929db580a5b8b57e33e92d2c7edd21abe08a32c456edfb119"
SOURCE_NAME = "uspto_lookup.json"
SOURCE_SHA256 = "3b13c548f726f56398715165f330c4979a0db26fbe363d9955e8c68d92215576"
TYPED_SHA256 = "05ae3f96f198f1fb06bedc787f6feb86eaeb10ffe6430275d9103bd75319021e"
COMPUTED = "COMPUTED_RULE_CONTEXT_NOT_INDEPENDENT_EXPERIMENT"
FIELDS = ("source_file_sha256", "source_row", "source_role", "payload_json", "quality_reasons_json",
          "source_reference_hash_exposure", "task_admitted", "adapter_supported")
MAX_PREFIX_BYTES = 4 * 1024**2
MAX_PREFIX_ENTRIES = 256
MAX_ENTRY_BYTES = 1024**2
MAX_ROW_GROUP_BYTES = 64 * 1024**2
FIXTURE_BASE = BATCH + "/curated/bounded-PaRoutes-lookup-original-prefix-fixture-v1"
IDENTITY_PATH = FIXTURE_BASE + "/lookup-original-typed-current-identity-supplement-v1.json"
IDENTITY_SHA256 = "94d117bc3a93ce8e09db3f1065254bc204cf036954085a4c982faf3c4e5fb77a"
FIXTURE_HANDOFF_PATH = FIXTURE_BASE + "/handoff-eight-PaRoutes-lookup-original-prefix-entries-v1.json"
FIXTURE_HANDOFF_SHA256 = "4f0ddefdd265701bc5e595655ceae99c981ac20cf047903381e0e49163912097"
FIXTURE_PATH = FIXTURE_BASE + "/closed-eight-native-lookup-original-and-typed-entries-v1.json"
FIXTURE_SHA256 = "ed4c84af4b376ea2c40dec148f07377808f63df9eb61bfed6479a2efba4434c0"
PREFIX_SHA256 = "451f67a90c6dd2d25d18288dc16f9671c44b0985717cee62f1f9760dbe6d9e4a"
FIELD_ROLES = {
    "_id":"source_rule_identifier_audit_only",
    "count":"source_template_count_not_independent_measurements",
    "dimer_only":"source_computed_applicability_flag_not_validated_feasibility",
    "intra_only":"source_computed_applicability_flag_not_validated_feasibility",
    "index":"native_source_index_no_automatic_join",
    "necessary_reagent":"source_computed_rule_context_not_observed_reaction_conditions",
    "reaction_smarts":"source_computed_SMARTS_not_molecule_or_unknown_gold_label",
    "references":"literal_source_rule_references_no_automatic_join_or_heldout_clearance",
    "template_set":"literal_source_template_set_not_model_partition"
}

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _int(value):
    return type(value) is int and value >= 0

def _hex(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None

def _strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate proof/typed JSON key")
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON literal")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid bounded JSON") from error

def _json_bytes(raw):
    if type(raw) is not bytes:
        raise ValueError("immutable canonical bytes required")
    value = _strict_json(raw)
    if _canonical(value) != raw:
        raise ValueError("exact canonical typed/binding bytes required")
    return value

def _identity(stat):
    return dict(dev=stat.st_dev, ino=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)

def _valid_identity(value):
    return (type(value) is dict and set(value) == {"dev","ino","size","mtime_ns","ctime_ns"}
            and all(_int(v) for v in value.values()))

def _utc_ns(value):
    delta = datetime.fromisoformat(value) - datetime(1970,1,1,tzinfo=timezone.utc)
    return ((delta.days*86400+delta.seconds)*10**6+delta.microseconds)*1000

@dataclass(frozen=True)
class NativeJSONNumber:
    """Literal JSON number: no float conversion, coercion or loss of -0/exponent."""
    lexical: str
    def __post_init__(self):
        if type(self.lexical) is not str or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", self.lexical) is None:
            raise ValueError("exact finite JSON number lexeme required")
    @property
    def kind(self):
        return "integer" if re.fullmatch("-?(?:0|[1-9][0-9]*)",self.lexical) else "noninteger_syntax"

def _native_value(value):
    if value is None or type(value) is bool or type(value) is NativeJSONNumber:
        return True
    if type(value) is str:
        return not any(0xd800 <= ord(c) <= 0xdfff for c in value)
    if type(value) is NativeJSONObject:
        return True
    return type(value) is tuple and all(_native_value(v) for v in value)

@dataclass(frozen=True)
class NativeJSONObject:
    """Ordered member pairs; duplicate keys and repeated values are retained."""
    members: tuple
    def __post_init__(self):
        if (type(self.members) is not tuple or any(type(p) is not tuple or len(p)!=2
            or type(p[0]) is not str or not _native_value(p[0]) or not _native_value(p[1]) for p in self.members)):
            raise ValueError("immutable native JSON member pairs/types required")
    def values(self, key):
        if type(key) is not str:
            raise ValueError("literal lookup member name required")
        return tuple(v for k,v in self.members if k == key)
    def unique(self, key):
        values = self.values(key)
        if not values:
            raise ValueError("native member not encoded; no inferred null/default")
        if len(values)!=1:
            raise ValueError("ambiguous duplicate native member; no guessed last-value semantic")
        return values[0]
    def status(self, key):
        values = self.values(key)
        if not values: return "not_encoded"
        if len(values)!=1: return "ambiguous_duplicate_member"
        if values[0] is None: return "native_null"
        if type(values[0]) is str and values[0]=="": return "native_empty_string"
        return "native_present"

def parse_native_entry(raw):
    if type(raw) is not bytes or not raw or len(raw)>MAX_ENTRY_BYTES:
        raise ValueError("bounded complete immutable native entry bytes required")
    def pairs(items):
        return NativeJSONObject(tuple((k,_freeze(v)) for k,v in items))
    def invalid(value):
        raise ValueError("nonfinite native JSON literal")
    def _freeze(value):
        if type(value) is list:
            return tuple(_freeze(v) for v in value)
        return value
    try:
        value = _freeze(json.loads(raw, object_pairs_hook=pairs, parse_int=NativeJSONNumber,
                                  parse_float=NativeJSONNumber, parse_constant=invalid))
    except (json.JSONDecodeError,UnicodeError,RecursionError) as error:
        raise ValueError("invalid native JSON object") from error
    if type(value) is not NativeJSONObject or not _native_value(value):
        raise ValueError("native lookup entry must be an object, not template/molecule conversion")
    return value

def _duplicates(value,path="$"):
    out=[]
    if type(value) is NativeJSONObject:
        seen=set()
        for key,item in value.members:
            if key in seen:out.append(path+":"+key)
            seen.add(key);out.extend(_duplicates(item,path+"."+key))
    elif type(value) is tuple:
        for i,item in enumerate(value):out.extend(_duplicates(item,path+"["+str(i)+"]"))
    return tuple(out)

def _source_canonical(raw):
    """Exact producer projection only: original bytes/order are stored separately."""
    def invalid(value):
        raise ValueError("nonfinite producer projection")
    try:
        return _canonical(json.loads(raw,parse_constant=invalid))
    except (json.JSONDecodeError,UnicodeError,OverflowError,ValueError,RecursionError) as error:
        raise ValueError("unsupported native-to-canonical producer projection") from error

@dataclass(frozen=True)
class PaLookupSelector:
    source_row: int
    source_file_sha256: str
    typed_file_sha256: str
    def __post_init__(self):
        if (not _int(self.source_row) or self.source_row>=42554 or not _hex(self.source_file_sha256)
                or not _hex(self.typed_file_sha256)):
            raise ValueError("exact lookup occurrence and source/typed SHA bindings required")

@dataclass(frozen=True)
class ComputedRuleEvidence:
    syntax: str
    native_text: str
    role: str = "source_computed_rule_not_molecule_or_measured_reaction"

@dataclass(frozen=True)
class PaLookupCandidate:
    selector: PaLookupSelector
    native_entry_bytes: bytes = field(repr=False)
    typed_record_bytes: bytes = field(repr=False)
    binding_bytes: bytes = field(repr=False)
    def __post_init__(self):
        if type(self.selector) is not PaLookupSelector:
            raise ValueError("typed lookup selector required")
        native=parse_native_entry(self.native_entry_bytes)
        row=_json_bytes(self.typed_record_bytes);binding=_json_bytes(self.binding_bytes)
        if (type(row) is not dict or set(row)!=set(FIELDS) or type(binding) is not dict
            or binding.get("schema")!="mora3.PaRoutesNativeLookupBinding/v1"
            or binding.get("handoff_sha256")!=HANDOFF_SHA256 or binding.get("manifest_sha256")!=MANIFEST_SHA256
            or binding.get("model_use_allowed") is not False or binding.get("task_admitted") is not False
            or row["source_file_sha256"]!=self.selector.source_file_sha256 or not _int(row["source_row"])
            or row["source_row"]!=self.selector.source_row or row["source_role"]!=COMPUTED
            or row["source_reference_hash_exposure"] is not None
            or row["task_admitted"] is not False or row["adapter_supported"] is not False
            or type(row["payload_json"]) is not str or _source_canonical(self.native_entry_bytes)!=row["payload_json"].encode()
            or type(row["quality_reasons_json"]) is not str):
            raise ValueError("exact native/canonical/source/type/role/mask binding mismatch")
        reasons=_json_bytes(row["quality_reasons_json"].encode())
        if type(reasons) is not list or any(type(r) is not str or not r for r in reasons):
            raise ValueError("exact literal source quality reasons required")
        # Parsing succeeds without making a native rule into a molecular identity.
        if type(native) is not NativeJSONObject:
            raise ValueError("native lookup object required")
    def native_value(self):
        return parse_native_entry(self.native_entry_bytes)
    def native_record(self):
        return _json_bytes(self.typed_record_bytes)
    def binding(self):
        return _json_bytes(self.binding_bytes)
    def field_status(self,key):
        return self.native_value().status(key)
    def field_role(self,key):
        if type(key) is not str:raise ValueError("literal native member required")
        return FIELD_ROLES.get(key,"native_unqualified_field") if self.field_status(key)!="not_encoded" else "not_encoded"
    @property
    def duplicate_member_paths(self):
        return _duplicates(self.native_value())
    @property
    def source_reference_hash_exposure(self):
        return self.native_record()["source_reference_hash_exposure"]
    @property
    def quality_reasons(self):
        return tuple(_strict_json(self.native_record()["quality_reasons_json"]))
    @property
    def status(self):
        return "native_computed_quarantined_unadmitted" if self.quality_reasons or self.duplicate_member_paths else "native_computed_candidate_unadmitted"
    @property
    def neural_token_count(self):
        return None
    @property
    def binding_sha256(self):
        return _sha(_canonical(dict(selector=self.selector.__dict__,native_hex=self.native_entry_bytes.hex(),
                    typed_hex=self.typed_record_bytes.hex(),binding_hex=self.binding_bytes.hex())))
    def rule_evidence(self):
        if self.quality_reasons or self.duplicate_member_paths:
            raise ValueError("quarantined or duplicate-key native lookup has unresolved context semantics")
        text=self.native_value().unique("reaction_smarts")
        if type(text) is not str or not text:
            raise ValueError("literal native computed reaction_smarts member unavailable")
        return ComputedRuleEvidence("SMARTS",text)
    def require_model_use(self,purpose,*,admission=None):
        if type(purpose) is not str or not purpose:
            raise ValueError("explicit model-use purpose required")
        raise ValueError("computed lookup has no constructor/master/model or heldout admission; reference/exposure/joins remain unsupported")

class PaRoutesLookupSource:
    """Full release bound; original-order access is explicitly bounded-prefix only."""
    def __init__(self,*,root,handoff=HANDOFF_PATH,expected_manifest_sha256=MANIFEST_SHA256):
        self.root=Path(root).resolve()
        if handoff!=HANDOFF_PATH or expected_manifest_sha256!=MANIFEST_SHA256:
            raise ValueError("exact finalized source release required")
        self._snapshot()
    def _path(self,path,suffixes):
        if type(path) is not str or not path:
            raise ValueError("exact source path required")
        p=(self.root/path).resolve()
        if not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes:
            raise ValueError("unsafe source path/format; no pickle/generated code")
        return p
    def _json(self,path,sha,size=None):
        p=self._path(path,{".json"})
        with p.open("rb") as handle:raw=handle.read(8*1024**2+1)
        if not _hex(sha) or _sha(raw)!=sha or len(raw)>8*1024**2 or size is not None and len(raw)!=size:
            raise ValueError("exact pinned proof integrity mismatch")
        return _strict_json(raw),dict(path=str(p),sha256=sha,bytes=len(raw))
    def _base_snapshot(self):
        h,hr=self._json(HANDOFF_PATH,HANDOFF_SHA256,13107)
        m,mr=self._json(MANIFEST_PATH,MANIFEST_SHA256,12293)
        plan,pr=self._json(m["plan_path"],PLAN_SHA256)
        available,ar=self._json(AVAILABLE_PATH,AVAILABLE_SHA256)
        proof,fr=self._json(h["handoff_proof"]["path"],PROOF_SHA256,97338)
        if (h["complete_native_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256 or m["plan_sha256"]!=PLAN_SHA256
            or plan["code_and_input_bindings"].get(str((self.root/AVAILABLE_PATH).resolve()))!=AVAILABLE_SHA256
            or _canonical(h["source_worker_task_admitted"])!=b"0" or h["model_use_allowed"] is not False
            or h["adapter_supported"] is not False or _canonical(m["task_admitted_rows"])!=b"0"
            or m["adapter_supported"] is not False or _canonical(m["source_files"])!=b"15"
            or _canonical(m["source_rows_nonadditive"])!=b"6032740" or len(m["assets"])!=15 or len(plan["assets"])!=15
            or proof["source_manifests"]["PaRoutes"]["sha256"]!=MANIFEST_SHA256
            or proof["no_new_task_admission"] is not True or proof["no_missing_or_inprogress_payloads"] is not True):
            raise ValueError("exact whole15 source/protocol/QC/closed-admission proof required")
        originals={Path(a["path"]).name:a for a in plan["assets"]}
        assets={Path(a["typed_path"]).name.removesuffix(".typed.parquet"):a for a in m["assets"]}
        if len(originals)!=15 or len(assets)!=15 or set(originals)!=set(assets):
            raise ValueError("complete unique whole15 native allocation required")
        a=assets[SOURCE_NAME];o=originals[SOURCE_NAME]
        expected_asset=dict(source_rows=42554,quality_pass_rows=42554,quarantined_rows=0,typed_bytes=23222765,
                            typed_sha256=TYPED_SHA256,source_role=COMPUTED)
        if (any(_canonical(a.get(k))!=_canonical(v) for k,v in expected_asset.items())
            or _canonical(o["bytes"])!=b"86066275" or o["sha256"]!=SOURCE_SHA256):
            raise ValueError("exact native lookup source/typed/QC occurrence binding required")
        self._path(o["path"],{".json"});self._path(a["typed_path"],{".parquet"})
        return dict(asset=a,original=o),dict(handoff=hr,manifest=mr,plan=pr,available=ar,proof=fr)
    def _snapshot(self):
        entry,refs=self._base_snapshot();a=entry["asset"];o=entry["original"]
        identity,ir=self._json(IDENTITY_PATH,IDENTITY_SHA256,2666)
        fixture_handoff,hr=self._json(FIXTURE_HANDOFF_PATH,FIXTURE_HANDOFF_SHA256,1260)
        fixture,fr=self._json(FIXTURE_PATH,FIXTURE_SHA256,61357)
        if (identity["full_QC_manifest_sha256"]!=MANIFEST_SHA256
            or identity["full_QC_manifest_path"]!=str((self.root/MANIFEST_PATH).resolve())
            or identity["QC_receipt_path"]!=a["receipt_path"] or identity["QC_receipt_sha256"]!=a["receipt_sha256"]
            or identity["original_raw_hash_newly_computed"] is not False
            or _canonical(identity["source_native_entries_inherited"])!=b"42554"
            or _canonical(fixture_handoff["task_admitted"])!=b"0" or _canonical(fixture_handoff["model_used"])!=b"0"
            or fixture_handoff["identity_supplement"]!={"path":str((self.root/IDENTITY_PATH).resolve()),"sha256":IDENTITY_SHA256,"bytes":2666}
            or fixture_handoff["fixture"]!={"path":str((self.root/FIXTURE_PATH).resolve()),"sha256":FIXTURE_SHA256,"bytes":61357}
            or fixture_handoff["full_release_duplicate_or_order_coverage_claim"] is not False
            or fixture_handoff["raw_entry_byte_offsets_and_JSON_roundtrip_pass"] is not True
            or _canonical(fixture["task_admitted"])!=b"0" or _canonical(fixture["model_used"])!=b"0"
            or _canonical(fixture["original_fresh_prefix_bytes"])!=b"65536" or fixture["original_prefix_sha256"]!=PREFIX_SHA256
            or _canonical(fixture["actual_selected_entries"])!=b"8" or len(fixture["entries"])!=8):
            raise ValueError("exact bounded original-order fixture/identity/closed-role proof required")
        manifest,_=self._json(MANIFEST_PATH,MANIFEST_SHA256,12293)
        guards={}
        for g in identity["guards"]:
            kind=g["kind"]
            if kind not in ("original","typed") or kind in guards:
                raise ValueError("unique original/typed native guard allocation required")
            info=g["file_identity"];receipt_info=g["QC_receipt_file_identity"]
            expected=(o["path"],o["sha256"],o["bytes"]) if kind=="original" else (a["typed_path"],a["typed_sha256"],a["typed_bytes"])
            if (not _valid_identity(info) or not _valid_identity(receipt_info)
                or (g["path"],g["expected_whole_sha256"],info["size"])!=expected
                or g["current_mtime_ctime_predate_individual_QC_receipt"] is not True
                or max(info["mtime_ns"],info["ctime_ns"])>min(receipt_info["mtime_ns"],receipt_info["ctime_ns"],_utc_ns(manifest["completed_utc"]))
                or _canonical(_identity(self._path(a["receipt_path"],{".json"}).stat()))!=_canonical(receipt_info)
                or kind=="original" and g["fresh_whole_sha256"] is not None
                or kind=="typed" and g["fresh_whole_sha256"]!=TYPED_SHA256):
                raise ValueError("exact inherited identity/preQC timestamp/receipt guard required")
            guards[kind]=dict(g)
        if set(guards)!={"original","typed"}:
            raise ValueError("both original and typed native guards required")
        guards["original"]["bounded_prefix_anchor"]=dict(bytes=65536,sha256=PREFIX_SHA256)
        return entry,guards,{**refs,"identity_supplement":ir,"bounded_native_fixture_handoff":hr,"bounded_native_fixture":fr}
    def _guard(self,g):
        p=self._path(g["path"],{".json",".parquet"})
        if _canonical(_identity(p.stat()))!=_canonical(g["file_identity"]):
            raise ValueError("source identity changed; inherited whole proof cannot be reused")
        return p
    def _receipt(self,entry):
        a=entry["asset"];o=entry["original"]
        receipt,ref=self._json(a["receipt_path"],a["receipt_sha256"])
        expected=dict(source_path=o["path"],source_sha256=o["sha256"],source_bytes=o["bytes"],
             source_rows=42554,typed_path=a["typed_path"],typed_sha256=a["typed_sha256"],typed_bytes=a["typed_bytes"],
             source_role=COMPUTED,quality_pass_rows=42554,quarantined_rows=0,full_typed_readback_rows=42554,
             native_TSV_ordered_columns=None,source_native_format="JSON_ARRAY",task_admitted_rows=0,
             adapter_supported=False,source_reference_hash_exposure_rows=0)
        if any(_canonical(receipt.get(k))!=_canonical(v) for k,v in expected.items()):
            raise ValueError("exact computed lookup/native/source-QC receipt binding mismatch")
        return receipt,ref
    @contextmanager
    def _typed_file(self,g):
        p=self._guard(g)
        with p.open("rb") as handle:
            before=_identity(os.fstat(handle.fileno()));h=hashlib.sha256()
            if _canonical(before)!=_canonical(g["file_identity"]):
                raise ValueError("typed source handle identity mismatch")
            for block in iter(lambda:handle.read(8*1024**2),b""):h.update(block)
            if h.hexdigest()!=g["expected_whole_sha256"]:
                raise ValueError("fresh selected typed integrity mismatch")
            handle.seek(0);yield handle
            if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):
                raise ValueError("typed source changed during read")
        self._guard(g)
    def _rows(self,g,indices,total):
        import pyarrow as pa, pyarrow.parquet as pq
        with self._typed_file(g) as handle:
            p=pq.ParquetFile(handle)
            expected=pa.schema(list(zip(FIELDS,[pa.string(),pa.int64(),pa.string(),pa.string(),pa.string(),pa.bool_(),pa.bool_(),pa.bool_()])))
            if not p.schema_arrow.equals(expected,check_metadata=True) or p.metadata.num_rows!=total:
                raise ValueError("exact native typed lookup schema/axis required")
            wanted=set(indices);offset=0;out={}
            for group in range(p.num_row_groups):
                info=p.metadata.row_group(group);end=offset+info.num_rows
                selected=[i for i in wanted if offset<=i<end]
                if selected:
                    if info.total_byte_size>MAX_ROW_GROUP_BYTES:
                        raise ValueError("native rowgroup exceeds bounded runtime; no crop")
                    table=p.read_row_group(group,use_threads=False)
                    for i in selected:out[i]=table.slice(i-offset,1).to_pylist()[0]
                offset=end
            if set(out)!=wanted:raise ValueError("selected native occurrence unavailable")
            return out
    def _native_entries(self,g,indices,total):
        last=max(indices)
        if last>=MAX_PREFIX_ENTRIES:
            raise ValueError("original-order late lookup exceeds bounded prefix; separate source locator proof required")
        p=self._guard(g);out={}
        with p.open("rb") as handle:
            before=_identity(os.fstat(handle.fileno()))
            if _canonical(before)!=_canonical(g["file_identity"]):
                raise ValueError("original source handle identity mismatch")
            buffer=b"";position=0;consumed=0;requested=0;prefix_digest=hashlib.sha256()
            def next_byte():
                nonlocal buffer,position,consumed,requested
                if position==len(buffer):
                    remaining=MAX_PREFIX_BYTES-requested
                    if remaining<=0:raise ValueError("native prefix byte bound exceeded; no truncation")
                    buffer=handle.read(min(65536,remaining));position=0;requested+=len(buffer);prefix_digest.update(buffer)
                    if not buffer:raise ValueError("native JSON prefix truncated")
                    anchor=g.get("bounded_prefix_anchor")
                    if anchor is not None and requested==len(buffer):
                        if (not _int(anchor["bytes"]) or anchor["bytes"]!=len(buffer)
                            or not _hex(anchor["sha256"]) or _sha(buffer)!=anchor["sha256"]):
                            raise ValueError("fresh bounded prefix integrity mismatch")
                value=buffer[position];position+=1;consumed+=1
                return value
            def nonspace():
                value=next_byte()
                while value in b" \t\r\n":value=next_byte()
                return value
            if nonspace()!=ord("["):raise ValueError("native JSON_ARRAY required")
            start_token=nonspace()
            for index in range(last+1):
                if start_token!=ord("{"):raise ValueError("complete native lookup object required")
                start=consumed-1;raw=bytearray(b"{");depth=1;quoted=False;escape=False
                while depth:
                    value=next_byte();raw.append(value)
                    if len(raw)>MAX_ENTRY_BYTES:raise ValueError("native entry exceeds bounded extent")
                    if quoted:
                        if escape:escape=False
                        elif value==92:escape=True
                        elif value==34:quoted=False
                    elif value==34:quoted=True
                    elif value in (123,91):depth+=1
                    elif value in (125,93):depth-=1
                native=bytes(raw);parse_native_entry(native);stop=consumed
                delimiter=nonspace()
                if index==total-1:
                    if delimiter!=93:raise ValueError("native final array extent mismatch")
                    # Full final extent must be whitespace, within the same explicit byte budget.
                    if before["size"]>MAX_PREFIX_BYTES:
                        raise ValueError("final native array extent exceeds bounded prefix")
                    tail=buffer[position:]
                    if any(v not in b" \t\r\n" for v in tail):
                        raise ValueError("native array trailing data/extent mismatch")
                    consumed+=len(tail);position=len(buffer)
                    while requested<before["size"]:
                        extra=handle.read(min(65536,before["size"]-requested))
                        if not extra:raise ValueError("native final extent truncated")
                        requested+=len(extra);consumed+=len(extra);prefix_digest.update(extra)
                        if any(v not in b" \t\r\n" for v in extra):
                            raise ValueError("native array trailing data/extent mismatch")
                else:
                    if delimiter!=44:raise ValueError("native array occurrence/delimiter mismatch")
                    start_token=nonspace()
                    if start_token!=123:raise ValueError("native array missing next object/trailing comma")
                if index in indices:
                    out[index]=(native,dict(native_start=start,native_stop=stop,native_entry_sha256=_sha(native),
                        entries_parsed=index+1,logical_prefix_bytes_consumed=consumed,
                        read_request_bytes=requested,prefix_read_sha256=prefix_digest.hexdigest(),
                        full_raw_hash_performed=False,original_member_order_preserved=True))
            if _canonical(_identity(os.fstat(handle.fileno())))!=_canonical(before):
                raise ValueError("original source changed during bounded prefix read")
        self._guard(g)
        return out
    def read_entries(self,selectors):
        if (type(selectors) is not tuple or not 1<=len(selectors)<=8 or any(type(s) is not PaLookupSelector for s in selectors)
            or len({s.source_row for s in selectors})!=len(selectors)):
            raise ValueError("1..8 unique exact lookup occurrences from one typed shard required")
        entry,guards,refs=self._snapshot();a=entry["asset"];o=entry["original"]
        if any(s.source_file_sha256!=o["sha256"] or s.typed_file_sha256!=a["typed_sha256"] or s.source_row>=a["source_rows"] for s in selectors):
            raise ValueError("wrong native source/typed/row binding")
        receipt,rref=self._receipt(entry)
        original=self._native_entries(guards["original"],tuple(s.source_row for s in selectors),a["source_rows"])
        rows=self._rows(guards["typed"],tuple(s.source_row for s in selectors),a["source_rows"])
        self._guard(guards["original"])
        out=[]
        for s in selectors:
            native,accounting=original[s.source_row]
            binding=dict(schema="mora3.PaRoutesNativeLookupBinding/v1",handoff_sha256=HANDOFF_SHA256,manifest_sha256=MANIFEST_SHA256,
                references={**refs,"receipt":rref,"original":dict(path=o["path"],bytes=o["bytes"],sha256=o["sha256"]),
                "typed":dict(path=a["typed_path"],bytes=a["typed_bytes"],sha256=a["typed_sha256"])},
                raw_identity_guard=guards["original"],typed_identity_guard=guards["typed"],native_read=accounting,
                source_QC_digest=receipt["semantic_rows_digest"],model_use_allowed=False,task_admitted=False,
                original_role="exact selected JSON object bytes/order/types; computed rule/context only",
                typed_role="source QC canonical JSON projection; sorted keys and duplicate-last-value semantics are not original serialization",
                raw_authority="inherited completeQC wholeSHA under exact current identity/preQCtimestamp/protected history guard; bounded original prefix reread",
                unsupported_gates=["official_reference_exposure_join","context_applicability","template_molecule_conversion","model_admission"])
            out.append(PaLookupCandidate(s,native,_canonical(rows[s.source_row]),_canonical(binding)))
        return tuple(out)
    def verify(self,candidate):
        if type(candidate) is not PaLookupCandidate:raise ValueError("exact native lookup candidate required")
        checked,=self.read_entries((candidate.selector,))
        if checked.binding_sha256!=candidate.binding_sha256:
            raise ValueError("exact native source/order/typed/binding replay mismatch")
        return checked
    def project_product_inputs(self,candidate):
        self.verify(candidate)
        raise ValueError("computed SMARTS context is not a native product/molecule input; no gold class/precursor/map/outcome/audit projection")
    def join_context(self,candidate,key):
        self.verify(candidate)
        raise ValueError("unsupported lookup/reference/template/context join; no applicability or heldout proof")
