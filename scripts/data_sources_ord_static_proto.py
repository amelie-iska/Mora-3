"""Read pinned ORD descriptor literals without importing or executing generated source.

Source preparation only. Original protobuf bytes remain authoritative; JSON excludes
unknown wire fields and is quarantined whenever unknowns exist. Proto3 scalar fields
without presence cannot distinguish omitted/default wire values; never fill defaults.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory
from google.protobuf.message import DecodeError

REACTION_DESCRIPTOR_NAME = "ord-schema/proto/reaction.proto"

def extract_descriptor(path, expected_sha256, expected_name, expected_dependencies=()):
    raw = Path(path).read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("descriptor source size exceeds bounded static parser")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("descriptor source SHA256 mismatch")
    tree = ast.parse(raw.decode("utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "AddSerializedFile"]
    if len(calls) != 1:
        raise ValueError("expected exactly one AddSerializedFile call")
    call = calls[0]
    if (len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, bytes)):
        raise ValueError("descriptor must be a literal bytes constant")
    blob = call.args[0].value
    fd = descriptor_pb2.FileDescriptorProto()
    try:
        fd.ParseFromString(blob)
    except DecodeError as exc:
        raise ValueError("descriptor parse failure") from exc
    if (fd.name, fd.package, fd.syntax) != (expected_name, "ord", "proto3"):
        raise ValueError("descriptor identity mismatch")
    if tuple(fd.dependency) != tuple(expected_dependencies):
        raise ValueError("descriptor dependencies mismatch")
    return blob

def load_reaction_class(path, expected_sha256):
    blob = extract_descriptor(path, expected_sha256, REACTION_DESCRIPTOR_NAME, ())
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(blob)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("ord.Reaction"))

def decode_reaction(reaction_class, native_bytes, expected_id, max_bytes=64 * 1024 * 1024):
    if not isinstance(native_bytes, bytes) or not 0 <= len(native_bytes) <= max_bytes:
        raise ValueError("native protobuf size/type exceeds bounded decoder")
    msg = reaction_class()
    try:
        msg.ParseFromString(native_bytes)
    except DecodeError as exc:
        raise ValueError("native protobuf parse/decode failure") from exc
    canonical = msg.SerializeToString(deterministic=True)
    clean = reaction_class()
    clean.CopyFrom(msg)
    clean.DiscardUnknownFields()
    clean_bytes = clean.SerializeToString(deterministic=True)
    unknown = canonical != clean_bytes
    semantic = json_format.MessageToDict(
        clean, preserving_proto_field_name=True, use_integers_for_enums=False)
    # JSON exposes enums, native units and oneof presence; original bytes retain
    # wire order/default encodings and any fields not covered by this descriptor.
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False)
    back = reaction_class()
    json_format.Parse(encoded, back)
    roundtrip = back.SerializeToString(deterministic=True) == clean_bytes
    reasons = []
    if unknown:
        reasons.append("unknown_wire_fields")
    if not msg.reaction_id or not expected_id:
        reasons.append("missing_reaction_id")
    if msg.reaction_id != expected_id:
        reasons.append("column_message_reaction_id_mismatch")
    if not roundtrip:
        reasons.append("semantic_json_roundtrip_mismatch")
    return {
        "original_proto_bytes": len(native_bytes),
        "original_proto_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "native_reaction_id": msg.reaction_id,
        "unknown_wire_fields": unknown,
        "semantic_json_roundtrip_equal": roundtrip,
        "semantic": semantic,
        "quality_reasons": reasons,
        "task_admitted": False,
        "adapter_supported": False,
        "proto3_default_presence_limit": True,
    }
