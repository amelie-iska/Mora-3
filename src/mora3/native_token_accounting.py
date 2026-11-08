"""Source-bound accounting of observed runtime positions, never corpus estimates.

Graph attention, pooled payload embeddings, decoder teacher prefixes, scored
targets and native continuous/image elements are separate units. This observer
does not tokenize unsupported objectives, run training or alter model interfaces.
Whole-corpus totals remain unknown. Caller-owned event IDs distinguish repeated
runtime exposures from unique versioned task views.
"""
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path

import torch

from mora3.model.payload import PayloadEncoder
from mora3.model.tokengt import TokenGT

UNITS = {
    "condition_graph_positions": "graph_attention_positions",
    "condition_graph_allocated_positions": "graph_attention_slots",
    "condition_graph_padding_positions": "graph_attention_padding_slots",
    "graph_node_positions": "graph_node_positions",
    "graph_edge_positions": "graph_edge_positions",
    "graph_special_positions": "graph_special_positions",
    "condition_payload_embedding_positions": "prepool_payload_embedding_positions",
    "ar_scored_positions": "autoregressive_target_positions",
    "ar_lexical_scored_positions": "autoregressive_lexical_target_positions",
    "ar_eos_scored_positions": "autoregressive_EOS_target_positions",
    "ar_bos_prefix_positions": "decoder_BOS_prefix_positions",
    "ar_prefix_valid_positions": "decoder_teacher_prefix_positions",
    "ar_prefix_padding_positions": "decoder_prefix_padding_slots",
    "ar_target_padding_positions": "decoder_target_padding_slots",
    "ar_allocated_positions": "decoder_target_slots",
    "image_patch_positions": "image_patch_positions",
    "image_mask_token_positions": "image_mask_token_positions",
}
AR_NAMES = tuple(k for k in UNITS if k.startswith("ar_"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return _sha(_canonical(dict(shape=list(value.shape), dtype=str(value.dtype))) +
        value.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True)
class Measure:
    value: int | None
    unit: str
    status: str
    reason: str

    def __post_init__(self):
        common = (self.status in ("measured", "inapplicable", "unsupported", "unmeasured")
            and isinstance(self.unit, str) and bool(self.unit)
            and isinstance(self.reason, str) and bool(self.reason))
        count = (type(self.value) is int and self.value >= 0) if self.status == "measured" else self.value is None
        if not common or not count:
            raise ValueError("typed nonnegative measured count or explicit null status required")


def _measured(name, value, reason):
    return Measure(int(value), UNITS[name], "measured", reason)


@dataclass(frozen=True)
class SourceView:
    source_family: str
    source_revision: str
    view_id: str
    source_payload_sha256: str
    manifest_sha256: str
    reader_sha256: str
    split: str
    primary_family: str
    modalities: tuple[str, ...]
    input_sha256: str
    target_sha256: str
    target_fields: tuple[str, ...] = ()

    def __post_init__(self):
        if (any(not isinstance(v, str) or not v for v in (self.source_family,
                self.source_revision, self.view_id, self.primary_family))
                or self.split not in ("train", "validation", "test")
                or any(not _hex(v) for v in (self.source_payload_sha256, self.manifest_sha256,
                    self.reader_sha256, self.input_sha256, self.target_sha256))
                or type(self.modalities) is not tuple or not self.modalities
                or len(set(self.modalities)) != len(self.modalities)
                or any(not isinstance(v, str) or not v for v in self.modalities)
                or type(self.target_fields) is not tuple or not self.target_fields
                or len(set(self.target_fields)) != len(self.target_fields)
                or any(not isinstance(v, str) or not v for v in self.target_fields)):
            raise ValueError("complete typed source/view/reader/admission binding required")

    @classmethod
    def bind(cls, *, conditions, targets, **metadata):
        if not isinstance(conditions, dict) or not isinstance(targets, dict) or not targets:
            raise ValueError("separate explicit condition and target mappings required")
        metadata["modalities"] = tuple(metadata["modalities"])
        metadata["target_fields"] = tuple(sorted(targets))
        return cls(**metadata, input_sha256=_sha(_canonical(conditions)),
            target_sha256=_sha(_canonical(targets)))

    def identity(self):
        # A view inherited by a newer master is still the same source task.
        return self.source_family, self.source_revision, self.view_id

    def stable_bytes(self):
        return _canonical({k: v for k, v in asdict(self).items()
            if k not in ("manifest_sha256", "reader_sha256")})


def _model_signature(model):
    if model is None:
        return None
    configurations = {}
    for name, module in model.named_modules():
        if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
            configurations[name] = module.get_extra_state()
    return dict(type=type(model).__module__ + "." + type(model).__qualname__,
        configurations=configurations, modes={n: m.training for n, m in model.named_modules()},
        parameters={n: dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device))
            for n, p in model.named_parameters()},
        runtime=dict(threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32))


def _tokenizer_signature(tokenizer):
    if hasattr(tokenizer, "vocabulary"):
        return dict(type=type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            payload_hex=[v.hex() for v in tokenizer.vocabulary],
            controls=dict(tokenizer._controls), size=tokenizer.size)
    if all(hasattr(tokenizer, k) for k in ("symbols", "pad", "bos", "eos", "size")):
        return dict(type=type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            symbols=tuple(tokenizer.symbols), pad=tokenizer.pad, bos=tokenizer.bos,
            eos=tokenizer.eos, size=tokenizer.size)
    raise ValueError("actual installed payload tokenizer or native target codec required")


@dataclass(frozen=True)
class RuntimeBinding:
    specification_bytes: bytes

    def __post_init__(self):
        d = self.data()
        if (type(self.specification_bytes) is not bytes
                or _canonical(d) != self.specification_bytes
                or d.get("schema") != "mora3.NativeRuntimePositionBinding/v1"
                or not d.get("objective_key") or not d.get("code_files")):
            raise ValueError("canonical exact runtime binding required")

    @property
    def sha256(self):
        return _sha(self.specification_bytes)

    def data(self):
        return json.loads(self.specification_bytes)

    @classmethod
    def create(cls, *, root, reader_functions, model, tokenizers, objective_key,
               context, packing, vocabulary_paths=()):
        root = Path(root).resolve()
        if (not reader_functions or not isinstance(objective_key, str) or not objective_key
                or context.get("objective_kind") not in ("autoregressive", "continuous", "graph_only", "unsupported")
                or type(context.get("graph_required")) is not bool
                or not isinstance(packing, dict) or not packing.get("policy")):
            raise ValueError("reader, objective, applicability, context and packing policies required")
        if context["objective_kind"] == "unsupported" and (model is not None or tokenizers):
            raise ValueError("unsupported objectives must not be tokenized or run")
        paths = {Path(__file__).resolve()}
        reader_paths = set()
        for reader in reader_functions:
            path = Path(inspect.getfile(reader)).resolve()
            paths.add(path)
            reader_paths.add(path)
        if model is not None:
            for module in model.modules():
                if type(module).__module__.startswith("mora3."):
                    paths.add(Path(inspect.getfile(type(module))).resolve())
        for tokenizer in tokenizers.values():
            if type(tokenizer).__module__.startswith("mora3."):
                paths.add(Path(inspect.getfile(type(tokenizer))).resolve())
        code = []
        for path in sorted(paths):
            if not path.is_relative_to(root) or path.suffix != ".py":
                raise ValueError("project-owned code revisions required")
            code.append(dict(path=str(path.relative_to(root)), sha256=_sha(path.read_bytes())))
        specs = {k: _tokenizer_signature(t) for k, t in tokenizers.items()}
        vocabulary = []
        for name in vocabulary_paths:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or path.suffix != ".json":
                raise ValueError("project-owned vocabulary artifact required")
            raw = path.read_bytes()
            d = json.loads(raw)
            controls = {v: 256 + len(d["learned_hex"]) + i for i, v in enumerate(d["controls"])}
            if not any(s.get("payload_hex", [])[256:] == d["learned_hex"]
                    and s.get("controls") == controls for s in specs.values()):
                raise ValueError("vocabulary artifact does not match actual ordered tokenizer IDs")
            vocabulary.append(dict(path=str(path.relative_to(root)), sha256=_sha(raw)))
        spec = dict(schema="mora3.NativeRuntimePositionBinding/v1", root=str(root),
            objective_key=objective_key, code_files=code,
            reader_files=[r for r in code if (root/r["path"]).resolve() in reader_paths],
            reader_callables=[f.__module__+"."+f.__qualname__ for f in reader_functions],
            vocabulary_files=vocabulary,
            tokenizers=specs, model=_model_signature(model), context=context, packing=packing,
            scope="position/configuration binding;not learned-weight or model-quality evidence")
        return cls(_canonical(spec))

    def verify(self, *, model, tokenizers):
        d = self.data()
        for ref in d["code_files"] + d["vocabulary_files"]:
            path = (Path(d["root"]) / ref["path"]).resolve()
            if not path.is_relative_to(Path(d["root"])) or _sha(path.read_bytes()) != ref["sha256"]:
                raise ValueError("runtime code/vocabulary revision mismatch")
        if (_canonical(_model_signature(model)) != _canonical(d["model"])
                or _canonical({k: _tokenizer_signature(t) for k, t in tokenizers.items()}) != _canonical(d["tokenizers"])):
            raise ValueError("live model/tokenizer/context configuration mismatch")


def measure_autoregressive(prefix, target, codec, *, scored_mask=None):
    """Count actual teacher IDs/masks, retaining prefix versus condition roles."""
    if (not isinstance(prefix, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prefix.dtype != torch.long or target.dtype != torch.long
            or prefix.ndim != 2 or prefix.shape != target.shape or min(prefix.shape) < 1
            or bool(((prefix < 0) | (prefix >= codec.size)).any())
            or bool(((target < 0) | (target >= codec.size)).any())):
        raise ValueError("aligned valid actual teacher-prefix/target IDs required")
    available = target.ne(codec.pad)
    mask = available if scored_mask is None else scored_mask
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != target.shape or mask.device != target.device or bool((mask & ~available).any())):
        raise ValueError("scoring mask must select available nonpadding target positions")
    out = []
    for p, t, valid, scored in zip(prefix, target, available, mask):
        n = int(valid.sum())
        if (n < 1 or not bool(valid[:n].all()) or bool(valid[n:].any())
                or int(t[n-1]) != codec.eos or bool((t[:n-1] < 3).any())
                or int(p[0]) != codec.bos or not torch.equal(p[1:n], t[:n-1])
                or bool(p[n:].ne(codec.pad).any())):
            raise ValueError("complete native BOS/prefix and lexical+EOS/right-padding convention required")
        counts = dict(ar_scored_positions=int(scored.sum()),
            ar_lexical_scored_positions=int((scored & t.ge(3)).sum()),
            ar_eos_scored_positions=int((scored & t.eq(codec.eos)).sum()),
            ar_bos_prefix_positions=int(p.eq(codec.bos).sum()),
            ar_prefix_valid_positions=int(p.ne(codec.pad).sum()),
            ar_prefix_padding_positions=int(p.eq(codec.pad).sum()),
            ar_target_padding_positions=int(t.eq(codec.pad).sum()),
            ar_allocated_positions=t.numel())
        out.append({k: _measured(k, v, "actual teacher IDs and scored-target mask") for k, v in counts.items()})
    return tuple(out)


def native_tensor_measure(values, *, unit, observed=None):
    """Native element denominator only; never assign a neural token unit."""
    if (not isinstance(values, torch.Tensor) or unit in set(UNITS.values())
            or not isinstance(unit, str) or not unit):
        raise ValueError("actual native tensor and distinct native-element unit required")
    mask = torch.ones_like(values, dtype=torch.bool) if observed is None else observed
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != values.shape or mask.device != values.device
            or not torch.isfinite(values[mask]).all()):
        raise ValueError("aligned finite observed native elements and boolean availability required")
    return Measure(int(mask.sum()), unit, "measured", "actual native tensor availability;not neural tokens")


def condition_descriptor(value):
    """Accounting-only descriptor of actual condition arguments; never features.

    Native task IDs, target masks and provenance are excluded. Pixel digests bind
    actual input bytes without passing audit hashes to the neural model.
    """
    from mora3.data.raster_morphology import RasterMorphologyTask
    from mora3.data.sciplex_admission import ObservationalSciPlexTask
    if isinstance(value, RasterMorphologyTask):
        if value.direction != "image_to_mask":
            return None
        result = dict(image_pixels_sha256=_sha(bytes.fromhex(value.image_hex)),
            image_dtype=value.image_dtype, shape=[value.height, value.width],
            direction=value.direction)
    elif isinstance(value, ObservationalSciPlexTask):
        result = json.loads(value.task.sidecar_bytes)["conditions"]
    elif isinstance(value, dict):
        result = value
    elif callable(getattr(value, "inputs", None)):
        result = value.inputs()
    else:
        return None
    try:
        _canonical(result)
    except (TypeError, ValueError):
        return None
    return result


def _target_descriptor(value):
    from mora3.data.raster_morphology import RasterMorphologyTask
    from mora3.data.sciplex_admission import ObservationalSciPlexTask
    if isinstance(value, RasterMorphologyTask) and value.direction == "image_to_mask":
        return dict(labels_sha256=_sha(bytes.fromhex(value.labels_hex)))
    if isinstance(value, ObservationalSciPlexTask):
        return json.loads(value.task.sidecar_bytes)["targets"]
    if callable(getattr(value, "targets", None)):
        return value.targets()
    return None


@dataclass(frozen=True)
class RuntimeRecord:
    source: SourceView
    objective_key: str
    runtime_binding_sha256: str
    measures_bytes: bytes
    native_units_bytes: bytes
    input_trace_sha256: str
    runtime_supported: bool
    condition_descriptor_verified: bool = False
    target_descriptor_verified: bool = False

    @property
    def measures(self):
        return {k: Measure(**v) for k, v in json.loads(self.measures_bytes).items()}

    @property
    def native_units(self):
        return {k: Measure(**v) for k, v in json.loads(self.native_units_bytes).items()}

    def data(self):
        return dict(source=asdict(self.source), objective_key=self.objective_key,
            runtime_binding_sha256=self.runtime_binding_sha256,
            measures={k: asdict(v) for k, v in self.measures.items()},
            native_units={k: asdict(v) for k, v in self.native_units.items()},
            input_trace_sha256=self.input_trace_sha256, runtime_supported=self.runtime_supported,
            condition_descriptor_verified=self.condition_descriptor_verified,
            target_descriptor_verified=self.target_descriptor_verified)


def _record(source, binding, measures, native_units, trace_sha, supported,
            condition_verified=False, target_verified=False):
    if (set(measures) != set(UNITS) or any(not isinstance(v, Measure) for v in measures.values())
            or set(native_units) & set(UNITS) or any(not isinstance(v, Measure) for v in native_units.values())):
        raise ValueError("complete typed runtime measures and separate native units required")
    return RuntimeRecord(source, binding.data()["objective_key"], binding.sha256,
        _canonical({k: asdict(v) for k, v in measures.items()}),
        _canonical({k: asdict(v) for k, v in native_units.items()}), trace_sha, supported, condition_verified, target_verified)


def unsupported_record(source, binding, *, reason, native_units=None):
    if binding.data()["context"]["objective_kind"] != "unsupported":
        raise ValueError("explicit unsupported model objective binding required")
    binding.verify(model=None, tokenizers={})
    if source.reader_sha256 not in {r["sha256"] for r in binding.data()["reader_files"]}:
        raise ValueError("source inspection reader differs from bound native reader")
    measures = {k: Measure(None, u, "unsupported", reason) for k, u in UNITS.items()}
    return _record(source, binding, measures, native_units or {}, _sha(b"no-model-runtime"), False)


class RuntimeCapture:
    """Temporary read-only observer hooks on caller-owned diagnostic instances.

    One graph batch row maps to one supplied source view in each observed call.
    Different-sized calls fail closed rather than guessing source allocation.
    Payload embedding streams are observed before pooling, never recounted from
    serialized source strings. Hooks and configuration checks leave no defaults.
    """
    def __init__(self, binding, *, model, tokenizers, expected_binding_sha256=None):
        if expected_binding_sha256 is not None and binding.sha256 != expected_binding_sha256:
            raise ValueError("expected runtime/context/packing binding mismatch")
        binding.verify(model=model, tokenizers=tokenizers)
        if binding.data()["context"]["objective_kind"] == "unsupported" or model is None:
            raise ValueError("unsupported objective cannot have runtime positions")
        self.binding, self.model, self.tokenizers = binding, model, dict(tokenizers)
        self.graphs, self.pending, self.prefix_calls, self.handles = [], [], [], []
        self.condition_calls, self.target_calls, self.source_payload_calls = [], [], []
        self.ar, self.ar_codec, self.closed, self.used, self.failed = None, None, False, False, False

    def __enter__(self):
        if self.used:
            raise ValueError("runtime capture is single-use")
        self.used = True
        seen = set()
        for module in self.model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if type(module).__module__.startswith("mora3."):
                self.handles.append(module.register_forward_pre_hook(self._conditions))
            if isinstance(module, PayloadEncoder):
                self.handles.append(module.embedding.register_forward_pre_hook(self._payload))
            if isinstance(module, TokenGT):
                self.handles.append(module.register_forward_hook(self._graph))
        if hasattr(self.model, "codec") and isinstance(getattr(self.model, "embedding", None), torch.nn.Embedding):
            self.handles.append(self.model.embedding.register_forward_pre_hook(self._prefix))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True
        self.failed = exc_type is not None
        if exc_type is None:
            try:
                self.binding.verify(model=self.model, tokenizers=self.tokenizers)
            except Exception:
                self.failed = True
                raise
        return False

    def _conditions(self, module, args):
        if not args:
            return
        values = args[0] if isinstance(args[0], (tuple, list)) else (args[0],)
        if not values:
            return
        descriptors = [condition_descriptor(v) for v in values]
        if all(d is not None for d in descriptors):
            self.condition_calls.append(tuple(_sha(_canonical(d)) for d in descriptors))
            targets = [_target_descriptor(v) for v in values]
            if all(t is not None for t in targets):
                self.target_calls.append(tuple(_sha(_canonical(t)) for t in targets))
            if all(callable(getattr(v, "canonical_bytes", None)) for v in values):
                self.source_payload_calls.append(tuple(_sha(v.canonical_bytes()) for v in values))

    def _payload(self, module, args):
        ids = args[0]
        if ids.dtype != torch.long or ids.ndim != 1:
            raise ValueError("actual unpadded entity payload embedding IDs required")
        self.pending.append(ids.detach().cpu().clone())

    def _prefix(self, module, args):
        self.prefix_calls.append(args[0].detach().cpu().clone())

    def _graph(self, module, args, output):
        graphs, valid = args[0], output[1]
        if valid.dtype != torch.bool or valid.ndim != 2 or len(graphs) != valid.shape[0]:
            raise ValueError("actual TokenGT validity matrix required")
        total = sum(g["roles"].numel() for g in graphs)
        streams = list(self.pending)
        self.pending.clear()
        if streams and len(streams) != total:
            raise ValueError("ambiguous graph/entity payload stream allocation")
        rows, cursor = [], 0
        for i, graph in enumerate(graphs):
            roles = graph["roles"].detach().cpu()
            v = valid[i].detach().cpu()
            if roles.dtype != torch.long or bool(((roles != 0) & (roles != 1)).any()) or int(v.sum()) != roles.numel()+1:
                raise ValueError("actual node/edge/graph-special allocation mismatch")
            n = roles.numel()
            payload = streams[cursor:cursor+n] if streams else None
            cursor += n
            trace = dict(graph={k: _tensor_sha(graph[k]) for k in
                ("features", "endpoints", "roles", "identifiers")},
                payload=None if payload is None else [_tensor_sha(x) for x in payload])
            rows.append(dict(valid=int(v.sum()), allocated=v.numel(), padding=int((~v).sum()),
                nodes=int(roles.eq(0).sum()), edges=int(roles.eq(1).sum()), special=1,
                payload=None if payload is None else sum(x.numel() for x in payload), trace=trace))
        self.graphs.append(rows)

    def autoregressive(self, prefix, target, codec, *, scored_mask=None):
        if self.closed or self.ar is not None:
            raise ValueError("one actual teacher batch per runtime capture required")
        policy = self.binding.data()["packing"]
        if _canonical(_tokenizer_signature(codec)) != _canonical(self.binding.data()["tokenizers"].get("target")):
            raise ValueError("teacher codec differs from bound actual target codec")
        self.ar_codec = codec
        if (policy.get("policy") != "native-right-padded-teacher-forcing-v1"
                or policy.get("scoring") != "target-not-PAD"
                or any(type(policy.get(k)) is not int or policy[k] != getattr(codec, k) for k in ("pad", "bos", "eos"))
                or (scored_mask is not None and not torch.equal(scored_mask, target.ne(codec.pad)))):
            raise ValueError("actual native decoder objective/packing policy mismatch")
        self.ar = (prefix.detach().cpu().clone(), target.detach().cpu().clone(),
            measure_autoregressive(prefix, target, codec, scored_mask=scored_mask))

    def records(self, sources, *, native_units=None):
        sources = tuple(sources)
        if not self.closed or self.failed or not sources or any(not isinstance(s, SourceView) for s in sources):
            raise ValueError("closed observed runtime and source-bound rows required")
        if self.pending or any(len(batch) != len(sources) for batch in self.graphs):
            raise ValueError("runtime/source row allocation mismatch")
        context = self.binding.data()["context"]
        reader_hashes = {r["sha256"] for r in self.binding.data()["reader_files"]}
        if any(source.reader_sha256 not in reader_hashes for source in sources):
            raise ValueError("source reader revision differs from bound runtime reader")
        if self.condition_calls and any(
                len(call) != len(sources) or any(h != source.input_sha256 for h, source in zip(call, sources))
                for call in self.condition_calls):
            raise ValueError("actual forward conditions differ from source view input binding")
        for calls, field in ((self.target_calls, "target_sha256"),
                             (self.source_payload_calls, "source_payload_sha256")):
            if calls and any(len(call) != len(sources) or
                    any(h != getattr(source, field) for h, source in zip(call, sources)) for call in calls):
                raise ValueError("actual source target/payload differs from supplied view binding")
        native_target_verified = False
        if self.ar is not None and type(self.ar_codec).__name__ == "NucleotideCodec":
            for source, row in zip(sources, self.ar[1]):
                target = [int(x) for x in row[row.ne(self.ar_codec.pad)].tolist()]
                decoded = self.ar_codec.decode([self.ar_codec.bos] + target)
                if len(source.target_fields) != 1 or _sha(_canonical({source.target_fields[0]: decoded})) != source.target_sha256:
                    raise ValueError("native teacher target differs from source target descriptor")
            native_target_verified = True
        if context["graph_required"] and not self.graphs:
            raise ValueError("no actual model graph forward observed")
        if context["objective_kind"] == "autoregressive":
            if (self.ar is None or len(self.ar[2]) != len(sources)
                    or not any(torch.equal(x, self.ar[0]) for x in self.prefix_calls)):
                raise ValueError("no matching actual decoder teacher-prefix forward observed")
        elif self.ar is not None:
            raise ValueError("nonautoregressive objective cannot acquire neural target tokens")
        native_units = tuple({} for _ in sources) if native_units is None else tuple(native_units)
        if len(native_units) != len(sources):
            raise ValueError("one native-element denominator map per source row required")
        out = []
        for i, source in enumerate(sources):
            measures = {k: Measure(None, unit, "inapplicable", "no token representation on this runtime path")
                for k, unit in UNITS.items()}
            image_representation = context.get("image_representation")
            if image_representation not in (None, "native-CNN-single-image-graph-v1"):
                for name in ("image_patch_positions", "image_mask_token_positions"):
                    measures[name] = Measure(None, UNITS[name], "unmeasured",
                        "image representation has no registered position observer")
            rows = [batch[i] for batch in self.graphs]
            if rows:
                fields = dict(condition_graph_positions="valid", condition_graph_allocated_positions="allocated",
                    condition_graph_padding_positions="padding", graph_node_positions="nodes",
                    graph_edge_positions="edges", graph_special_positions="special")
                for name, field in fields.items():
                    measures[name] = _measured(name, sum(x[field] for x in rows), "actual TokenGT roles/validity/packing")
                if all(x["payload"] is not None for x in rows):
                    measures["condition_payload_embedding_positions"] = _measured(
                        "condition_payload_embedding_positions", sum(x["payload"] for x in rows),
                        "actual prepool entity embedding input IDs;not graph attention positions")
            if self.ar is not None:
                measures.update(self.ar[2][i])
            trace = _sha(_canonical([x["trace"] for x in rows]))
            out.append(_record(source, self.binding, measures, native_units[i], trace, True,
                condition_verified=bool(self.condition_calls),
                target_verified=bool(self.target_calls) or native_target_verified))
        return tuple(out)


class AccountingLedger:
    """Bounded runtime-event ledger; unique views are not sampled exposure totals."""
    def __init__(self, allowed_bindings, *, max_events=4096):
        if (not isinstance(allowed_bindings, dict) or not allowed_bindings
                or any(not isinstance(k, str) or not k or not _hex(v) for k, v in allowed_bindings.items())
                or type(max_events) is not int or not 1 <= max_events <= 4096):
            raise ValueError("explicit objective/runtime bindings and bounded event budget required")
        self.allowed = dict(allowed_bindings)
        self.max_events, self.events, self.views = max_events, {}, {}
        self.native_schema = {}

    def add(self, exposure_id, record, *, kind):
        if (not isinstance(exposure_id, str) or not exposure_id or exposure_id in self.events
                or len(self.events) >= self.max_events or not isinstance(record, RuntimeRecord)
                or kind not in ("diagnostic_scoring", "optimizer_training", "source_inspection")
                or self.allowed.get(record.objective_key) != record.runtime_binding_sha256
                or (not record.runtime_supported and kind != "source_inspection")):
            raise ValueError("unique bounded event and matching actual runtime/source role required")
        key = record.source.identity()
        stable = record.source.stable_bytes()
        if key in self.views and self.views[key].stable_bytes() != stable:
            raise ValueError("same versioned source view changed identity/conditions/targets/partition")
        # Validate all measures before committing any ledger state.
        record.measures
        native = record.native_units
        if any(name in self.native_schema and self.native_schema[name] != q.unit for name, q in native.items()):
            raise ValueError("native element roles cannot merge incompatible units")
        self.native_schema.update({name: q.unit for name, q in native.items()})
        self.views.setdefault(key, record.source)
        self.events[exposure_id] = (record, kind)

    def snapshot(self):
        def presence(primary):
            result = {}
            for source in self.views.values():
                keys = (source.primary_family,) if primary else source.modalities
                for key in keys:
                    result.setdefault(key, dict(unique_views=0, sampled_model_exposures=0))
                    result[key]["unique_views"] += 1
            for record, kind in self.events.values():
                if kind == "source_inspection":
                    continue
                keys = (record.source.primary_family,) if primary else record.source.modalities
                for key in keys:
                    result[key]["sampled_model_exposures"] += 1
            return result
        metrics = {}
        for name, unit in UNITS.items():
            known, nulls, inapplicable = [], 0, 0
            for record, kind in self.events.values():
                q = record.measures[name]
                if kind != "source_inspection" and q.status == "measured":
                    known.append(q.value)
                elif q.status == "inapplicable":
                    inapplicable += 1
                else:
                    nulls += 1
            metrics[name] = dict(unit=unit, measured_sum=sum(known) if known else None,
                measured_runtime_events=len(known), inapplicable_events=inapplicable,
                unsupported_or_unmeasured_events=nulls, whole_corpus_total=None)
        native_metrics = {}
        for name, unit in self.native_schema.items():
            runtime, inspection, registered = [], [], 0
            for record, kind in self.events.values():
                q = record.native_units.get(name)
                if q is None:
                    continue
                registered += 1
                if q.status == "measured":
                    (inspection if kind == "source_inspection" else runtime).append(q.value)
            native_metrics[name] = dict(unit=unit,
                sampled_runtime_measured_sum=sum(runtime) if runtime else None,
                source_inspection_measured_sum=sum(inspection) if inspection else None,
                registered_events=registered, not_registered_events=len(self.events)-registered,
                whole_corpus_total=None)
        return dict(schema="mora3.NativeRuntimeAccountingLedger/v1",
            unique_registered_views=len(self.views), registered_events=len(self.events),
            sampled_model_exposures=sum(k != "source_inspection" for _, k in self.events.values()),
            diagnostic_scoring_exposures=sum(k == "diagnostic_scoring" for _, k in self.events.values()),
            optimizer_training_exposures=sum(k == "optimizer_training" for _, k in self.events.values()),
            source_inspections=sum(k == "source_inspection" for _, k in self.events.values()),
            primary_partitions=presence(True), modality_presence=presence(False), position_metrics=metrics,
            native_unit_metrics=native_metrics,
            whole_corpus_model_positions=None, token_metric_scope="observed runtime exposure events only",
            unique_views_are_not_token_totals=True, modality_presence_overlaps_and_is_not_additive=True)
