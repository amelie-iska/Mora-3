"""Native-byte PubChem prior and exact standardSELFIES codec objective.

Uses the existing shared complete-condition entity/positional-payload reference
and conditional-initial-state GRU pattern. No atom/bond expansion, flow-method
replacement, original GroupSELFIES, chemical validity or training quality claim.
All byte values remain representable with a fixed existing vocabulary; no fit.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.tokenization.shortest_path import ByteTokenizer
from .annotation_multihead import SharedAnnotationModel
from .mixed_payload_projection import exact_wire
from .sequence_mup import (IndependentParameterGRU,recurrent_parameterization,initialize_recurrent_head,
    recurrent_readout,validate_recurrent_head,recurrent_checkpoint_preflight)

native_reader = importlib.import_module("mora3.data.pubchem_native")
READER_SHA256 = "cc66030a0b9d89b7c91d11e39adf388c548d83c46c68ac46def12662cd224a37"
PILOT_SHA256 = "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
CONDITION_KIND = "molecular-representation"
TARGETS = dict(reference_prior="native_smiles", smiles_to_selfies="standard_selfies",
               selfies_to_smiles="source_canonical_isomeric_smiles")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _exact_configuration(actual, expected):
    """Torch child extra-state validators distinguish tuple/list and bool/int."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_exact_configuration(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(_exact_configuration(a, b) for a, b in zip(actual, expected))
    return _canonical(actual) == _canonical(expected)


class PubChemByteCodec:
    """Existing shortest-path byte vocabulary, with distinct new-head controls.

    Lexical positions are byte/subword token IDs, not SELFIES symbol counts.
    Unknown native characters never need replacement: all 256 bytes are present.
    """
    def __init__(self, tokenizer):
        if (not isinstance(tokenizer, ByteTokenizer)
                or tokenizer.vocabulary[:256] != tuple(bytes([i]) for i in range(256))):
            raise ValueError("complete ordered existing byte vocabulary required")
        self.tokenizer = ByteTokenizer(tokenizer.vocabulary[256:], controls=("PAD", "BOS", "EOS"))
        self.pad = self.tokenizer.control_id("PAD")
        self.bos = self.tokenizer.control_id("BOS")
        self.eos = self.tokenizer.control_id("EOS")
        self.size = self.tokenizer.size

    def configuration(self):
        return dict(version=1, payload_hex=[x.hex() for x in self.tokenizer.vocabulary],
            controls=dict(PAD=self.pad, BOS=self.bos, EOS=self.eos), size=self.size,
            encoding="existing ByteTokenizer minimum-token UTF8/raw bytes; no normalization",
            unknown_policy="all256bytes_present;no_unknown_or_replacement_symbol")

    def encode(self, raw):
        if type(raw) is not bytes or not raw:
            raise ValueError("complete nonempty native bytes required")
        return [self.bos] + list(self.tokenizer.encode(raw)) + [self.eos]

    def decode(self, ids):
        ids = list(ids)
        if (not ids or any(type(i) is not int for i in ids) or ids[0] != self.bos or self.eos not in ids):
            raise ValueError("exact native BOS/EOS token sequence required")
        end = ids.index(self.eos)
        if any(i != self.pad for i in ids[end + 1:]):
            raise ValueError("only PAD may follow EOS")
        return self.tokenizer.decode(ids[1:end])

    def partial_bytes(self, ids):
        ids = list(ids)
        if not ids or any(type(i) is not int for i in ids) or ids[0] != self.bos:
            raise ValueError("exact BOS-prefixed partial sequence required")
        if self.eos in ids:
            return self.decode(ids)
        return self.tokenizer.decode(ids[1:])


def _direction(conditions):
    if type(conditions) is not dict:
        raise ValueError("explicit target-free PubChem conditions required")
    if set(conditions) == {"prior_domain"}:
        if conditions["prior_domain"] != "molecular_reference_native_SMILES":
            raise ValueError("unsupported native molecular prior domain")
        return "reference_prior"
    if (set(conditions) not in ({"representation_task", "native_smiles"},
                               {"representation_task", "standard_selfies"})
            or conditions["representation_task"] != "standard_SELFIES_codec"):
        raise ValueError("exact representation condition allowlist required; no targets/audit/length")
    key = "native_smiles" if "native_smiles" in conditions else "standard_selfies"
    if type(conditions[key]) is not str or not conditions[key]:
        raise ValueError("complete known native representation required")
    conditions[key].encode("utf8")  # Reject invalid Unicode; never repair it.
    return "smiles_to_selfies" if key == "native_smiles" else "selfies_to_smiles"


def pubchem_condition_graph(conditions):
    """Exact declared context only; full known strings, no target-derived topology.

    The existing shared encoder pools the complete entity payload lossily. This
    interface does not claim an atom/bond-expanded molecular graph representation.
    Graph payload packing preserves full known native string bytes before pooling.
    """
    _direction(conditions)
    record = GraphRecord("input", (Node(CONDITION_KIND, (
        Measurement("condition", _canonical(conditions).decode(), None, "observed", True),
        Measurement("requested_output", None, None, "missing", False))),), (),
        (Provenance("native-model-input", "PubChem-byte-v1", "input", "0"*64, "source-admission-external"),))
    task = ConditionalTask(((("node", 0, "condition"), "condition"),
                            (("node", 0, "requested_output"), "target")), topology_known=True)
    return record, task


class PubChemDecoder(nn.Module):
    """Source-bound native-byte head on the unchanged existing shared encoder."""
    def __init__(self, shared, *, source_root, manifest, expected_manifest_sha256,parameterization_policy='legacy'):
        super().__init__()
        if not isinstance(shared, SharedAnnotationModel):
            raise ValueError("existing shared annotation graph/payload model required")
        self._root = Path(source_root).resolve()
        path = (self._root / manifest).resolve()
        # Check source/code scope before creating any new random parameters.
        if (_sha(Path(native_reader.__file__).read_bytes()) != READER_SHA256
                or expected_manifest_sha256 not in native_reader.SUPPORTED_MASTER_SHA256
                or not path.is_relative_to(self._root)
                or _sha(path.read_bytes()) != expected_manifest_sha256):
            raise ValueError("exact pinned native reader and admitted v20/v21/v22 source required")
        metadata = json.loads(path.read_bytes())
        if expected_manifest_sha256 != PILOT_SHA256:
            native_reader._whole_metadata(self._root, metadata)
        self._manifest_sha256 = expected_manifest_sha256
        self._shards = {s["sha256"]: s["path"] for s in metadata["shards"]
            if s.get("source_family") in {x[0] for x in native_reader.DIRECTIONS.values()}}
        self._manifest = path.relative_to(self._root)
        spec=recurrent_parameterization(shared,parameterization_policy)
        self.shared = shared
        self._recurrent_parameterization=spec;self._recurrent_owned_spec=copy.deepcopy(spec)
        self._recurrent_shared_reference=(shared,)
        self.codec = PubChemByteCodec(shared.payload.codec.tokenizer)
        self._recurrent_codec=_canonical(self.codec.configuration())
        width = shared.payload.embedding.embedding_dim
        self.embedding = nn.Embedding(self.codec.size, width, padding_idx=self.codec.pad)
        self.rnn = (nn.GRU if spec is None else IndependentParameterGRU)(width,width,batch_first=True)
        self.output = nn.Linear(width, self.codec.size)
        initialize_recurrent_head(self,spec)
        p = next(shared.parameters())
        for head in (self.embedding, self.rnn, self.output):
            head.to(device=p.device, dtype=p.dtype)

    def _validate_recurrent_parameterization(self):
        if (not exact_wire(self._recurrent_parameterization,self._recurrent_owned_spec)
                or not exact_wire(self._recurrent_parameterization,recurrent_parameterization(self.shared,'mup-sequence-v1'))
                or self.shared is not self._recurrent_shared_reference[0]
                or self._recurrent_codec!=_canonical(self.codec.configuration())):
            raise ValueError('PubChem recurrent source/codec/parameterization drift')
        validate_recurrent_head(self,codec_size=self.codec.size,padding_idx=self.codec.pad)

    def get_extra_state(self):
        if self._recurrent_parameterization is not None:self._validate_recurrent_parameterization()
        result=dict(version=1, architecture="existing-conditional-initial-state-GRU",
            representation="native UTF8/raw-byte shortest-path tokens;notSELFIES-symbol-vocab",
            codec=self.codec.configuration(), shared=self.shared.get_extra_state(),
            payload=self.shared.payload.get_extra_state(),
            source_binding=dict(manifest=str(self._manifest), manifest_sha256=self._manifest_sha256, reader_sha256=READER_SHA256,
                **(dict(whole_component=native_reader.WHOLE_BINDING) if self._manifest_sha256 != PILOT_SHA256 else {})),
            condition_graph=dict(node_kind=CONDITION_KIND, known_entities=1, target_payload_excluded=True,
                condition_serialization="canonicalUTF8JSON;sourceIDs_audit_not_encoded"),
            target_roles=TARGETS, input_roles="exact admitted conditions only;target-free known one-entity context",
            objective="mean-all-lexical-next-token-and-EOS-CE;ignorePADonly;no-target-truncation",
            known_context="complete native strings before existing explicit lossy positional pooling",
            source_semantics="native prior versus computed standardSELFIES codec;notassay_or_GroupSELFIES")
        if self._recurrent_parameterization is not None:result.update(version=2,mup=copy.deepcopy(self._recurrent_parameterization))
        return copy.deepcopy(result)

    def set_extra_state(self, state):
        if not _exact_configuration(state, self.get_extra_state()):
            raise ValueError("PubChem native decoder/source/vocabulary/context configuration mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Preflight exact new-head state/config before mutating any parameters."""
        current = self.state_dict()
        if strict is not True or assign is not False or type(state_dict) not in (dict, type(current)):
            raise ValueError("strict identity-preserving PubChem state load required")
        if set(state_dict) != set(current):
            raise ValueError("complete exact PubChem/shared state keys required")
        for key, expected in current.items():
            value = state_dict[key]
            if isinstance(expected, torch.Tensor):
                if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype or value.shape != expected.shape
                        or not torch.isfinite(value).all()):
                    raise ValueError("PubChem/shared state tensor shape/dtype/value mismatch")
            elif not _exact_configuration(value, expected):
                raise ValueError("strict PubChem vocabulary/source/context state mismatch")
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._recurrent_parameterization is not None:recurrent_checkpoint_preflight(self,state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout(self,hidden):
        return recurrent_readout(self,hidden)

    def _views(self, views):
        views = tuple(views)
        if not views or len(views) > 64 or any(not isinstance(v, native_reader.PubChemNativeView) for v in views):
            raise ValueError("bounded nonempty corrected-reader PubChem views required")
        # Validate ALL source observations before target allocation or neural work.
        for view in views:
            try:
                if (view.direction not in TARGETS or any(type(x) is not bytes for x in
                        (view.input_bytes, view.target_bytes, view.target_native_bytes, view.provenance_bytes))):
                    raise ValueError("immutable native direction/bytes required")
                p = view.provenance()
                if (p["schema"] != "mora3.PubChemNativeAdmission/v1" or p["direction"] != view.direction
                        or p["corpus_binding"]["manifest_sha256"] != self._manifest_sha256):
                    raise ValueError("exact configured source version/component scope required")
                checked, = native_reader.read_pubchem_native_views(self._manifest, root=self._root,
                    expected_manifest_sha256=self._manifest_sha256, direction=view.direction,
                    task_ids=[p["envelope"]["task_id"]], limit=1,
                    shard_paths=(self._shards[p["corpus_binding"]["shard_sha256"]],))
                for field in ("direction", "input_bytes", "target_bytes", "target_native_bytes", "provenance_bytes", "binding_sha256"):
                    if getattr(view, field) != getattr(checked, field):
                        raise ValueError("exact source/target/type/receipt/group/view binding mismatch")
                if _direction(view.inputs()) != view.direction:
                    raise ValueError("native condition/direction mismatch")
            except (KeyError, TypeError, OSError, UnicodeError) as exc:
                raise ValueError("complete pinned PubChem native source admission required") from exc
        return views

    def condition(self, conditions):
        """Public target-free conditions; this interface does not grant admission."""
        conditions = tuple(conditions)
        if not conditions:
            raise ValueError("nonempty PubChem context required")
        records = [pubchem_condition_graph(c) for c in conditions]
        graphs = [self.shared.payload.forward_task(record, task=task,
            identifiers=self.shared.sequence_identifiers()) for record, task in records]
        return self.shared.backbone(graphs)[0][:, 0]

    def teacher_batch(self, views):
        views = self._views(views)
        ids = [self.codec.encode(v.target_native_bytes) for v in views]
        p = next(self.parameters())
        n = max(len(x)-1 for x in ids)
        prefix = torch.full((len(ids), n), self.codec.pad, device=p.device, dtype=torch.long)
        target = prefix.clone()
        for i, x in enumerate(ids):
            prefix[i, :len(x)-1] = prefix.new_tensor(x[:-1])
            target[i, :len(x)-1] = target.new_tensor(x[1:])
        return prefix, target

    def forward(self, conditions, prefix):
        if self._recurrent_parameterization is not None:self._validate_recurrent_parameterization()
        conditions = tuple(conditions)
        if (not isinstance(prefix, torch.Tensor) or prefix.dtype != torch.long or prefix.ndim != 2
                or prefix.device != self.embedding.weight.device or prefix.shape[0] != len(conditions)
                or prefix.shape[1] < 1 or bool(((prefix < 0) | (prefix >= self.codec.size)).any())
                or bool((prefix[:, 0] != self.codec.bos).any())):
            raise ValueError("valid caller-supplied native BOS/payload prefix required")
        context = self.condition(conditions)
        hidden, _ = self.rnn(self.embedding(prefix), context.unsqueeze(0).contiguous())
        return self._readout(hidden)

    def objective(self, views):
        views = tuple(views)
        prefix, target = self.teacher_batch(views)
        logits = self(tuple(v.inputs() for v in views), prefix)
        return F.cross_entropy(logits.reshape(-1, self.codec.size), target.reshape(-1), ignore_index=self.codec.pad)

    @torch.no_grad()
    def generate(self, conditions, *, max_new_tokens):
        """Caller budget only; preserve nontermination and invalid representation bytes."""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("explicit positive caller generation budget required")
        conditions = tuple(conditions)
        directions = tuple(_direction(c) for c in conditions)
        context = self.condition(conditions)
        result = self._generate_from_context(context, max_new_tokens=max_new_tokens)
        result['target_fields'] = tuple(TARGETS[d] for d in directions)
        return result

    @torch.no_grad()
    def _generate_from_context(self, context, *, max_new_tokens):
        reference = self.embedding.weight
        if (type(max_new_tokens) is not int or max_new_tokens < 1
                or not isinstance(context, torch.Tensor) or context.ndim != 2
                or not context.shape[0] or context.shape[1] != self.rnn.hidden_size
                or context.dtype != reference.dtype or context.device != reference.device
                or not bool(torch.isfinite(context).all())):
            raise ValueError('Complete finite native decoder context and positive generation budget required')
        if self._recurrent_parameterization is not None:
            self._validate_recurrent_parameterization()
        h = context.unsqueeze(0).contiguous()
        batch_size = context.shape[0]
        token = torch.full((batch_size, 1), self.codec.bos, device=h.device, dtype=torch.long)
        sequences = [[self.codec.bos] for _ in range(batch_size)]
        done = [False] * batch_size
        for _ in range(max_new_tokens):
            hidden, h = self.rnn(self.embedding(token), h)
            logits = self._readout(hidden[:, 0])
            logits[:, self.codec.pad] = -torch.inf
            logits[:, self.codec.bos] = -torch.inf
            token = logits.argmax(-1)[:, None]
            for i, value in enumerate(token[:, 0].tolist()):
                if not done[i]:
                    sequences[i].append(value)
                    done[i] = value == self.codec.eos
            if all(done):
                break
        raw = tuple(self.codec.partial_bytes(x) for x in sequences)
        text = []
        valid = []
        for value in raw:
            try:
                text.append(value.decode("utf8"))
                valid.append(True)
            except UnicodeDecodeError:
                text.append(None)
                valid.append(False)
        return dict(evidence_class="model_produced", token_ids=tuple(tuple(x) for x in sequences),
            terminated=tuple(done), representation_bytes=raw, text=tuple(text), utf8_valid=tuple(valid),
            representation_validated=False,
            caller_budget=max_new_tokens)
