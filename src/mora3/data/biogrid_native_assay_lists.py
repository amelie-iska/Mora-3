"""Read complete source-reported BioGRID assay lists under prospective source roles."""
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq


ROLE_CODES = ("audit_only", "train", "validation", "test")
ROLE_FIELDS = {
    "MCSA_guarded": ("prospective_MCSA_guarded_pair_role", "original_MCSA_event_role"),
    "MCSA_and_prospective_AFDB_guarded": (
        "prospective_MCSA_and_AFDB_guarded_pair_role", "original_MCSA_and_AFDB_event_role",
    ),
}
SOURCE_TASK = "biogrid_reported_assay_list"
SEQUENCE_TASK = "biogrid_reported_assay_list_from_current_reference_sequences"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(raw):
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class BioGRIDNativeAssayEvent:
    original_row_ordinal: int
    literal_assay_id: int
    original_publication_source: bytes | None
    original_order_reversed: bool
    original_guard_bits: int
    original_prospective_event_role: str


@dataclass(frozen=True)
class BioGRIDNativeAssayList:
    source_pair_ordinal: int
    unordered_source_pair: tuple[int, int]
    group_minimum_original_ordinal: int
    prospective_arm: str
    prospective_role: str
    events: tuple[BioGRIDNativeAssayEvent, ...]
    optional_current_reference_status_bits: tuple[int, int]
    optional_current_reference_ordinals: tuple[int | None, int | None]
    EOS_id: int

    @property
    def original_role(self):
        return None

    @property
    def optimizer_authorized(self):
        return False

    def condition_inputs(self):
        """Expose native pair IDs only. Assay and publication fields are audit/target data."""
        return {"task": SOURCE_TASK, "native_pair": tuple(map(str, self.unordered_source_pair))}

    def target_ids(self):
        """Keep exact original order and duplicates, followed by one list terminator."""
        if self.prospective_role == "audit_only":
            raise ValueError("an unassigned or partially guarded pair has no source objective")
        return tuple(event.literal_assay_id for event in self.events) + (self.EOS_id,)


class BioGRIDNativeAssayLists:
    """Exact read-only binding of the independently verified whole source list view."""

    def __init__(self, root, manifest_path, manifest_sha256, *, independent_proof_path, independent_proof_sha256):
        self._root = Path(root).resolve(strict=True)
        manifest_raw = self._read(manifest_path, manifest_sha256)
        independent_raw = self._read(independent_proof_path, independent_proof_sha256)
        manifest = json.loads(manifest_raw)
        proof = json.loads(independent_raw)
        if (
            manifest["schema"] != "Mora3.BioGRIDCompleteNativeAssayListCandidates/v1"
            or manifest["source_release"] != "5.0.262"
            or manifest["optimizer_admission"] is not False
            or manifest["prospective_pair_roles_review_pending"] is not True
            or proof["schema"] != "Mora3.IndependentOriginalBioGRIDAssayListProjectionVerification/v1"
            or proof["producer"]["sha256"] != manifest_sha256
            or self._path(proof["producer"]["path"]) != self._path(manifest_path)
            or proof["producer"]["bytes"] != len(manifest_raw)
            or proof["all_source_pair_lists_verified"] != manifest["exact_native_unordered_pairs"]
            or proof["selected_original_native_events_verified"] != manifest["original_human_physical_associations"]
            or proof["all_original_source_row_ordinals_accounted"] != manifest["exact_original_source_rows"]
            or proof["original_native_BioGRID"] != manifest["original_native_BioGRID"]
            or proof["no_partially_masked_pair_is_eligible"] is not True
            or proof["source_reference_coordinates_and_optional_endpoint_conservative_checks_verified"] is not True
            or proof["optimizer_admission"] is not False
        ):
            raise ValueError("exact independent full source-list and role proof required")
        axis = json.loads(self._read_reference(manifest["exact_native_assay_axis"]))
        symbols = axis["symbols"]
        names = tuple(symbol["native_Experimental_System"] for symbol in symbols)
        eos = len(names)
        if (
            axis["schema"] != "Mora3.BioGRIDLiteralSourceExperimentalSystemAxis/v1"
            or not names or any(type(name) is not str or not name or name == "-" for name in names)
            or tuple(sorted(set(names), key=str.encode)) != names
            or tuple(symbol["id"] for symbol in symbols) != tuple(range(eos))
            or axis["EOS_id"] != eos or axis["BOS_input_only_id"] != eos + 1
            or axis["output_classes"] != eos + 1 or axis["input_classes"] != eos + 2
            or manifest["total_output_classes"] != eos + 1 or manifest["total_additional_input_symbols"] != eos + 2
            or sum(symbol["original_rows"] for symbol in symbols) != manifest["original_human_physical_associations"]
        ):
            raise ValueError("exact ordered native assay axis and independent EOS/BOS required")
        self._axis = names
        self._eos = eos
        self._axis_sha256 = manifest["exact_native_assay_axis"]["sha256"]
        self._manifest_sha256 = manifest_sha256
        self._maximum_complete_assay_list = manifest["maximum_original_complete_assay_list"]
        self._pairs = self._parquet(manifest["all_pairs"])
        if self._pairs.metadata.num_rows != manifest["exact_native_unordered_pairs"]:
            raise ValueError("complete original pair count differs")
        counts = [0]
        for index in range(self._pairs.num_row_groups):
            counts.append(counts[-1] + self._pairs.metadata.row_group(index).num_rows)
        self._offsets = tuple(counts)
        self._reference = self._parquet(manifest["immutable_current_reference_coordinates"]).read(use_threads=False)
        self._pair_group = lru_cache(maxsize=4)(self._read_pair_group)
        self._reference_group = lru_cache(maxsize=8)(self._read_reference_part)

    @property
    def literal_assay_axis(self):
        return self._axis

    @property
    def assay_axis_sha256(self):
        return self._axis_sha256

    @property
    def manifest_sha256(self):
        return self._manifest_sha256

    @property
    def EOS_id(self):
        return self._eos

    @property
    def BOS_input_only_id(self):
        return self._eos + 1

    def _path(self, value):
        path = (self._root / value).resolve(strict=True)
        if not path.is_relative_to(self._root) or not path.is_file():
            raise ValueError("source reference must be a file inside the bound project")
        return path

    def _read(self, value, expected, *, size=None):
        if type(expected) is not str or _SHA256.fullmatch(expected) is None:
            raise ValueError("literal expected source SHA256 required")
        raw = self._path(value).read_bytes()
        if _digest(raw) != expected or (size is not None and len(raw) != size):
            raise ValueError("pinned original or source projection changed")
        return raw

    def _read_reference(self, ref):
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
            raise ValueError("exact positive source file size required")
        return self._read(ref["path"], ref["sha256"], size=ref["bytes"])

    def _parquet(self, ref):
        return pq.ParquetFile(io.BytesIO(self._read_reference(ref)))

    def _read_pair_group(self, group_index):
        return self._pairs.read_row_group(group_index, use_threads=False)

    def _read_reference_part(self, path, digest):
        raw = self._read(path, digest)
        return pq.ParquetFile(io.BytesIO(raw)).read(
            columns=["primary_accession", "full_source_json", "source_structured_sha256"],
            use_threads=False,
        )

    def _decode(self, source, ordinal, arm):
        pair_field, event_field = ROLE_FIELDS[arm]
        pair = (source["native_BioGRID_interactor_A"], source["native_BioGRID_interactor_B"])
        if not all(type(x) is int and x > 0 for x in pair) or pair[0] > pair[1]:
            raise ValueError("literal positive canonical BioGRID pair required")
        role = source[pair_field]
        event_roles = source[event_field]
        all_original_roles = set(event_roles)
        if (
            type(role) is not int or role not in range(4)
            or not all(type(value) is int and value in range(4) for value in all_original_roles)
            or not all_original_roles
            or len(all_original_roles - {0}) > 1
            or role != min(all_original_roles)
        ):
            raise ValueError("source pair role cannot retain any partially guarded event list")
        events = []
        for original, assay, pub, reversed_pair, bits, event_role in zip(
            source["original_source_row_ordinals"], source["original_assay_axis_ids"],
            source["original_publication_source_fields"], source["original_endpoint_order_reversed_from_pair"],
            source["original_exposure_bits"], event_roles, strict=True,
        ):
            if (
                type(original) is not int or original < 0
                or (events and events[-1].original_row_ordinal >= original)
                or type(assay) is not int or not 0 <= assay < self._eos
                or (pub is not None and type(pub) is not bytes)
                or type(reversed_pair) is not bool or (pair[0] == pair[1] and reversed_pair)
                or type(bits) is not int or not 0 < bits < 256 or not bits & 1
            ):
                raise ValueError("complete original ordered assay/publication event required")
            events.append(BioGRIDNativeAssayEvent(original, assay, pub, reversed_pair, bits, ROLE_CODES[event_role]))
        if not 0 < len(events) <= self._maximum_complete_assay_list:
            raise ValueError("empty or incomplete source event list")
        component = source["known_component_min_original_ordinal"]
        if type(component) is not int or not 0 <= component <= events[0].original_row_ordinal:
            raise ValueError("complete known original pair/publication group required")
        status = tuple(source[f"pair_{side}_current_reference_status_bits"] for side in ("A", "B"))
        references = tuple(source[f"pair_{side}_unique_current_reference_index"] for side in ("A", "B"))
        for state, reference in zip(status, references, strict=True):
            if (
                type(state) is not int or not 0 <= state < 32
                or (state == 0 and (type(reference) is not int or not 0 <= reference < len(self._reference)))
                or (state != 0 and reference is not None)
            ):
                raise ValueError("unresolved native endpoint must not fabricate a current reference")
        return BioGRIDNativeAssayList(
            ordinal, pair, component, arm, ROLE_CODES[role], tuple(events), status, references, self._eos,
        )

    def pair_at(self, ordinal, *, arm):
        """Read an exact source pair by derived table ordinal, including audit-only rows."""
        if type(arm) is not str or arm not in ROLE_FIELDS:
            raise ValueError("explicit sealed prospective source arm required")
        if type(ordinal) is not int or not 0 <= ordinal < self._offsets[-1]:
            raise ValueError("original-derived pair table ordinal out of bounds")
        group = bisect_right(self._offsets, ordinal) - 1
        item = self._pair_group(group).slice(ordinal - self._offsets[group], 1).to_pylist()[0]
        return self._decode(item, ordinal, arm)

    def iter_pairs(self, *, arm, prospective_role):
        """Stream complete lists for one reviewed-as-candidate source role."""
        if type(arm) is not str or arm not in ROLE_FIELDS:
            raise ValueError("explicit sealed prospective source arm required")
        if type(prospective_role) is not str or prospective_role not in ROLE_CODES:
            raise ValueError("explicit original-derived prospective role required")
        code = ROLE_CODES.index(prospective_role)
        role_field = ROLE_FIELDS[arm][0]
        for group, offset in enumerate(self._offsets[:-1]):
            table = self._pair_group(group)
            matched = [i for i, value in enumerate(table[role_field].to_pylist()) if value == code]
            selected = table.take(pa.array(matched, type=pa.int64())).to_pylist()
            for local, record in zip(matched, selected, strict=True):
                yield self._decode(record, offset + local, arm)

    def require_pair(self, pair, *, arm, prospective_role):
        """Rebind an exact whole source list with explicit role at a consumer boundary."""
        if (
            type(pair) is not BioGRIDNativeAssayList or pair.prospective_arm != arm
            or pair.prospective_role != prospective_role
            or self.pair_at(pair.source_pair_ordinal, arm=arm) != pair
        ):
            raise ValueError("authentic complete BioGRID pair, source arm and exact candidate role required")

    def _current_sequence(self, index):
        expected = self._reference.slice(index, 1).to_pylist()[0]
        part = self._reference_group(expected["source_full_parquet_path"], expected["source_full_parquet_sha256"])
        source = part.slice(expected["source_full_parquet_row_ordinal"], 1).to_pylist()[0]
        source_raw = source["full_source_json"].encode("utf-8")
        payload = json.loads(source_raw)
        sequence = payload["sequence"]["value"]
        if (
            expected["source_release"] != "2026_03"
            or source["primary_accession"] != expected["primary_accession"]
            or payload["primaryAccession"] != expected["primary_accession"]
            or payload["organism"]["taxonId"] != 9606 or expected["organism_taxon_id"] != 9606
            or _digest(source_raw) != source["source_structured_sha256"]
            or source["source_structured_sha256"] != expected["source_structured_entry_sha256"]
            or payload["entryAudit"]["entryVersion"] != expected["entry_version"]
            or payload["entryAudit"]["sequenceVersion"] != expected["sequence_version"]
            or type(sequence) is not str or not sequence
            or len(sequence) != expected["canonical_sequence_length"]
            or _digest(sequence.encode("ascii")) != expected["canonical_sequence_sha256"]
        ):
            raise ValueError("exact original frozen UniProt accession, version and canonical sequence required")
        return sequence

    def current_reference_condition_inputs(self, pair, *, arm, prospective_role):
        """Optionally bind both complete current reference sequences, preserving pair order."""
        self.require_pair(pair, arm=arm, prospective_role=prospective_role)
        if pair.prospective_role == "audit_only" or pair.optional_current_reference_status_bits != (0, 0):
            raise ValueError("both original endpoints must uniquely support an eligible current reference")
        references = tuple(self._current_sequence(index) for index in pair.optional_current_reference_ordinals)
        return {"task": SEQUENCE_TASK, "current_reference_pair_sequences": references}
