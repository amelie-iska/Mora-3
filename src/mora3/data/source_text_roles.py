"""Consume the reviewed source text and observed root role ledger."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

from . import source_text_native_rows as native
from .source_text_native_rows import NativeTextRow, SourceTextNativeBinding


ROLES = ("train", "validation", "test", "evaluation_only")
ROOT_FAMILY = "INV-082/OSHB-observed-root-list"
FAMILIES = ("INV-081", "INV-080", "INV-082/OSHB", "INV-084/GSM8K", "INV-084/HumanEval", "INV-083", ROOT_FAMILY)


def _sha(raw):
    return hashlib.sha256(raw).digest()


def _row_key(row):
    return row.source_family, _sha(row.source_coordinate_json)


def _row_fingerprint(row):
    return (
        _sha(row.conditions_json), _sha(row.full_target_utf8), _sha(row.source_audit_json),
        row.group_keys, row.original_role, row.evaluation_only, row.rights_disposition,
    )


def _selection(values, supported, name):
    if (type(values) not in (tuple, list) or not values or len(set(values)) != len(values)
            or any(value not in supported for value in values)):
        raise ValueError("exact unique reviewed " + name + " required")
    return frozenset(values)


@dataclass(frozen=True)
class ReviewedTextRow:
    native_row: NativeTextRow
    prospective_role: str

    @property
    def original_role(self):
        return self.native_row.original_role

    @property
    def validation_selection_permitted(self):
        return self.prospective_role == "validation"

    @property
    def training_authorized(self):
        return False


class NativeTextRoleView:
    """An immutable source role assignment with separate original roles.

    Rows are issued only after the corresponding native data and reviewed
    ledger entry agree. A caller can authenticate those rows again at the
    model boundary without an independent role or target assertion.
    """

    @classmethod
    def from_review_packet(cls, path, expected_sha256, *, root):
        return cls(path, expected_sha256, root=root)

    def __init__(self, path, expected_sha256, *, root):
        self._root = Path(root).resolve(strict=True)
        self._identities = {}
        self._issued = {}
        review_path = Path(path)
        review_raw = self._read({
            "path": str(review_path), "bytes": review_path.stat().st_size, "sha256": expected_sha256,
        })
        review = native._json(review_raw)
        if (review.get("schema") != "Mora3.RootReviewedNativeTextRootRoleAssignment/v1"
                or review.get("review_state") != "accepted-scoped-source-role-assignment"
                or review.get("owner") != "root-coordinator"
                or review.get("full_project_training_authorized") is not False
                or review.get("optimizer_launch_authorized") is not False
                or review.get("source_original_roles_rewritten") is not False
                or review.get("source_body_or_label_rewrite") is not False):
            raise ValueError("exact coordinator review of prospective source roles required")

        base = native._json(self._read(review["base_role_proposal"]))
        delta = native._json(self._read(review["inline_word_delta"]))
        proof = native._json(self._read(review["independent_role_verification"]))
        counts = review["prospective_role_counts"]
        if (proof["schema"] != "Mora3.IndependentFiniteTextRootRoleProposal/v1"
                or proof["base_role_proposal"] != review["base_role_proposal"]
                or proof["inline_delta_proposal"] != review["inline_word_delta"]
                or delta["parent_role_proposal"] != review["base_role_proposal"]
                or proof["proposed_roles"] != counts or delta["new_proposed_role_counts"] != counts
                or set(counts) != set(ROLES) or any(type(n) is not int or n < 1 for n in counts.values())
                or review["source_views"] != sum(counts.values()) or review["seed"] != base["seed"]
                or proof["source_role_not_rewritten"] is not True
                or proof["all_known_parent_roles_consistent"] is not True
                or base["source_handoff"] != review["source_handoff"]
                or delta["source_handoff"] != review["source_handoff"]
                or delta["production_reader"] != review["production_reader"]):
            raise ValueError("review, independent proof and source ledgers differ")
        reader = review["production_reader"]
        if self._path(reader) != Path(native.__file__).resolve(strict=True):
            raise ValueError("reviewed production reader differs from the loaded module")
        self._read(reader)
        handoff = review["source_handoff"]
        self._source = SourceTextNativeBinding(self._path(handoff), handoff["sha256"], root=self._root)
        self._base_roles = self._load_text_roles(base["text_role_ledger"])
        self._word_qc = base["native_word_root_role_QC"]
        inline = native._json(self._read(delta["all11_word_deltas"]))
        additions = [native._json(line) for line in self._read(delta["additional2_observed_root_list_candidates"]).splitlines()]
        self._inline = {_sha(native._wire(row["source_coordinate"])): native._wire(row) for row in inline}
        self._root_additions = {_sha(native._wire(row["source_coordinate"])): native._wire(row) for row in additions}
        if len(self._inline) != len(inline) or len(self._root_additions) != len(additions):
            raise ValueError("duplicate reviewed inline coordinate")
        self._family_counts = Counter(key[0] for key in self._base_roles)
        self._family_counts[ROOT_FAMILY] = proof["qualified_source_root_list_views"]
        self._word_count = proof["complete_native_word_contexts"]
        if (len(self._base_roles) != proof["complete_native_text_rows"]
                or self._family_counts != delta["new_counts_by_source_family"]
                or set(self._family_counts) != set(FAMILIES)
                or self._family_counts[ROOT_FAMILY] != base["counts_by_source_family"][ROOT_FAMILY] + len(additions)):
            raise ValueError("complete source text and observed root denominators differ")
        expected = Counter(item[0] for item in self._base_roles.values())
        expected["train"] += self._family_counts[ROOT_FAMILY]
        if expected != counts:
            raise ValueError("source role ledger counts differ from the reviewed role totals")
        self._review_sha256 = expected_sha256
        self._role_counts = tuple((role, counts[role]) for role in ROLES)

    @property
    def role_counts(self):
        return dict(self._role_counts)

    @property
    def review_sha256(self):
        return self._review_sha256

    def _path(self, reference):
        native._hash(reference)
        path = (self._root / reference["path"]).resolve(strict=True)
        if not path.is_relative_to(self._root):
            raise ValueError("reviewed source artifacts must be inside the bound repository")
        return path

    def _read(self, reference):
        path = self._path(reference)
        before = native._identity(path)
        raw = path.read_bytes()
        if (before != native._identity(path) or len(raw) != reference["bytes"]
                or _sha(raw).hex() != reference["sha256"]
                or self._identities.get(path, before) != before):
            raise ValueError("reviewed source artifact identity, length or SHA differs")
        self._identities[path] = before
        return raw

    def _load_text_roles(self, reference):
        roles = {}
        groups = {}
        for line in io.BytesIO(self._read(reference)):
            item = native._json(line)
            family, role = item["source_family"], item["proposed_role"]
            key = family, _sha(native._wire(item["source_coordinate"]))
            original = item["original_role"]
            if (family not in FAMILIES or family == ROOT_FAMILY or key in roles or role not in ROLES
                    or item["validation_selection_permitted"] is not (role == "validation")
                    or item["source_candidate_not_optimization_authorization"] is not True
                    or original == "test" and role != "test"
                    or (original == "evaluation_only") != (role == "evaluation_only")):
                raise ValueError("reviewed source role or original evaluation control differs")
            for group in item["group_keys"]:
                if groups.setdefault(group, role) != role:
                    raise ValueError("one reviewed source parent crosses prospective roles")
            roles[key] = (
                role, original, tuple(item["group_keys"]), item["target_UTF8_bytes"],
                bytes.fromhex(item["target_SHA256"]), bytes.fromhex(item["condition_JSON_SHA256"]),
                bytes.fromhex(item["source_audit_canonical_JSON_SHA256"]),
            )
        return roles

    def _require_unchanged(self):
        if any(native._identity(path) != identity for path, identity in self._identities.items()):
            raise ValueError("reviewed role artifact changed")
        self._source.require_sources_unchanged()

    def _issue(self, row, role):
        key = _row_key(row)
        value = role, _row_fingerprint(row)
        if self._issued.setdefault(key, value) != value:
            raise ValueError("reviewed source row changed within this view")
        return ReviewedTextRow(row, role)

    def require_row(self, row, *, role):
        if (type(row) is not ReviewedTextRow or type(row.native_row) is not NativeTextRow
                or role not in ROLES or row.prospective_role != role
                or self._issued.get(_row_key(row.native_row)) != (role, _row_fingerprint(row.native_row))):
            raise ValueError("unaltered issued source row and reviewed prospective role required")
        self._require_unchanged()

    def _text_role(self, row):
        entry = self._base_roles.get(_row_key(row))
        if entry is None:
            raise ValueError("native text coordinate is absent from the reviewed role ledger")
        role = entry[0]
        supplied = (
            row.original_role, row.group_keys, len(row.full_target_utf8), _sha(row.full_target_utf8),
            _sha(row.conditions_json), _sha(row.source_audit_json),
        )
        if entry[1:] != supplied or row.evaluation_only is not (role == "evaluation_only"):
            raise ValueError("native text contents, provenance or original role differ from the review")
        return role

    def _iter_root_rows(self):
        import pyarrow.parquet as pq

        original = pq.ParquetFile(io.BytesIO(self._read(self._word_qc)))
        contexts = self._source.iter_oshb_word_contexts()
        word_count = root_count = 0
        seen_inline = set()
        seen_additions = set()
        qc_rows = (row for batch in original.iter_batches(batch_size=1024) for row in batch.to_pylist())
        for context, qc in zip(contexts, qc_rows, strict=True):
            word_count += 1
            audit = native._json(context.source_audit_json)
            for name in ("source_object_ordinal", "source_word_ordinal", "native_word_ID", "source_book_OSIS_ID"):
                if audit[name] != qc[name]:
                    raise ValueError("exact original OSHB word coordinate differs")
            roots = native._json(context.direct_root_assertions_json)
            coord = _sha(context.source_coordinate_json)
            if (context.root_target_status != qc["root_target_status"]
                    or len(roots) != qc["observed_root_binding_count"]
                    or _sha(context.direct_root_assertions_json) != qc["native_ordered_root_assertions_SHA256"]
                    or _sha(audit["native_word_direct_text"].encode("utf-8")) != qc["observed_word_UTF8_SHA256"]
                    or qc["original_role"] is not None):
                raise ValueError("exact original OSHB word or direct root evidence differs")
            inline = self._inline.get(coord)
            if inline is not None:
                item = native._json(inline)
                seen_inline.add(coord)
                if (item["old_word_input_status"] != qc["word_input_status"]
                        or item["new_word_input_status"] != context.word_input_status
                        or item["full_native_word_UTF8_SHA256"] != _sha(context.observed_word_utf8).hex()
                        or item["native_inline_subtree"] != audit["original_native_word_XML_subtree"]):
                    raise ValueError("reviewed complete inline word evidence differs")
            elif context.word_input_status != qc["word_input_status"]:
                raise ValueError("unreviewed OSHB word input change")
            addition = self._root_additions.get(coord)
            if not roots:
                if addition is not None or qc["proposed_role"] is not None or qc["root_list_target_UTF8_SHA256"] is not None:
                    raise ValueError("unlabeled source word cannot become root supervision")
                continue
            row = context.observed_root_list_view()
            target_sha = _sha(row.full_target_utf8)
            if addition is None:
                if qc["proposed_role"] != "train" or qc["root_list_target_UTF8_SHA256"] != target_sha:
                    raise ValueError("reviewed source root target differs")
            else:
                item = native._json(addition)
                seen_additions.add(coord)
                if (inline is None or qc["proposed_role"] is not None or qc["root_list_target_UTF8_SHA256"] is not None
                        or item["proposed_role"] != "train" or item["original_role"] is not None
                        or item["target_SHA256"] != target_sha.hex()
                        or native._wire(item["target_UTF8_JSON_root_list"]) != row.full_target_utf8):
                    raise ValueError("reviewed inline native root list differs")
            root_count += 1
            yield row
        if (word_count != self._word_count or root_count != self._family_counts[ROOT_FAMILY]
                or seen_inline != set(self._inline) or seen_additions != set(self._root_additions)):
            raise ValueError("complete reviewed OSHB word/root universe differs")

    def iter_rows(self, *, roles=ROLES, families=FAMILIES):
        """Read all selected sources with exact prospective role dispositions."""
        roles = _selection(roles, ROLES, "prospective roles")
        families = _selection(families, FAMILIES, "source families")
        self._require_unchanged()
        native_families = tuple(
            family for family in native.FAMILIES
            if family in families or family == "INV-084" and any(x.startswith("INV-084/") for x in families)
        )
        seen = set()
        counts = Counter()
        if native_families:
            for row in self._source.iter_rows(families=native_families):
                if row.source_family not in families:
                    continue
                key = _row_key(row)
                if key in seen:
                    raise ValueError("duplicate original native text coordinate")
                seen.add(key)
                counts[row.source_family] += 1
                role = self._text_role(row)
                if role in roles:
                    yield self._issue(row, role)
        expected = Counter({family: self._family_counts[family] for family in families if family != ROOT_FAMILY})
        if counts != expected:
            raise ValueError("native text row coverage differs from the reviewed universe")
        if ROOT_FAMILY in families and "train" in roles:
            for row in self._iter_root_rows():
                yield self._issue(row, "train")
        self._require_unchanged()
