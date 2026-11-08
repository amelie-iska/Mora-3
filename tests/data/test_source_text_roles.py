from dataclasses import replace

import pytest

from mora3.data.source_text_native_rows import NativeTextRow
from mora3.data.source_text_roles import NativeTextRoleView, ReviewedTextRow, _selection


def test_roles_are_literal_unique_and_preserve_the_native_role():
    native = NativeTextRow(
        "INV-084/HumanEval", b'{"row":1}', b'{"prompt":"example"}', b"full solution",
        b'{"native":"audit"}', ("exact-parent",), "evaluation_only", True, "fixture",
    )
    row = ReviewedTextRow(native, "evaluation_only")
    assert row.original_role == "evaluation_only"
    assert not row.validation_selection_permitted and not row.training_authorized
    with pytest.raises(ValueError):
        _selection(("evaluation_only", "evaluation_only"), ("evaluation_only",), "roles")


def test_source_fingerprint_refuses_mutated_audit_target_and_role():
    native = NativeTextRow("INV-081", b'{"row":1}', b"{}", b"exact", b"{}",
                           ("parent",), None, False, "fixture")
    # Isolate mutation checking. The actual integration verification supplies
    # the coordinator-reviewed packet and authentic source files.
    view = object.__new__(NativeTextRoleView)
    view._issued = {}
    view._require_unchanged = lambda: None
    row = view._issue(native, "validation")
    view.require_row(row, role="validation")
    assert row.original_role is None and row.validation_selection_permitted
    for tampered in (
        replace(row, prospective_role="train"),
        replace(row, native_row=replace(native, full_target_utf8=b"prefix")),
        replace(row, native_row=replace(native, original_role="train")),
        replace(row, native_row=replace(native, source_audit_json=b'{"new":true}')),
    ):
        with pytest.raises(ValueError, match="unaltered issued"):
            view.require_row(tampered, role=tampered.prospective_role)
