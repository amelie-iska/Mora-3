"""Prepared full registry for model fixtures; production reader tests stay uncached.

Only immutable metadata is shared, never models, optimizers, RNG, tasks or paths.
The complete source registry is verified by the native reader on first use.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from mora3.data.annotation_axes import AnnotationAxes, read_annotation_axes


@dataclass(frozen=True)
class PreparedRegistry(AnnotationAxes):
    axis_items: tuple
    binding_json: str

    @property
    def categories(self):
        return tuple(k for k, _ in self.axis_items)

    def axis(self, category):
        return dict(self.axis_items)[category]

    def binding(self):
        return json.loads(self.binding_json)


@lru_cache(maxsize=1)
def fixture_registry():
    from test_annotation_axes import REAL
    source = read_annotation_axes(REAL)
    parsed = json.loads(source.specification_json)
    return PreparedRegistry(source.specification_json,
        tuple((k, tuple(r['axis_id'] for r in rows)) for k, rows in parsed['rows'].items()),
        json.dumps(parsed['binding'], sort_keys=True, allow_nan=False))
