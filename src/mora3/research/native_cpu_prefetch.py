"""Explicitly admitted bounded CPU staging for native source selections.

This helper only stages the existing source-only selector. It never evaluates a
model, commits a route cursor, or reads source data before ``admit`` is called.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

from .primary_native_routes import NativeRouteBinding, select_native_route_targets


@dataclass(frozen=True)
class _AdmittedSelection:
    binding: NativeRouteBinding
    cursor: bytes
    max_windows: int
    future: Future


class NativeRouteCPUStager:
    """Stage up to four explicitly admitted source-only route selections.

    Tickets are local handles, not route state. ``consume`` requires the exact
    binding object and literal cursor bytes supplied at admission. Selection
    output and selector exceptions are returned unchanged.
    """

    def __init__(self, *, workers=4):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('Native CPU staging worker count must be in [1, 4]')
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mora3-native-stage')
        self._lock = RLock()
        self._pending = {}
        self._retiring = set()
        self._next_ticket = 0
        self._closed = False

    def admit(self, binding, cursor, *, max_windows):
        """Start one bounded source-only selection after caller admission."""
        if type(binding) is not NativeRouteBinding or type(cursor) is not bytes:
            raise ValueError('Exact native route binding and serialized cursor required')
        if type(max_windows) is not int or max_windows < 1:
            raise ValueError('Finite positive selection window bound required')
        with self._lock:
            if self._closed:
                raise RuntimeError('Native route CPU stager is closed')
            if len(self._pending) >= 4:
                raise RuntimeError('Native route CPU stager outstanding-selection limit is four')
            if any(item.binding is binding and item.cursor == cursor for item in self._pending.values()):
                raise ValueError('Identical binding/cursor selection is already outstanding')
            ticket = self._next_ticket
            self._next_ticket += 1
            future = self._executor.submit(select_native_route_targets, binding, cursor, max_windows=max_windows)
            self._pending[ticket] = _AdmittedSelection(binding, cursor, max_windows, future)
            return ticket

    def consume(self, ticket, binding, cursor):
        """Return exactly the admitted selector result for its still-current cursor."""
        if type(ticket) is not int or type(binding) is not NativeRouteBinding or type(cursor) is not bytes:
            raise ValueError('Exact ticket, binding and serialized cursor required')
        with self._lock:
            item = self._pending.get(ticket)
            if item is None or ticket in self._retiring:
                raise ValueError('Unknown, consumed or discarded native staging ticket')
            if item.binding is not binding or item.cursor != cursor:
                raise ValueError('Native staging binding or literal cursor is stale')
            self._retiring.add(ticket)
        try:
            return item.future.result()
        finally:
            with self._lock:
                self._pending.pop(ticket,None)
                self._retiring.discard(ticket)

    def prefetch(self, binding, cursor, *, max_windows):
        """Start an upcoming exact selection if there is room in the queue."""
        with self._lock:
            if self._closed or len(self._pending)>=4:
                return False
            if any(item.binding is binding and item.cursor==cursor for item in self._pending.values()):
                return False
            self.admit(binding,cursor,max_windows=max_windows)
            return True

    def select(self, binding, cursor, *, max_windows):
        """Consume only an identical selection, or use the serial selector."""
        with self._lock:
            ticket=next((key for key,item in self._pending.items() if key not in self._retiring
                and item.binding is binding and item.cursor==cursor and item.max_windows==max_windows),None)
        if ticket is None:
            return select_native_route_targets(binding,cursor,max_windows=max_windows)
        return self.consume(ticket,binding,cursor)

    def discard(self, ticket):
        """Cancel stale queued work, or drain an already running read safely."""
        if type(ticket) is not int:
            raise ValueError('Exact native staging ticket required')
        with self._lock:
            item = self._pending.get(ticket)
            if item is None or ticket in self._retiring:
                return False
            self._retiring.add(ticket)
        try:
            if not item.future.cancel():
                try:
                    item.future.result()
                except Exception:
                    # Source failures on discarded selections do not become
                    # observations or affect the active campaign cursor.
                    pass
            return True
        finally:
            with self._lock:
                self._pending.pop(ticket,None)
                self._retiring.discard(ticket)

    def close(self):
        """Stop admission and cancel queued work; wait for active reads to finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
