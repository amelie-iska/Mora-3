"""Fresh configuration deepcopy without revisiting immutable string axes.

No retained cache, model/source validation bypass, tensor copy shortcut, or global
copy-dispatch change. Exact string tuples are already returned by identity by
stdlib deepcopy; pre-memoizing them only avoids repeated recursive atomic copies.
"""
import copy


def configuration_snapshot(value):
    """Keep stdlib copy/alias/cycle semantics for native plain metadata graphs.

    Preflight exact builtin containers and atoms only. For any unrecognized type
    or non-string dictionary key, use the unchanged stdlib path with a fresh memo:
    custom copy hooks therefore never observe our optimization memo. Mutable
    string lists receive fresh shallow copies (their atoms are immutable); other
    mutable containers are copied by stdlib. All detection/memo state is call-local.
    """
    pending = [value]
    seen = set()
    memo = {}
    while pending:
        item = pending.pop()
        kind = type(item)
        if (kind is str or kind is int or kind is float or kind is bool
                or kind is type(None) or kind is bytes or kind is complex):
            continue
        if kind is not dict and kind is not list and kind is not tuple:
            return copy.deepcopy(value)
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if kind is dict:
            if any(type(key) is not str for key in item):
                return copy.deepcopy(value)
            if all(type(v) is str or type(v) is int or type(v) is float
                   or type(v) is bool or type(v) is type(None)
                   or type(v) is bytes or type(v) is complex for v in item.values()):
                # Exact immutable values only; preserve order and atom identities.
                # Every call allocates new dict owners; aliases use this local memo.
                memo[identity] = item.copy()
            else:
                pending.extend(item.values())
        elif kind is tuple and len(item) >= 16 and all(type(v) is str for v in item):
            memo[identity] = item
        elif kind is list and len(item) >= 16 and all(type(v) is str for v in item):
            # Strings are exact immutable atoms; the mutable owner is ALWAYS new.
            # Call-local memo preserves aliases, never sharing the original list.
            memo[identity] = item.copy()
        else:
            pending.extend(item)
    return copy.deepcopy(value, memo)
