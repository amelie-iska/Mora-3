"""Passive flat string-key numeric mapping grammar for author dictionaries.

pickletools parses opcode operands only. This module never imports NumPy,
resolves serialized names, or invokes a serialized callable. REDUCE and BUILD
are interpreted exclusively as exact inert scalar/dtype recipes. Everything
outside this small protocol4/5 grammar refuses; no generic unpickle fallback.
"""
from collections import Counter
import hashlib
import math
import pickletools
import struct


class Refusal(ValueError):
    pass


class Node:
    __slots__ = ('kind', 'value', 'start', 'stop', 'state')

    def __init__(self, kind, value, start, stop):
        self.kind, self.value, self.start, self.stop = kind, value, start, stop
        self.state = None


DTYPES = {'i1': 'b', 'i2': 'h', 'i4': 'i', 'i8': 'q',
          'u1': 'B', 'u2': 'H', 'u4': 'I', 'u8': 'Q', 'f4': 'f', 'f8': 'd'}
SYMBOLS = {('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
           ('numpy', 'dtype')}


def decode_numeric_mapping(raw):
    if type(raw) is not bytes or not 4 <= len(raw) <= 1048576 or raw[:2] not in (b'\x80\x04', b'\x80\x05'):
        raise Refusal('bounded protocol4/5 immutable bytes required')
    try:
        operations = list(pickletools.genops(raw))
    except Exception as error:
        raise Refusal('invalid passive opcode syntax') from error
    if len(operations) > 400000 or operations[-1][0].name != 'STOP' or operations[-1][2] != len(raw) - 1:
        raise Refusal('opcode limit or complete STOP/EOF required')
    stack, memo, pairs, seen = [], {}, [], set()
    marker, root = object(), object()
    root_created = False
    frame_end = None
    literal_total = 0
    dtype_nodes, scalar_nodes, frames, memo_reads = [], [], [], []
    counts = Counter()

    def push(value):
        if len(stack) >= 4096:
            raise Refusal('bounded stack')
        stack.append(value)

    def pop():
        if not stack or stack[-1] is marker or stack[-1] is root:
            raise Refusal('exact scalar/recipe stack shape')
        return stack.pop()

    def group():
        index = next((i for i in range(len(stack)-1, -1, -1) if stack[i] is marker), None)
        if index is None:
            raise Refusal('MARK required')
        values = stack[index+1:]
        del stack[index:]
        return values

    def native_value(value):
        if type(value) is not Node or value.kind not in ('int', 'numpy_scalar'):
            raise Refusal('only exact primitive integer or inert numeric scalar values')
        if value.kind == 'int':
            return dict(kind='primitive_int', value=value.value, source_opcode_start=value.start,
                        source_opcode_stop=value.stop)
        dtype, payload, reduction = value.value
        if dtype.state is None:
            raise Refusal('scalar dtype state not built')
        token, order = dtype.value, dtype.state
        if payload.kind != 'bytes' or len(payload.value) != int(token[1:]):
            raise Refusal('exact dtype scalar payload byte count')
        fmt = ('<' if order == '|' else order) + DTYPES[token]
        number = struct.unpack(fmt, payload.value)[0]
        finite = math.isfinite(number) if token[0] == 'f' else True
        return dict(kind='numpy_scalar', numpy_dtype=token, byte_order=order,
                    value=number if finite else None, finite=finite, payload_hex=payload.value.hex(),
                    payload_opcode_start=payload.start, payload_opcode_stop=payload.stop,
                    scalar_reduce_offset=reduction,
                    dtype_recipe_start=dtype.start, dtype_recipe_stop=dtype.stop)

    def commit(values):
        if len(values) % 2 or len(pairs) + len(values)//2 > 100000:
            raise Refusal('complete bounded ordered mapping pairs')
        for key, value in zip(values[::2], values[1::2]):
            if type(key) is not Node or key.kind != 'str' or key.value in seen:
                raise Refusal('unique literal string keys; no overwrite or coercion')
            seen.add(key.value)
            pairs.append(dict(ordinal=len(pairs), key=key.value, key_opcode_start=key.start,
                              key_opcode_stop=key.stop, value=native_value(value)))

    for i, (op, arg, start) in enumerate(operations):
        name = op.name
        stop = operations[i+1][2] if i+1 < len(operations) else start+1
        counts[name] += 1
        if frame_end is not None and start == frame_end:
            frame_end = None
        if frame_end is not None and stop > frame_end:
            raise Refusal('operand crosses declared frame')
        if name == 'PROTO':
            if i != 0 or start != 0 or arg not in (4, 5):
                raise Refusal('one exact protocol prefix')
        elif name == 'FRAME':
            if frame_end is not None or type(arg) is not int or not 0 < arg <= 1048576 or stop + arg > len(raw) or len(frames) >= 256:
                raise Refusal('bounded nonoverlapping frame')
            frame_end = stop + arg
            frames.append(dict(start=start, payload_start=stop, payload_stop=frame_end))
        elif name == 'EMPTY_DICT':
            if stack or root_created:
                raise Refusal('one flat mapping root only')
            root_created = True
            push(root)
        elif name == 'MARK':
            push(marker)
        elif name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'SHORT_BINBYTES', 'BINBYTES', 'BINBYTES8'):
            if type(arg) not in (str, bytes):
                raise Refusal('bounded exact UTF8/byte literal')
            try:
                literal_bytes = arg.encode('utf8', 'strict') if type(arg) is str else arg
            except UnicodeError as error:
                raise Refusal('strict UTF8 literal required') from error
            if len(literal_bytes) > 4096:
                raise Refusal('bounded exact UTF8/byte literal')
            literal_total += len(literal_bytes)
            if literal_total > 8388608:
                raise Refusal('bounded total literals')
            push(Node('str' if type(arg) is str else 'bytes', arg, start, stop))
        elif name in ('BININT1', 'BININT2', 'BININT', 'LONG1', 'LONG4'):
            if type(arg) is not int or not -(2**63) <= arg < 2**63:
                raise Refusal('bounded primitive int64')
            push(Node('int', arg, start, stop))
        elif name in ('NONE', 'NEWFALSE', 'NEWTRUE'):
            push(Node('none' if name == 'NONE' else 'bool',
                      None if name == 'NONE' else name == 'NEWTRUE', start, stop))
        elif name in ('TUPLE', 'TUPLE1', 'TUPLE2', 'TUPLE3'):
            values = group() if name == 'TUPLE' else [pop() for _ in range(int(name[-1]))][::-1]
            push(Node('tuple', tuple(values), start, stop))
        elif name == 'STACK_GLOBAL':
            symbol, module = pop(), pop()
            if module.kind != 'str' or symbol.kind != 'str' or (module.value, symbol.value) not in SYMBOLS:
                raise Refusal('unknown serialized symbol; never import/resolve')
            push(Node('symbol', (module.value, symbol.value), start, stop))
        elif name == 'REDUCE':
            args, symbol = pop(), pop()
            if args.kind != 'tuple' or symbol.kind != 'symbol':
                raise Refusal('only exact inert scalar/dtype recipe')
            if symbol.value == ('numpy', 'dtype'):
                values = args.value
                if (len(values) != 3 or any(type(x) is not Node for x in values) or
                    [x.kind for x in values] != ['str', 'bool', 'bool'] or
                    values[0].value not in DTYPES or values[1].value is not False or values[2].value is not True):
                    raise Refusal('exact primitive numeric dtype constructor recipe')
                dtype = Node('dtype', values[0].value, symbol.start, stop)
                dtype_nodes.append(dtype)
                push(dtype)
            elif symbol.value in (('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')):
                if (len(args.value) != 2 or any(type(x) is not Node for x in args.value) or
                    args.value[0].kind != 'dtype' or args.value[1].kind != 'bytes' or args.value[0].state is None):
                    raise Refusal('built primitive dtype and immutable scalar bytes only')
                value = Node('numpy_scalar', (args.value[0], args.value[1], start), symbol.start, stop)
                native_value(value)  # Validate payload exactly before it can enter the mapping.
                scalar_nodes.append(value)
                push(value)
            else:
                raise Refusal('unknown inert recipe')
        elif name == 'BUILD':
            state = pop()
            if not stack or type(stack[-1]) is not Node or stack[-1].kind != 'dtype' or stack[-1].state is not None:
                raise Refusal('one explicit dtype BUILD only')
            values = state.value if state.kind == 'tuple' else ()
            if (any(type(x) is not Node for x in values) or
                [x.kind for x in values] != ['int', 'str', 'none', 'none', 'none', 'int', 'int', 'int'] or
                [x.value for x in values if x.kind != 'str'] != [3, None, None, None, -1, -1, 0]):
                raise Refusal('exact dtype version3 state')
            order = values[1].value
            if order not in ('<', '>') and not (order == '|' and stack[-1].value[1:] == '1'):
                raise Refusal('explicit byte order required')
            stack[-1].state = order
            stack[-1].stop = stop
        elif name in ('MEMOIZE', 'BINPUT', 'LONG_BINPUT'):
            slot = len(memo) if name == 'MEMOIZE' else arg
            if not stack or stack[-1] is marker or type(slot) is not int or not 0 <= slot < 200000 or slot in memo or len(memo) >= 200000:
                raise Refusal('exact bounded memo assignment')
            memo[slot] = stack[-1]
        elif name in ('BINGET', 'LONG_BINGET'):
            if type(arg) is not int or arg not in memo:
                raise Refusal('existing memo required')
            push(memo[arg])
            memo_reads.append(dict(offset=start, slot=arg))
        elif name == 'SETITEMS':
            values = group()
            if stack != [root]:
                raise Refusal('flat root mapping mutation only')
            commit(values)
        elif name == 'SETITEM':
            value, key = pop(), pop()
            if stack != [root]:
                raise Refusal('flat root mapping mutation only')
            commit([key, value])
        elif name == 'STOP':
            if not root_created or stack != [root] or frame_end not in (None, stop):
                raise Refusal('complete mapping/frame/stack required')
        else:
            raise Refusal('unsupported opcode: ' + name)
    if any(x.state is None for x in dtype_nodes):
        raise Refusal('all dtype states explicitly built')
    return dict(schema='Mora3.PassiveNumericMapping/v1', source_bytes=len(raw),
                source_sha256=hashlib.sha256(raw).hexdigest(), protocol=raw[1], pairs=pairs,
                pair_count=len(pairs), numpy_scalar_recipe_count=len(scalar_nodes),
                dtype_recipes=[dict(token=x.value, byte_order=x.state, start=x.start, stop=x.stop) for x in dtype_nodes],
                opcode_counts=dict(counts), frames=frames, memo_reads=memo_reads,
                imports_or_execution_of_serialized_symbols=False, source_native_semantic_authority=None,
                task_admission=False, model_use=False, full_dataset_ready=False, training=False)
