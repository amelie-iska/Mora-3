"""Bounded original-storage gathers; no normalization, row admission or context.

Explicit larger application-I/O budget is confined to this three-row interface;
ordinary NativeMmapArray contiguous-span limits remain unchanged.
"""
import math
import numpy as np
from . import xpert_mmap as core

MAX_GATHER_PAYLOAD_BYTES = 3 * 1024**3


def plan_original_ordinals(array, rows):
    d = array.descriptor()
    if (len(d['native_shape']) != 1 or d['native_dtype'] not in ('<f4','<i4')
            or type(rows) is not tuple or not 1 <= len(rows) <= 3):
        raise ValueError('one-dimensional native arrays and at most three rows required')
    total = math.prod(d['native_shape'])
    for offsets in rows:
        if (type(offsets) is not np.ndarray or offsets.dtype.str != '<u4'
                or offsets.ndim != 1 or len(offsets) > 33694
                or (len(offsets) and (offsets[-1] >= total
                    or np.any(offsets[1:] <= offsets[:-1])))):
            raise ValueError('bounded strictly ordered original uint32 ordinals required')
    selected = [(i,c) for i,c in enumerate(d['chunks']) if any(
        np.searchsorted(o,c['axis0_stop']) > np.searchsorted(o,c['axis0_start']) for o in rows)]
    return selected, sum(c['numeric_bytes']+c['validity_bytes'] for _,c in selected)


def gather_original_ordinals(array, rows, *, max_payload_bytes):
    """Verify union chunks once and copy only selected original occurrences.

    Budget counts actual numeric plus packed-validity application bytes, not OS
    disk traffic. Returned byte pairs retain caller row order and native bits.
    """
    if (type(max_payload_bytes) is not int
            or not 1 <= max_payload_bytes <= MAX_GATHER_PAYLOAD_BYTES):
        raise ValueError('explicit bounded gather payload budget required')
    selected, planned = plan_original_ordinals(array, rows)
    if planned > max_payload_bytes:
        raise ValueError('union source chunks exceed explicit gather IO budget')
    # Snapshot selectors so they cannot change while mapping immutable sources.
    rows = tuple(o.copy() for o in rows)
    d = array.descriptor(); item = np.dtype(d['native_dtype']).itemsize
    values = [bytearray(len(o)*item) for o in rows]
    validity = [bytearray(len(o)) for o in rows]
    numeric_bytes = validity_bytes = mapped = 0
    with array._opened(d) as (f,g,headers):
        for _,c in selected:
            body,n = core._mapped_bytes(f.fileno(),c['numeric_offset'],c['numeric_bytes'])
            bits,v = core._mapped_bytes(g.fileno(),c['validity_offset'],c['validity_bytes'])
            numeric_bytes += len(body); validity_bytes += len(bits); mapped += n+v
            if core._sha(body) != c['numeric_sha256'] or core._sha(bits) != c['validity_sha256']:
                raise ValueError('consumed numeric/validity chunk integrity mismatch')
            a = np.frombuffer(body,dtype=d['native_dtype']); finite = np.isfinite(a)
            if len(a) != c['native_elements'] or np.packbits(finite,bitorder='little').tobytes() != bits:
                raise ValueError('native numeric validity/padding differs from exact per-source chunk')
            for offsets,out,mask in zip(rows,values,validity):
                left,right = map(int,np.searchsorted(offsets,[c['axis0_start'],c['axis0_stop']]))
                relative = offsets[left:right].astype(np.int64)-c['axis0_start']
                out[left*item:right*item] = a[relative].tobytes()
                mask[left:right] = finite[relative].tobytes()
    audit = dict(descriptor_sha256=core._sha(array.descriptor_bytes),
        native_array_file_sha256_inherited=d['sha256'],
        native_value_sha256_inherited=d['native_value_sha256'],
        selected_chunk_indices=[i for i,_ in selected],header_bytes_read=headers,
        numeric_payload_bytes=numeric_bytes,validity_payload_bytes=validity_bytes,
        mapped_page_bytes=mapped,output_value_bytes=sum(map(len,values)),
        output_validity_bytes=sum(map(len,validity)),whole_file_hash_recomputed=False,
        max_payload_bytes=max_payload_bytes,task_admitted=False,
        scope='application bytes/pages mapped; OS physical disk/cache IO not measured')
    return tuple((bytes(v),bytes(m)) for v,m in zip(values,validity)), audit
