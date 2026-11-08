"""Validate complete native tool membership; no claim of exhaustive homology."""
def parse_partition(text,expected):
    expected=set(expected);mapping={}
    for line in text.splitlines():
        fields=line.split('\t')
        if len(fields)!=2 or not all(fields):raise ValueError('two explicit native IDs required')
        representative,member=fields
        if representative not in expected or member not in expected:raise ValueError('unexpected native sequence ID')
        if member in mapping:raise ValueError('duplicate native member')
        mapping[member]=representative
    if set(mapping)!=expected:raise ValueError('missing native sequence IDs')
    if any(mapping.get(rep)!=rep for rep in mapping.values()):raise ValueError('representative must be its own member')
    return mapping
