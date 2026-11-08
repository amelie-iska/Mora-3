"""Native STRING output semantics; sequence qualification and admission stay separate."""
import hashlib
import re
import pyarrow as pa
import pyarrow.compute as pc

PHYSICAL_COLUMNS=['protein1','protein2','homology','experiments','experiments_transferred','database','database_transferred','textmining','textmining_transferred','combined_score']
PHYSICAL_SCHEMA=pa.schema([(x,pa.string() if i<2 else pa.uint16()) for i,x in enumerate(PHYSICAL_COLUMNS)])
PROVIDER=re.compile(r'([1-9][0-9]*)\.(\S+)')
PROTEIN_ALPHABET=frozenset('ACDEFGHIKLMNPQRSTVWYBXZJUO*')

def provider_id(identifier):
    m=PROVIDER.fullmatch(identifier) if isinstance(identifier,str) else None
    if m is None:raise ValueError('invalid native STRING provider namespace')
    return int(m[1]),m[2]

def count_true(values):return int(pc.sum(pc.cast(values,pa.int64())).as_py() or 0)

def physical_batch_qc(batch):
    if not batch.schema.equals(PHYSICAL_SCHEMA,check_metadata=False):raise ValueError('unexpected native physical schema')
    if any(c.null_count for c in batch.columns):raise ValueError('null native physical field')
    for name in ['protein1','protein2']:
        if not pc.all(pc.match_substring_regex(batch[name],r'^[1-9][0-9]*\.\S+$')).as_py():raise ValueError('invalid native STRING provider namespace')
    nonzero={}
    for name in PHYSICAL_COLUMNS[2:]:
        if pc.any(pc.greater(batch[name],1000)).as_py():raise ValueError('native score outside0..1000')
        nonzero[name]=count_true(pc.greater(batch[name],0))
    tax1=pc.struct_field(pc.extract_regex(batch['protein1'],r'^(?P<taxon>[1-9][0-9]*)\.'),'taxon')
    tax2=pc.struct_field(pc.extract_regex(batch['protein2'],r'^(?P<taxon>[1-9][0-9]*)\.'),'taxon')
    return dict(native_rows=batch.num_rows,self_pair_rows=count_true(pc.equal(batch['protein1'],batch['protein2'])),cross_taxon_rows=count_true(pc.not_equal(tax1,tax2)),channel_nonzero=nonzero)

def provider_sequence_qc(header,sequence,qualified_by_exact_sequence_sha256):
    identifier=header.split()[0] if header.split() else '';taxon,_=provider_id(identifier)
    sha=hashlib.sha256(sequence.encode('utf-8')).hexdigest();passed=bool(sequence) and set(sequence.upper())<=PROTEIN_ALPHABET
    return dict(provider_id=identifier,taxon_id=taxon,native_header=header,native_length=len(sequence),native_sequence_sha256=sha,uppercase_sequence_sha256=hashlib.sha256(sequence.upper().encode('utf-8')).hexdigest(),sequence_alphabet_pass=passed,has_stop_symbol='*' in sequence,qualified_exact_sequence_accessions=sorted(qualified_by_exact_sequence_sha256.get(sha,[])) if passed else [],task_admitted=False)

def alias_record(line):
    fields=line.rstrip('\r\n').split('\t')
    if len(fields)!=3 or not fields[1]:raise ValueError('invalid native alias row')
    provider_id(fields[0])
    return dict(provider_id=fields[0],alias=fields[1],native_source=fields[2],source_tokens=fields[2].split())
