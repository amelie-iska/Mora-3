"""Streaming source FASTA parsing with native sequence and full-axis retention."""
import hashlib

IUPAC_NUCLEIC=frozenset('ACGTURYSWKMBDHVN')

def fasta_records(stream):
    header=None;parts=[]
    for line in stream:
        text=line.rstrip('\r\n')
        if not text:continue
        if text.startswith('>'):
            if not text[1:].split():raise ValueError('missing source FASTA identifier')
            if header is not None:yield header,''.join(parts)
            header=text[1:];parts=[]
        else:
            if header is None:raise ValueError('source sequence before FASTA header')
            parts.append(text)
    if header is not None:yield header,''.join(parts)

def sequence_qc(header,sequence):
    normalized=sequence.upper()
    return dict(source_identifier=header.split()[0],native_length=len(sequence),native_sequence_sha256=hashlib.sha256(sequence.encode('utf-8')).hexdigest(),uppercase_sequence_sha256=hashlib.sha256(normalized.encode('utf-8')).hexdigest(),sequence_alphabet_pass=bool(sequence) and set(normalized)<=IUPAC_NUCLEIC,native_T_count=normalized.count('T'),native_U_count=normalized.count('U'),ambiguity_symbol_count=sum(normalized.count(x) for x in IUPAC_NUCLEIC-frozenset('ACGTU')),source_sequence_preserved=True,source_letter_axis='all released letters0..native_length; no truncation or T/U conversion')
