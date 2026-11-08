"""Source-native UniRef representative and cluster metadata quality checks."""
import collections
import hashlib
import re
import xxhash

AMINO_ACIDS=frozenset('ACDEFGHIKLMNPQRSTVWYBXZJUO')

def qc_record(raw,qualified_sequence_hashes):
    seq=raw['sequence'];seqhash=hashlib.sha256(seq.encode('ascii')).hexdigest() if seq.isascii() else hashlib.sha256(seq.encode('utf-8')).hexdigest()
    alphabet_ok=bool(seq) and set(seq)<=AMINO_ACIDS;length_ok=len(seq)==raw['sequence_length']
    xxh_ok=xxhash.xxh128_hexdigest(seq.encode('utf-8')).lower()==str(raw['sequence_xxh128']).lower()
    members=raw['member_ids'];rep=raw['rep_member_id'];seed=raw['seed_id'];accessions=raw['rep_accessions']
    flags=dict(member_count_matches_array=isinstance(raw['member_count'],int) and raw['member_count']==len(members),member_ids_unique=len(set(members))==len(members),seed_id_present=seed in members,representative_id_present=rep in members,representative_is_seed_flag_consistent=bool(raw['rep_is_seed'])==(rep==seed),cluster_id_has_namespace=isinstance(raw['id'],str) and raw['id'].startswith('UniRef50_'),representative_accessions_unique=len(set(accessions))==len(accessions))
    go_ids=[x for k in ['go_mf','go_bp','go_cc'] for x in raw[k]];go_syntax=all(isinstance(x,str) and re.fullmatch(r'GO:[0-9]{7}',x) for x in go_ids)
    qualified=[a for a in accessions if a in qualified_sequence_hashes];matches=[a for a in qualified if qualified_sequence_hashes[a]==seqhash];mismatches=[a for a in qualified if qualified_sequence_hashes[a]!=seqhash]
    return dict(cluster_id=raw['id'],seed_id=seed,representative_member_id=rep,representative_member_id_type=raw['rep_member_id_type'],representative_accessions=list(accessions),representative_is_seed=raw['rep_is_seed'],native_sequence_length=len(seq),native_sequence_sha256=seqhash,sequence_length_match=length_ok,sequence_alphabet_pass=alphabet_ok,sequence_xxh128_match=xxh_ok,sequence_integrity_pass=alphabet_ok and length_ok and xxh_ok,sequence_crc64_format_pass=bool(re.fullmatch(r'[0-9A-Fa-f]{16}',str(raw['sequence_crc64']))),sequence_crc64_recomputed=False,cluster_metadata_integrity_pass=all(flags.values()),cluster_flags=flags,source_GO_ID_syntax_pass=go_syntax,source_GO_metadata_count=len(go_ids),source_GO_evidence_available=False,qualified_representative_exact_sequence_accessions=matches,qualified_representative_sequence_mismatch_accessions=mismatches,member_annotation_transfer_allowed=False,new_uniprot_admissions=0,source_UniRef_biological_release_not_established=True)
