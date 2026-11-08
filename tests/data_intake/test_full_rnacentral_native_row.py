from copy import deepcopy
from pathlib import Path
import hashlib,sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_census import validate_native_row


def row(seq='aCGTUNRYS'):
 return dict(native_sequence=seq,native_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),uppercase_sequence_sha256=hashlib.sha256(seq.upper().encode()).hexdigest(),native_length=len(seq),source_identifier='URS001',native_header='URS001 native source description',source_record_index=7,source_payload='immutable-native.fasta.gz',sequence_alphabet_pass=True,native_T_count=seq.upper().count('T'),native_U_count=seq.upper().count('U'))


def test_native_T_U_case_ambiguity_are_preserved():
 r=row();seq,native,upper,n,valid=validate_native_row(r,original_FASTA='immutable-native.fasta.gz',expected_record_index=7)
 assert seq==r['native_sequence'] and valid and n==9
 assert native!=upper


@pytest.mark.parametrize('field,value',[('native_length',8),('native_sequence_sha256','wrong'),('uppercase_sequence_sha256','wrong')])
def test_corrupt_full_axis_or_source_identity_does_not_pass(field,value):
 r=row();r[field]=value
 with pytest.raises(ValueError,match='hash/full-axis'):validate_native_row(r,original_FASTA='immutable-native.fasta.gz',expected_record_index=7)


@pytest.mark.parametrize('field,value',[('source_identifier','URS_ALIAS'),('source_payload','different.fasta.gz'),('source_record_index',8)])
def test_source_alias_dropped_row_or_wrong_original_cannot_be_repaired(field,value):
 r=row();r[field]=value
 with pytest.raises(ValueError,match='provenance'):validate_native_row(r,original_FASTA='immutable-native.fasta.gz',expected_record_index=7)


def test_nonstandard_native_symbols_are_retained_for_quarantine():
 r=row('ACX123');r['sequence_alphabet_pass']=False
 out=validate_native_row(r,original_FASTA='immutable-native.fasta.gz',expected_record_index=7)
 assert out[0]=='ACX123' and out[-1] is False


def test_source_T_U_precision_is_not_silently_harmonized():
 r=row();r['native_U_count']=0
 with pytest.raises(ValueError,match='T-U precision'):validate_native_row(r,original_FASTA='immutable-native.fasta.gz',expected_record_index=7)
