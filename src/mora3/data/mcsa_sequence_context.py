"""Candidate MCSA/current-UniProt position associations, without version imputation.

Agreement at annotated positions is necessary evidence, not proof that the
original MCSA protein sequence/version equals a current UniProt sequence. Chain
and author insertion codes remain separate source contexts. No task admission,
experimental truth, negative labels, or training eligibility is granted here.
"""
import copy
import hashlib
import re

AA3=dict(zip(('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','SEC','PYL'),
             'ARNDCQEGHILKMFPSTWYVUO'))
AA1=frozenset(AA3.values())


def _sha(value):
    return type(value)is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _ref(value):
    return (type(value)is dict and type(value.get('path'))is str and bool(value['path'])
            and type(value.get('bytes'))is int and value['bytes']>0 and _sha(value.get('sha256'))
            and type(value.get('row_index'))is int and value['row_index']>=0)


def check_residue_mapping(mapping,sequence,reference_accession):
    if type(mapping)is not dict or type(sequence)is not str or not sequence or type(reference_accession)is not str or not reference_accession:
        raise ValueError('exact source mapping,current sequence and reference accession required')
    result=dict(native_mapping=copy.deepcopy(mapping),source_position=mapping.get('resid'),
        source_residue_code=mapping.get('code'),current_sequence_residue=None,
        source_position_base=1,original_MCSA_sequence_version=None,
        full_source_sequence_identity_proven=False,source_chain_or_author_numbering_converted=False)
    if mapping.get('uniprot_id')!=reference_accession:
        result['disposition']='DIFFERENT_SOURCE_ACCESSION'
    elif mapping.get('is_reference')is not True:
        result['disposition']='ALTERNATIVE_SOURCE_MAPPING_NOT_SELECTED_REFERENCE'
    elif type(mapping.get('resid'))is not int or mapping['resid']<1:
        result['disposition']='SOURCE_POSITION_UNQUALIFIED'
    elif mapping['resid']>len(sequence):
        result['disposition']='SOURCE_POSITION_OUTSIDE_CURRENT_SEQUENCE'
    else:
        code=mapping.get('code')
        letter=(code if type(code)is str and code in AA1 else AA3.get(code.upper()) if type(code)is str else None)
        if letter is None:
            result['disposition']='SOURCE_RESIDUE_CODE_UNQUALIFIED'
        else:
            actual=sequence[mapping['resid']-1]
            result['current_sequence_residue']=actual
            result['source_unambiguous_one_letter_code']=letter
            result['disposition']=('POSITION_CODE_AGREES_CURRENT_SEQUENCE' if actual==letter
                                   else 'SOURCE_CODE_CURRENT_SEQUENCE_DISAGREEMENT')
    return result


def sequence_context(entry,sequence,provenance):
    if (type(entry)is not dict or type(entry.get('mcsa_id'))is not int or entry['mcsa_id']<=0
        or type(sequence)is not str or re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+',sequence)is None
        or type(provenance)is not dict):
        raise ValueError('source entry and complete current canonical protein sequence required')
    accession=entry.get('reference_uniprot_id')
    if (type(accession)is not str or not accession or provenance.get('primary_accession')!=accession
        or provenance.get('canonical_sequence_sha256')!=hashlib.sha256(sequence.encode()).hexdigest()
        or type(provenance.get('source_release'))is not str or re.fullmatch(r'[0-9]{4}_[0-9]{2}',provenance['source_release'])is None
        or any(type(provenance.get(k))is not int or provenance[k]<=0 for k in ('sequence_version','entry_version'))
        or not _ref(provenance.get('raw_sequence_source')) or not _ref(provenance.get('original_entry_ref'))):
        raise ValueError('exact current accession/sequence/hash/version/source binding required')
    native_residues=entry.get('residues')
    if native_residues is not None and type(native_residues)is not list:
        raise ValueError('native residue annotation list required')
    protein=entry.get('protein')
    if protein is not None and type(protein)is not dict:
        raise ValueError('source protein context object required')
    source_sequence_rows=(protein or {}).get('sequences') or []
    if type(source_sequence_rows)is not list or any(type(x)is not dict for x in source_sequence_rows):
        raise ValueError('source protein identifier/sequence context list required')
    full_sequences=[x['sequence'] for x in source_sequence_rows
                    if x.get('uniprot_id')==accession and type(x.get('sequence'))is str and x['sequence']]
    source_full_available=bool(full_sequences)
    same_full_sequence=(len(set(full_sequences))==1 and full_sequences[0]==sequence) if source_full_available else None
    full_identity_proven=same_full_sequence is True
    targets=[]
    for ordinal,residue in enumerate(native_residues or []):
        if type(residue)is not dict or type(residue.get('residue_sequences'))is not list:
            raise ValueError('native residue sequence-mapping list required')
        checks=[dict(source_mapping_ordinal=i,**check_residue_mapping(mapping,sequence,accession))
                for i,mapping in enumerate(residue['residue_sequences'])]
        for check in checks:
            check['full_source_sequence_identity_proven']=full_identity_proven
        targets.append(dict(source_residue_ordinal=ordinal,native_annotation=copy.deepcopy(residue),mapping_checks=checks))
    return dict(schema='Mora3.MCSACurrentSequenceCandidateContext/v1',
        conditions=dict(canonical_sequence=sequence),
        targets=dict(source_residue_annotations=targets),
        audit=dict(source_mcsa_id=entry['mcsa_id'],reference_uniprot_id=accession,
            source_is_reference_uniprot_id=entry.get('is_reference_uniprot_id'),
            current_uniprot_provenance=copy.deepcopy(provenance),
            source_residue_list_present='residues'in entry,source_residue_list_null=native_residues is None,
            original_MCSA_full_sequence_available=source_full_available,original_MCSA_sequence_version=None,
            source_original_full_sequence_literal_count=len(full_sequences),
            source_original_full_sequence_literals_conflict=len(set(full_sequences))>1,
            source_original_vs_current_full_sequence_equality=same_full_sequence,
            full_source_sequence_identity_proven=full_identity_proven,
            evidence_role='source_curated_catalytic_annotation_with_current_position_consistency_check'),
        leakage_keys=[['protein_accession',accession],['protein_sequence_sha256',provenance['canonical_sequence_sha256']],
                      ['MCSA_source_entry',str(entry['mcsa_id'])]],
        source_chain_and_author_insertion_codes_preserved=True,
        unlisted_residues_are_negative=False,experimental_catalytic_validation=False,
        source_version_qualification_complete=False,global_group_or_heldout_qualification=False,
        task_admission=False,model_use=False,full_dataset_ready=False,training=False)
