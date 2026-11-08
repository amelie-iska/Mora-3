from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_publish_full_rnacentral_handoff import verify_publication_counts

def counts():
 old=dict(serialized_tasks=26322315,total_shard_bytes=10673072693,split_counts={'train':26302187,'validation':10076,'test':10052},serialized_distinct_record_counts_by_source={'RNAcentral-reference-priors':445486})
 comp=dict(counts={'new_task_views_serialized':54732469},task_shard_bytes=100)
 new=dict(serialized_tasks=81054784,total_shard_bytes=10673072793,split_counts={'train':81034656,'validation':10076,'test':10052},serialized_distinct_record_counts_by_source={'RNAcentral-reference-priors':55177955},shards=[1,2])
 proof=dict(actual_new_old_task_ID_intersections=0,known_newtrain_oldheldout_group_intersections=0,preserved_Rfam_and_old_pilot_exact_tasks_reused=6273790)
 return new,old,comp,proof


def test_source_reference_reuse_is_nonadditive_and_old_pilot_enters_once():
 v=verify_publication_counts(*counts())
 assert v['new_RNA_reference_prior_views']==54732469
 assert v['all_RNA_family_views']==55177955
 assert v['preserved_Rfam_reuse_views']==5828304
 assert v['actual_model_tokens'] is None and v['new_independent_experiments']==0


def test_reusing_every_source_sequence_as_new_corpus_is_rejected():
 new,old,comp,proof=counts();new['serialized_tasks']=old['serialized_tasks']+61006259
 with pytest.raises(ValueError):verify_publication_counts(new,old,comp,proof)


def test_train_only_prior_build_cannot_invent_heldout_or_accept_group_leakage():
 a=counts();a[0]['split_counts']['test']+=1
 with pytest.raises(ValueError):verify_publication_counts(*a)
 a=counts();a[3]['known_newtrain_oldheldout_group_intersections']=1
 with pytest.raises(ValueError):verify_publication_counts(*a)
