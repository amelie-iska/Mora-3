"""Isolated GOA worker with strict full GPAD validation for null source GOshortcodes."""
import argparse
import data_sources_goa_acquire as base
from data_sources_goa_acquire_v2 import EfficientSkillClient
from data_sources_goa_v2 import acquire_batch
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--connection-lease',required=True);args=parser.parse_args()
    base.SkillClient=EfficientSkillClient;base.acquire_batch=acquire_batch;base.run(args.plan,args.connection_lease)
