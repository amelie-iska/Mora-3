"""Isolated finite GOA worker using per-job incremental immutable-file accounting."""
import argparse
import json
import time
import data_sources_goa_acquire as base

class EfficientSkillClient(base.SkillClient):
    def __init__(self,root,plan):
        self.accounted_bytes=0;self.accounted_paths=set()
        super().__init__(root,plan)
    def raw_bytes(self):
        if self.batch_dir is not None:
            for path in self.batch_dir.iterdir():
                if path.is_file() and path not in self.accounted_paths:
                    self.accounted_bytes+=path.stat().st_size;self.accounted_paths.add(path)
        return self.accounted_bytes
    def request(self,path,params,name,accept='application/json'):
        started=time.monotonic()
        try:return super().request(path,params,name,accept)
        finally:
            data=dict(updated_utc=base.now(),pid=base.os.getpid(),job_id=self.plan['job_id'],last_endpoint=path,last_request_name=name,last_request_seconds=time.monotonic()-started,requests=self.request_count,accounted_own_job_raw_bytes=self.raw_bytes(),own_job_raw_cap_bytes=self.plan['raw_cap_bytes'],current_batch_dir=str(self.batch_dir),last_receipt=self.receipts[-1] if self.receipts else None,source_connections=1)
            (self.root/('live-request-progress-'+self.plan['job_id']+'.json')).write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--connection-lease',required=True);args=parser.parse_args()
    base.SkillClient=EfficientSkillClient;base.run(args.plan,args.connection_lease)
