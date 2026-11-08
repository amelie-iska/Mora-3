"""Finite new-build guards; no cleanup, overwrite, or old-index mutation."""
from pathlib import Path
import time,json,hashlib,shutil
from data_intake_corpus import canonical

def immutable_json(path,value):
    Path(path).open('x').write(json.dumps(value,indent=2)+'\n')

def task_identifier(family,source_record_id,direction):
    return hashlib.sha256(canonical([family,source_record_id,direction])).hexdigest()

class BuildGuard:
    def __init__(self,root,*,census_cap,total_cap,deadline_epoch,min_free=200*2**30):
        self.root=Path(root);self.census_cap=census_cap;self.total_cap=total_cap;self.deadline=deadline_epoch;self.min_free=min_free
    def check(self,phase,*,reserve=0):
        if phase not in ['census','materialized'] or type(reserve) is not int or reserve<0:raise ValueError('explicit phase and nonnegative byte reservation required')
        if time.time()>=self.deadline:raise RuntimeError('finite whole-build deadline reached; preserve checkpoints')
        total=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file());materialized=self.root/'materialized';second=sum(p.stat().st_size for p in materialized.rglob('*') if p.is_file()) if materialized.exists() else 0;first=total-second
        if total+reserve>self.total_cap:raise RuntimeError('total derived cap prewrite reservation failed')
        if phase=='census' and first+reserve>self.census_cap:raise RuntimeError('census phase cap prewrite reservation failed')
        if phase=='materialized' and second+reserve>self.total_cap-self.census_cap:raise RuntimeError('materialized phase cap prewrite reservation failed')
        if shutil.disk_usage(self.root).free<self.min_free+reserve:raise RuntimeError('free disk reserve prewrite reservation failed')
        return dict(all_bytes=total,census_bytes=first,materialized_bytes=second,reserved_next_write_bytes=reserve)
