"""Isolated QuickGO finite worker client: certified response replay, shared limits."""
import json,pathlib,time
from data_sources_goa_acquire_v2 import EfficientSkillClient
from data_sources_goa_acquire import write_new,now
from data_sources_quickgo_resume_v2 import request_key,certify_receipt,load_cached

class CachedSkillClient(EfficientSkillClient):
    def __init__(self,root,plan,limits,category):
        self.shared_limits=limits;self.category=category;self.shared_accounted=0;self.active_reserved=0;self.cache={};self.payload_refs=[];self.cached_requests=0
        super().__init__(root,plan)
        execute=self.skill.execute;outer=self
        def gated_execute(query):
            reserve=(64 if category=='annotation' else 16)*1024**2;limits.reserve_bytes(category,reserve);outer.active_reserved=reserve
            try:
                limits.wait_for_start();limits.ensure_open();result=execute(query);outer.raw_bytes()
                p=pathlib.Path(query['raw_output_path'])
                if p.exists() and p.stat().st_size>reserve:limits.block('oversized source response');raise ValueError('response exceeded reviewed envelope; original retained')
                if outer.last_http.get('status')==429:limits.block('HTTP429; preserve response and defer entire finite dispatcher')
                return result
            finally:
                if outer.active_reserved:limits.release_bytes(category,outer.active_reserved);outer.active_reserved=0
        self.skill.execute=gated_execute
    def raw_bytes(self):
        total=super().raw_bytes();delta=total-self.shared_accounted
        if delta:
            credit=min(delta,self.active_reserved);self.shared_limits.consume(self.category,delta,credit);self.active_reserved-=credit;self.shared_accounted=total
        return total
    def start_batch(self,directory,certificates=()):
        self.batch_dir=pathlib.Path(directory);self.payload_refs=[];self.cache={}
        for cert in certificates:
            key=cert['request_key']
            if key in self.cache and self.cache[key]['sha256']!=cert['sha256']:raise ValueError('ambiguous successful cache source responses')
            self.cache[key]=cert
    def request(self,path,params,name,accept='application/json'):
        key=request_key(path,params,accept);cert=self.cache.get(key) if path not in ['annotation/about','ontology/go/about'] else None
        if cert is not None:
            result=load_cached(cert,path,params,accept);reference=self.batch_dir/(name+'.reused-reference.json');write_new(reference,dict(schema='mora3.immutable_QuickGO_response_replay.v1',replayed_utc=now(),original_certificate=cert,source_bytes_unchanged=True,new_HTTP_request=False));self.receipts.append(str(reference));self.payload_refs.append(cert);self.cached_requests+=1;self.raw_bytes();return result
        result=super().request(path,params,name,accept);cert=certify_receipt(self.receipts[-1]);self.payload_refs.append(cert);return result
