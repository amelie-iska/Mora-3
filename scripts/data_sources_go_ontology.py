"""Native GO term metadata with explicit ontology version and current-child edges."""
import hashlib
import json
import math
import re

GO_ID=re.compile(r'GO:[0-9]{7}')

def check_version(response,timestamp,version):
    if response.get('go',{}).get('timestamp')!=timestamp or response.get('go',{}).get('version')!=version:
        raise ValueError('ontology snapshot/profile drift; annotation date is not an ontologyversion')

def validate_page(response,index,expected_hits,page_size=25):
    info=response.get('pageInfo',{});rows=response.get('results')
    if response.get('numberOfHits')!=expected_hits or info.get('current')!=index or info.get('total')!=math.ceil(expected_hits/page_size) or info.get('resultsPerPage')!=page_size:raise ValueError('ontology count/page metadata mismatch')
    expected=min(page_size,max(0,expected_hits-(index-1)*page_size))
    if not isinstance(rows,list) or len(rows)!=expected:raise ValueError('incomplete ontology page')
    ids=[r.get('id') for r in rows]
    if len(set(ids))!=len(ids) or any(not isinstance(x,str) or not GO_ID.fullmatch(x) for x in ids):raise ValueError('invalid or duplicate ontology IDs')
    return len(rows)

def project_term(term,source_path,index,ontology_version):
    if not GO_ID.fullmatch(str(term.get('id',''))) or not isinstance(term.get('name'),str) or not term['name'] or not isinstance(term.get('isObsolete'),bool):raise ValueError('incomplete native GO term')
    native=json.dumps(term,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    node=dict(go_id=term['id'],name=term['name'],aspect=term.get('aspect'),is_obsolete=term['isObsolete'],ontology_version=ontology_version,source_raw_path=source_path,source_result_index=index,native_term_sha256=hashlib.sha256(native.encode('utf-8')).hexdigest(),native_term_json=native)
    edges=[]
    for child in term.get('children') or []:
        if not isinstance(child.get('id'),str) or not child['id'] or not isinstance(child.get('relation'),str) or not child['relation']:raise ValueError('incomplete native currentchild relation')
        edges.append(dict(child_id=child['id'],relation=child['relation'],parent_id=term['id'],ontology_version=ontology_version,source_raw_path=source_path,source_result_index=index))
    return node,edges
