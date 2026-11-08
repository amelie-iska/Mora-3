import pathlib,json,hashlib,time,requests
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-protein-function-context-v1';B.mkdir(exist_ok=False);(B/'raw').mkdir();(B/'research').mkdir();(B/'logs').mkdir()
s=requests.Session();s.trust_env=False
fields='accession,annotation_score,reviewed,sequence,protein_name,gene_primary,cc_function,cc_catalytic_activity,go_id,xref_geneid,organism_id';query='(reviewed:true) AND ((annotation_score:4) OR (annotation_score:5))'
r=s.get('https://rest.uniprot.org/uniprotkb/search',params=dict(query=query,format='tsv',fields=fields,size=1),timeout=(20,60));data=r.content
p=B/'research/official-field-probe.tsv';p.write_bytes(data)
receipt=dict(schema='mora3.public_uniprot_field_probe.v1',http=r.status_code,url=r.url,release=r.headers.get('x-uniprot-release'),release_date=r.headers.get('x-uniprot-release-date'),total_results=r.headers.get('x-total-results'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),headers=data.decode(errors='replace').splitlines()[0],fields=fields,query=query,license='CC-BY-4.0 UniProt data;source license review already retained',maximum_next_download_bytes=268435456,credentials=False,environment_changes=False)
with (B/'research/official-field-probe.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
