"""Public-image ingestion contracts; generated fixtures are tests only."""
from pathlib import Path
import hashlib
import importlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile
import numpy as np
from PIL import Image
import pytest


def api():
    assert Path('src/mora3/data/imaging.py').is_file(), 'Image acquisition/adapter missing'
    return importlib.import_module('mora3.data.imaging')


@pytest.fixture
def server():
    body=bytes(range(256))*128
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            start=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
            if self.path=='/ignore':start=0
            self.send_response(206 if start else 200)
            if start:self.send_header('Content-Range',f'bytes {start}-{len(body)-1}/{len(body)}')
            self.send_header('Content-Length',str(len(body)-start));self.end_headers();self.wfile.write(body[start:])
    s=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=s.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{s.server_port}',body
    s.shutdown();thread.join();s.server_close()


def test_download_and_resume_are_byte_exact(server,tmp_path):
    base,body=server;p=tmp_path/'asset.bin'
    p.with_suffix('.bin.part').write_bytes(body[:123])
    receipt=api().download_asset(base+'/data',p,max_bytes=len(body),expected_bytes=len(body),expected_sha256=hashlib.sha256(body).hexdigest(),resume=True,min_free_gib=0)
    assert p.read_bytes()==body
    assert receipt['resumed_from_bytes']==123
    assert receipt['sha256']==hashlib.sha256(body).hexdigest()


def test_existing_output_and_nonresumable_partial_are_preserved(server,tmp_path):
    base,body=server;p=tmp_path/'asset.bin';p.write_bytes(b'protected')
    with pytest.raises(FileExistsError):api().download_asset(base+'/data',p,max_bytes=len(body),min_free_gib=0)
    assert p.read_bytes()==b'protected'
    other=tmp_path/'other.bin';part=other.with_suffix('.bin.part');part.write_bytes(body[:123])
    with pytest.raises(ValueError,match='Range'):api().download_asset(base+'/ignore',other,max_bytes=len(body),resume=True,min_free_gib=0)
    assert part.read_bytes()==body[:123]


def test_budget_and_checksum_errors_do_not_create_completed_assets(server,tmp_path):
    base,body=server
    with pytest.raises(ValueError):api().download_asset(base+'/data',tmp_path/'capped',max_bytes=100,min_free_gib=0)
    with pytest.raises(ValueError,match='checksum'):api().download_asset(base+'/data',tmp_path/'wrong',max_bytes=len(body),expected_sha256='0'*64,min_free_gib=0)
    assert not (tmp_path/'capped').exists() and not (tmp_path/'wrong').exists()


def test_archive_path_traversal_is_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../outside.png',b'bad')
    with pytest.raises(ValueError,match='archive'):api().archive_members(p)
    assert not (tmp_path.parent/'outside.png').exists()


def pair(mask_values=None):
    image=np.arange(64,dtype=np.uint8).reshape(8,8)
    mask=np.zeros((8,8),dtype=np.uint16);mask[1:3,1:4]=7;mask[4:7,4:6]=20
    if mask_values is not None:mask=mask_values
    im=io.BytesIO();ma=io.BytesIO();Image.fromarray(image).save(im,format='PNG');Image.fromarray(mask).save(ma,format='PNG')
    return im.getvalue(),ma.getvalue()


def test_image_mask_geometry_and_target_exclusion():
    m=api();image,mask=pair()
    a=m.image_mask_record(image,mask,dataset='fixture',record_id='r',split='test',license_id='test-only')
    assert a['cell_count']==2 and a['width']==8 and a['height']==8
    assert sorted(x['area_px'] for x in a['instances'])==[6,6]
    assert a['physical_pixel_size'] is None
    assert a['training_eligible'] is False
    new=np.ones((8,8),dtype=np.uint16)
    b=m.image_mask_record(image,pair(new)[1],dataset='fixture',record_id='r',split='test',license_id='test-only')
    assert m.project_image_task(a)['inputs']==m.project_image_task(b)['inputs']
    assert m.project_image_task(a)['targets']!=m.project_image_task(b)['targets']
    assert 'instances' not in m.project_image_task(a)['inputs']
    assert 'mask_sha256' not in m.project_image_task(a)['inputs']


def test_misaligned_image_and_mask_fail():
    image,_=pair();mask=io.BytesIO();Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(mask,format='PNG')
    with pytest.raises(ValueError,match='dimensions'):api().image_mask_record(image,mask.getvalue(),dataset='fixture',record_id='r',split='train',license_id='test-only')


def test_registry_requires_explicit_roles_rights_and_unique_sources():
    source=dict(id='a',name='dataset',role='train_and_evaluate',license_status='verified',source_url='https://example.org',status='registered',overlap_group='a')
    api().validate_catalog({'datasets':[source]})
    with pytest.raises(ValueError):api().validate_catalog({'datasets':[source,source]})
    bad=dict(source,role='all_images_for_training')
    with pytest.raises(ValueError):api().validate_catalog({'datasets':[bad]})


def test_npy_release_preserves_instance_ids_and_evaluation_role(tmp_path):
    images=np.zeros((2,4,5,3),dtype=np.uint8);masks=np.zeros((2,4,5),dtype=np.float32)
    masks[:,1:3,1:3]=9
    archive=tmp_path/'release.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,value in [('imgs.npy',images),('anns.npy',masks)]:
            data=io.BytesIO();np.save(data,value,allow_pickle=False);z.writestr('PBL/'+name,data.getvalue())
    records=api().npy_release_records(archive,dataset='PBL',split='evaluation',license_id='review-pending')
    assert len(records)==2 and records[0]['cell_count']==1
    assert records[0]['instances'][0]['instance_id']==9
    assert all(not r['training_eligible'] for r in records)


def test_noninteger_or_nonfinite_mask_conversion_fails():
    for value in [np.array([[.5]]),np.array([[float('nan')]]),np.array([[-1]])]:
        with pytest.raises(ValueError):api().mask_png(value)


def test_parquet_binary_instances_remain_separate_with_classes(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    image,_=pair();masks=[]
    for x in [1,5]:
        a=np.zeros((8,8),dtype=np.uint8);a[1:3,x:x+2]=1
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');masks.append(dict(bytes=b.getvalue(),path=None))
    p=tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([dict(image=dict(bytes=image,path=None),instances=masks,categories=[2,4])]),p)
    r=list(api().parquet_instance_records(p,dataset='fixture',split='train',license_id='test-only',limit=1))[0]
    assert r['cell_count']==2 and r['instance_classes']==[2,4]
    assert r['source_row_index']==0
    assert 'instance_classes' not in api().project_image_task(r)['inputs']


def test_acquisition_cli_downloads_pinned_asset_without_mutating_catalog(server,tmp_path):
    import subprocess,sys
    base,body=server
    source=dict(id='a',name='fixture',role='evaluation_only',license_status='verified',source_url='https://example.org',status='registered',overlap_group='a',assets=[dict(name='asset.bin',url=base+'/data',bytes=len(body),sha256=hashlib.sha256(body).hexdigest())])
    p=tmp_path/'catalog.json';p.write_text(json.dumps({'datasets':[source]}));before=p.read_bytes()
    script=Path('scripts/acquire_imaging.py')
    assert script.exists(), 'Image acquisition CLI missing'
    r=__import__('subprocess').run([sys.executable,str(script),'--catalog',str(p),'--dataset','a','--output',str(tmp_path/'download'),'--minimum-free-gib','0'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert (tmp_path/'download/a/asset.bin').read_bytes()==body
    assert p.read_bytes()==before
    result=json.loads((tmp_path/'download/acquisition_summary.json').read_text())
    assert result['datasets'][0]['status']=='downloaded_verified'
    assert result['datasets'][0]['training_ready'] is False


def test_zip_pair_records_preserve_source_members_and_holdout(tmp_path):
    from mora3.data import imaging
    def png(array):
        buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG');return buffer.getvalue()
    assert hasattr(imaging,'zip_instance_records')
    p=tmp_path/'source.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('test/source/a.png',png(np.zeros((3,4),dtype=np.uint8)))
        z.writestr('test/instance/a.png',png(np.array([[0,2,2,0],[0,0,0,0],[7,7,0,0]],dtype=np.uint8)))
    pairs=[dict(image='test/source/a.png',mask='test/instance/a.png',split='test')]
    records=list(imaging.zip_instance_records(p,dataset='fixture',license_id='CC-BY-4.0',pairs=pairs))
    assert records[0]['cell_count']==2
    assert records[0]['source_members']==pairs[0]
    assert records[0]['source_archive_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    assert records[0]['training_eligible'] is False
    assert 'cell_count' not in imaging.project_image_task(records[0])['inputs']


def test_overlapping_parquet_masks_preserve_each_source_annotation(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    image,_=pair();a=np.zeros((8,8),dtype=np.uint8);a[1:3,1:3]=1
    b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');blob=b.getvalue()
    p=tmp_path/'overlap.parquet'
    masks=[dict(bytes=blob,path=None)]*2
    pq.write_table(pa.Table.from_pylist([dict(image=dict(bytes=image,path=None),instances=masks)]),p)
    r=list(api().parquet_instance_records(p,dataset='fixture',split='test',license_id='test-only',limit=1))[0]
    assert r['cell_count']==2
    assert [x['area_px'] for x in r['instances']]==[4,4]
    assert r['overlap_pixels']==4
    assert r['mask_encoding']=='source_binary_instance_collection'
    assert r['instance_mask_sha256']==[hashlib.sha256(blob).hexdigest()]*2
    task=api().project_image_task(r)
    assert task['targets']['instance_mask_sha256']==r['instance_mask_sha256']
    assert 'overlap_pixels' not in task['inputs']
