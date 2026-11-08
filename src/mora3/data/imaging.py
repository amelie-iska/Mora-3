"""Bounded immutable acquisition and microscopy instance-mask task records.

Use source-specific licenses/splits. This module supplies data contracts, not an
image encoder or proof that image data are linked to molecular response labels.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import urllib.parse
import urllib.request
import zipfile
import numpy as np
from PIL import Image


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download_asset(url,destination,*,max_bytes,expected_bytes=None,expected_sha256=None,resume=False,min_free_gib=200):
    """Download without overwriting; incomplete/failed parts remain resumable."""
    u=urllib.parse.urlsplit(url)
    if u.scheme!='https' and not (u.scheme=='http' and u.hostname in {'127.0.0.1','localhost'}):
        raise ValueError('Dataset transport must use HTTPS')
    if u.username or u.password:raise ValueError('Credentials must not be embedded in asset URLs')
    if type(max_bytes) is not int or max_bytes<=0:raise ValueError('Positive byte budget required')
    if expected_bytes is not None and (type(expected_bytes) is not int or not 0<expected_bytes<=max_bytes):
        raise ValueError('Expected size exceeds byte budget')
    if expected_sha256 is not None and not re.fullmatch('[0-9a-f]{64}',expected_sha256):
        raise ValueError('Invalid expected checksum')
    destination=Path(destination);part=destination.with_suffix(destination.suffix+'.part')
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    if part.is_symlink():raise ValueError('Partial file must not be a symlink')
    if part.exists() and not resume:raise FileExistsError(part)
    destination.parent.mkdir(parents=True,exist_ok=True)
    offset=part.stat().st_size if part.exists() else 0
    if offset>max_bytes:raise ValueError('Existing partial exceeds byte budget')
    required=(expected_bytes if expected_bytes is not None else max_bytes)-offset
    if shutil.disk_usage(destination.parent).free < min_free_gib*1024**3+max(0,required):
        raise ValueError('Insufficient disk headroom for asset')
    if expected_bytes is None or offset!=expected_bytes:
        headers={'User-Agent':'Mora3-dataset-acquisition/1.0','Accept-Encoding':'identity'}
        if offset:headers['Range']=f'bytes={offset}-'
        request=urllib.request.Request(url,headers=headers)
        with urllib.request.urlopen(request,timeout=45) as response:
            if offset:
                content_range=response.headers.get('Content-Range','')
                if response.status!=206 or not content_range.startswith(f'bytes {offset}-'):
                    raise ValueError('Range resume unsupported; partial left unchanged')
            length=response.headers.get('Content-Length')
            if length and offset+int(length)>max_bytes:raise ValueError('Asset exceeds byte budget')
            mode='ab' if offset else 'xb'
            with part.open(mode) as stream:
                total=offset
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    total+=len(chunk)
                    if total>max_bytes:raise ValueError('Asset exceeds byte budget')
                    stream.write(chunk)
    actual=part.stat().st_size
    if expected_bytes is not None and actual!=expected_bytes:raise ValueError('Incomplete asset size')
    digest=file_sha256(part)
    if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Asset checksum mismatch')
    # Link atomically and exclusively: no destination can be replaced by a race.
    import os
    os.link(part,destination)
    # Retain the completed partial as a hard link; no protected download progress is removed.
    return dict(source_url=urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'','')),
                path=str(destination),bytes=actual,sha256=digest,resumed_from_bytes=offset,
                checksum_origin='upstream' if expected_sha256 else 'locally-computed')


def archive_members(path,*,max_uncompressed_bytes=100*1024**3):
    """Inspect ZIP metadata; no extraction or source mutation."""
    result=[];total=0;seen=set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name=info.filename;p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in p.parts[0] or name in seen:
                raise ValueError('Unsafe archive member')
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Archive symlinks are unsupported')
            seen.add(name);total+=info.file_size
            if total>max_uncompressed_bytes:raise ValueError('Archive expanded byte budget exceeded')
            result.append(dict(name=name,bytes=info.file_size,directory=info.is_dir()))
    return result


def image_mask_record(image_bytes,mask_bytes,*,dataset,record_id,split,license_id,training_eligible=False,max_pixels=16_000_000):
    if split not in {'train','validation','test','evaluation'}:raise ValueError('Explicit source split required')
    if not all(isinstance(x,str) and x for x in [dataset,record_id,license_id]):raise ValueError('Source provenance required')
    if training_eligible and split!='train':raise ValueError('Held-out source cannot be training eligible')
    with Image.open(io.BytesIO(image_bytes)) as im,Image.open(io.BytesIO(mask_bytes)) as ma:
        if im.size!=ma.size:raise ValueError('Image and mask dimensions differ')
        width,height=im.size
        if width*height>max_pixels:raise ValueError('Image exceeds pixel budget')
        if getattr(im,'n_frames',1)!=1 or getattr(ma,'n_frames',1)!=1:raise ValueError('Volumetric/time-series data require a separate adapter')
        image_mode=im.mode;channels=len(im.getbands());mask=np.array(ma)
    if mask.ndim!=2 or mask.dtype.kind not in 'ui' or (mask<0).any():
        raise ValueError('Mask must contain nonnegative integer instance IDs')
    labels,inverse,counts=np.unique(mask,return_inverse=True,return_counts=True);inverse=inverse.reshape(-1)
    xs=np.tile(np.arange(width),height);ys=np.repeat(np.arange(height),width)
    sumx=np.bincount(inverse,weights=xs);sumy=np.bincount(inverse,weights=ys)
    minx=np.full(len(labels),width);miny=np.full(len(labels),height)
    maxx=np.full(len(labels),-1);maxy=np.full(len(labels),-1)
    np.minimum.at(minx,inverse,xs);np.minimum.at(miny,inverse,ys)
    np.maximum.at(maxx,inverse,xs);np.maximum.at(maxy,inverse,ys)
    instances=[]
    for i,label in enumerate(labels):
        if label==0:continue
        instances.append(dict(instance_id=int(label),area_px=int(counts[i]),
            centroid_px=[float(sumx[i]/counts[i]),float(sumy[i]/counts[i])],
            bbox_xyxy=[int(minx[i]),int(miny[i]),int(maxx[i])+1,int(maxy[i])+1]))
    return dict(schema='mora3.image_instance_record.v1',dataset=dataset,record_id=record_id,
        split=split,license=license_id,training_eligible=bool(training_eligible),
        width=width,height=height,channels=channels,image_mode=image_mode,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
        annotation_type='instance_mask',physical_pixel_size=None,
        biological_context={'perturbation':None,'expression':None,'cell_line':None},
        cell_count=len(instances),instances=instances)


def project_image_task(record):
    """Image-to-mask task: every mask-derived field belongs only to targets."""
    inputs={key:record[key] for key in ['image_sha256','width','height','channels','image_mode']}
    targets={key:record[key] for key in ['mask_sha256','cell_count','instances']}
    for key in ['mask_encoding','instance_mask_sha256','instance_classes','overlap_pixels']:
        if key in record:targets[key]=record[key]
    audit={key:record[key] for key in ['schema','dataset','record_id','split','license','training_eligible']}
    return dict(task='image_to_instance_mask',inputs=inputs,targets=targets,audit=audit)


def validate_catalog(catalog):
    seen=set()
    if not isinstance(catalog.get('datasets'),list) or not catalog['datasets']:raise ValueError('Dataset catalog required')
    for entry in catalog['datasets']:
        required={'id','name','role','license_status','source_url','status','overlap_group'}
        if not required.issubset(entry) or any(not isinstance(entry[x],str) or not entry[x] for x in required):raise ValueError('Incomplete source record')
        if entry['id'] in seen:raise ValueError('Duplicate dataset source')
        if entry['role'] not in {'train_and_evaluate','evaluation_only','candidate_review','collection_manifest'}:raise ValueError('Invalid source role')
        if entry['license_status'] not in {'verified','source_review_pending','restricted_access'}:raise ValueError('Unreviewed source rights')
        if not entry['source_url'].startswith('https://'):raise ValueError('Source URL must use HTTPS')
        seen.add(entry['id'])
    return catalog


def mask_png(values):
    """Lossless integer-ID conversion; no rounding, binarization or relabeling."""
    a=np.asarray(values)
    if a.ndim!=2 or a.dtype.kind not in 'biuf' or not np.isfinite(a).all():raise ValueError('Invalid numeric mask')
    if (a<0).any() or (a>65535).any() or not np.equal(a,np.floor(a)).all():raise ValueError('Mask IDs are not lossless uint16 values')
    encoded=io.BytesIO();Image.fromarray(a.astype(np.uint16)).save(encoded,format='PNG')
    return encoded.getvalue()


def npy_release_records(archive_path,*,dataset,split,license_id):
    """Read the public SAMCell PBL release safely, without NumPy pickle loading."""
    members=archive_members(archive_path,max_uncompressed_bytes=256*1024**2)
    names=[x['name'] for x in members if not x['directory']]
    image_name=next((x for x in names if x.endswith('/imgs.npy')),None)
    mask_name=next((x for x in names if x.endswith('/anns.npy')),None)
    if not image_name or not mask_name:raise ValueError('PBL image/annotation arrays missing')
    with zipfile.ZipFile(archive_path) as archive:
        image_blob=archive.read(image_name);mask_blob=archive.read(mask_name)
    images=np.load(io.BytesIO(image_blob),allow_pickle=False)
    masks=np.load(io.BytesIO(mask_blob),allow_pickle=False)
    if images.ndim!=4 or masks.ndim!=3 or images.shape[:3]!=masks.shape or images.dtype!=np.uint8:
        raise ValueError('Incompatible PBL array shapes or image dtype')
    records=[]
    for index,(image,mask) in enumerate(zip(images,masks)):
        raw=io.BytesIO();Image.fromarray(image).save(raw,format='PNG')
        record=image_mask_record(raw.getvalue(),mask_png(mask),dataset=dataset,record_id=f'{dataset}:{index}',split=split,license_id=license_id)
        record.update(source_index=index,source_archive_sha256=file_sha256(archive_path),
                      source_image_array_sha256=hashlib.sha256(image_blob).hexdigest(),
                      source_mask_array_sha256=hashlib.sha256(mask_blob).hexdigest(),
                      mask_transform='integer IDs cast losslessly to uint16 PNG; no relabeling')
        records.append(record)
    return records


def parquet_instance_records(path,*,dataset,split,license_id,limit):
    """Preserve source binary instances, including overlap and original classes."""
    import pyarrow.parquet as pq
    if type(limit) is not int or limit<=0:raise ValueError('Positive record limit required')
    source_hash=file_sha256(path);offset=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
        row=batch.to_pylist()[0]
        image_bytes=row['image']['bytes']
        if not image_bytes:raise ValueError('Embedded image bytes required')
        with Image.open(io.BytesIO(image_bytes)) as image:width,height=image.size
        if width*height>16_000_000:raise ValueError('Image pixel budget exceeded')
        annotations=row['instances']
        if len(annotations)>65535:raise ValueError('Instance ID range exceeded')
        coverage=np.zeros((height,width),dtype=np.uint16)
        instances=[];mask_hashes=[]
        for label,annotation in enumerate(annotations,1):
            with Image.open(io.BytesIO(annotation['bytes'])) as image:mask=np.array(image)
            if mask.shape!=coverage.shape or not np.isin(mask,[0,1,255]).all():raise ValueError('Invalid binary instance mask')
            foreground=mask!=0
            ys,xs=np.nonzero(foreground)
            if not len(xs):raise ValueError('Empty instance annotation')
            coverage+=foreground
            instances.append(dict(instance_id=label,area_px=len(xs),
                centroid_px=[float(xs.mean()),float(ys.mean())],
                bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]))
            mask_hashes.append(hashlib.sha256(annotation['bytes']).hexdigest())
        # The collection hash identifies ordered original masks, not a flattened map.
        result=image_mask_record(image_bytes,mask_png(np.zeros_like(coverage)),dataset=dataset,record_id=f'{dataset}:{split}:{offset}',split=split,license_id=license_id)
        result.update(instances=instances,cell_count=len(instances),
                      instance_mask_sha256=mask_hashes,mask_encoding='source_binary_instance_collection',
                      mask_sha256=hashlib.sha256(json.dumps(mask_hashes,separators=(',',':')).encode()).hexdigest(),
                      overlap_pixels=int((coverage>1).sum()))
        result.update(source_row_index=offset,source_parquet_sha256=source_hash,
                      mask_transform='original binary masks preserved independently; IDs follow source list order; overlap retained')
        if 'categories' in row:
            if len(row['categories'])!=len(annotations):raise ValueError('Instance class alignment mismatch')
            result['instance_classes']=row['categories']
        result['source_context']={k:row[k] for k in ['patient','tissue'] if k in row}
        yield result
        offset+=1
        if offset>=limit:return


def zip_instance_records(path,*,dataset,license_id,pairs):
    """Read explicitly reviewed single-frame instance pairs without extraction."""
    inventory={x['name']:x for x in archive_members(path) if not x['directory']}
    digest=file_sha256(path)
    with zipfile.ZipFile(path) as archive:
        for pair in pairs:
            for key in ['image','mask']:
                if pair[key] not in inventory or inventory[pair[key]]['bytes']>256*1024**2:
                    raise ValueError('Missing or oversized source member')
            image_bytes=archive.read(pair['image']);mask_bytes=archive.read(pair['mask'])
            record=image_mask_record(image_bytes,mask_bytes,dataset=dataset,
                record_id=dataset+':'+pair['image'],split=pair['split'],license_id=license_id)
            record.update(source_archive=str(path),source_archive_sha256=digest,source_members=dict(pair))
            yield record
