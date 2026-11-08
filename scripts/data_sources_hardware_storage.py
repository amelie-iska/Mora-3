"""Latest USER storage policy: physical available hardware less64GiB, no allocations."""
import hashlib,json,os,pathlib

def load_policy(path,expected_sha256):
    data=pathlib.Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('exact governing USERpolicy hash changed')
    p=json.loads(data);s=p['storage']
    if s['project_raw_or_derived_caps_bytes'] is not None or s['family_source_helper_allocation_caps_bytes'] is not None or s['free_space_buffer_bytes']!=64*1024**3 or s['dynamic_hardware_guard_required'] is not True:raise ValueError('hardware-only storage policy with64GiBbuffer required')
    if p['training']['full_or_substantive_training_authorized_before_complete_dataset'] is not False:raise ValueError('dataset-before-training gate changed; new review needed')
    return p

def hardware_snapshot(path,buffer_bytes=64*1024**3):
    s=os.statvfs(path);available=s.f_bavail*s.f_frsize
    return dict(storage_basis='actual filesystem available bytes less free buffer',total_bytes=s.f_blocks*s.f_frsize,available_bytes=available,free_space_buffer_bytes=buffer_bytes,usable_available_bytes=max(0,available-buffer_bytes))

def disk_reason(free_bytes,elapsed,limits,incoming_bytes=0):
    if any(k in limits for k in ['stage_cap_bytes','stop_margin_bytes','raw_cap_bytes','derived_cap_bytes','allocation_cap_bytes']):raise ValueError('superseded artificial storage allocation in hardware guard')
    if free_bytes<0 or incoming_bytes<0 or elapsed<0:raise ValueError('invalid measured hardware/runtime values')
    if free_bytes-incoming_bytes<=limits['minimum_free_bytes']:return 'hardware_free_space_buffer'
    if elapsed>=limits['max_seconds']:return 'finite_runtime_highwater'
    return None
