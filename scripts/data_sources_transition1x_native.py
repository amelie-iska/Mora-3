"""Original Transition1x native frames; no derivative-coordinate/force mixing."""
import collections,hashlib,re
import numpy as np
ROLES=('reactant','product','transition_state')
ELEMENT={1:'H',6:'C',7:'N',8:'O'}
def native_record(formula,reaction,split,atomic_numbers,roles,path_frames,HDF5_group,source_path):
    z=np.asarray(atomic_numbers)
    if split not in ['train','val','test'] or z.ndim!=1 or len(z)==0 or z.dtype.kind not in 'iu' or not set(map(int,z))<=set(ELEMENT) or not isinstance(path_frames,int) or path_frames<1:raise ValueError('native source split/atom/configuration axes invalid')
    tokens=re.findall(r'([A-Z][a-z]?)([0-9]*)',formula)
    if ''.join(a+b for a,b in tokens)!=formula or collections.Counter({a:int(b or 1) for a,b in tokens})!=collections.Counter(ELEMENT[int(n)] for n in z):raise ValueError('source formula/atom composition mismatch')
    n=len(z);out={}
    for name in ROLES:
        q=roles[name];p=np.asarray(q['positions']);f=np.asarray(q['forces']);e=np.asarray(q['energy'])
        if not np.array_equal(np.asarray(q['atomic_numbers']),z) or p.shape!=(1,n,3) or f.shape!=p.shape or e.shape!=(1,) or any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in [p,f,e]):raise ValueError('native role axis, atomorder, dtype or numerical integrity failure')
        out[name]=dict(positions=p.tolist(),forces=f.tolist(),energy=e.tolist(),native_hash=int(q['hash']),native_configuration_count=1,native_shape=list(p.shape),native_float_dtype='float64',position_array_sha256=hashlib.sha256(p.tobytes()).hexdigest(),force_array_sha256=hashlib.sha256(f.tobytes()).hexdigest(),frame='original native ASE source row frame',frame_transformed=False,source_HDF5_group=HDF5_group+'/'+name)
    return dict(source_record_id='Transition1x_original:v4:'+formula+':'+reaction,source_reaction_group='Transition1x:v4:'+formula+':'+reaction,formula=formula,source_native_split=split,atomic_numbers=list(map(int,z)),source_atomic_number_dtype=str(z.dtype),source_hash_is_global_identity=False,roles=out,native_path_configurations=path_frames,native_path_shape=[path_frames,n,3],path_source_HDF5_group=HDF5_group,source_HDF5_path=source_path,path_axis_kind='saved NEB optimization image, not physical time',coordinate_unit='angstrom',energy_unit='eV',force_unit='eV/angstrom',unit_evidence='publisher methods/ASE database convention; HDF5 has no unit attributes',method='wB97x/6-31G(d)',evidence_class='computational quantum chemistry; not experimental measurement',electronic_charge=None,spin_multiplicity=None,electronic_charge_known=False,spin_multiplicity_known=False,stationary_saddle_confirmed=False,conditional_TS_inference_admitted=False,phaseII_dynamics_admitted=False,task_admitted=False)
