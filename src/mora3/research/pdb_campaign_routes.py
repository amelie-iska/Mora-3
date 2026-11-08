"""Guarded campaign binding for the deposited PDB masked-coordinate objective."""
from dataclasses import dataclass

from mora3.data.admet import canonical
from mora3.data.pdb_masked_coordinates import FAMILY, TASK, PDBCoordinateAxes, pdb_coordinate_packets
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.pdb_coordinate_model import PDBMaskedCoordinates

from .assay_campaign_routes import _original_train_route, _require_original_task_batch, _view_scope
from .primary_native_routes import NativeRouteWindow, PreparedNativeTargets


@dataclass(frozen=True)
class PDBCampaignAdapter:
    original_view: object
    current_view: object
    source_snapshot_sha256: str
    axis: PDBCoordinateAxes

    family=FAMILY
    task_key=TASK
    loss_unit='mean per-original-packet masked Cartesian squared error in angstrom squared'
    target_unit='observed original masked Cartesian components in angstrom'

    def __post_init__(self):
        if type(self.axis) is not PDBCoordinateAxes:
            raise ValueError('Complete original PDB coordinate axes required')
        paths={p for p,_,_ in self.axis.source_coordinates}
        a,b=self.original_view,self.current_view
        if (any(a._sealed_by_path.get(path)!=b._sealed_by_path.get(path)for path in paths)
                or {r for r in a._ordinal_exclusions if r[0] in paths}!={r for r in b._ordinal_exclusions if r[0] in paths}):
            raise ValueError('Current effective view changes an original PDB coordinate')
        object.__setattr__(self,'_identity',canonical(self._document()))

    def _document(self):
        original,current=_view_scope(self.original_view,self.current_view,self.source_snapshot_sha256)
        if self.axis.native_view_policy_sha256!=original:
            raise ValueError('Original PDB axis and exact source view differ')
        return dict(schema='Mora3.PDBMaskedCoordinateCampaignAdapter/v1',source_snapshot_sha256=self.source_snapshot_sha256,
            original_view_policy_sha256=original,current_view_policy_sha256=current,
            axes=self.axis.binding(),family=FAMILY,task=TASK,split='train',loss_unit=self.loss_unit,target_unit=self.target_unit)

    def executable_binding(self):
        doc=self._document()
        if canonical(doc)!=self._identity:
            raise ValueError('Exact PDB source view or deposited atom axis changed')
        return doc

    def train_binding(self,*,batch_size):
        return _original_train_route(self,batch_size=batch_size,allowed_paths={p for p,_,_ in self.axis.source_coordinates})

    def prepare(self,window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original train PDB window required')
        batch=window.batch
        _require_original_task_batch(self,batch,'train')
        packets=pdb_coordinate_packets(self.original_view,batch,split='train')
        known=set(self.axis.source_coordinates)
        if len(packets)!=len(batch.source_row_ordinals) or any(p.source_coordinate not in known for p in packets):
            raise ValueError('Every original PDB train coordinate must belong to the verified source axis')
        observed=sum(len(x) for p in packets for x in p.masked_targets_angstrom if x is not None)
        if not observed:
            return PreparedNativeTargets(0,'original masked Cartesian target positions unavailable',None)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch=owner.pdb
            if type(branch) is not PDBMaskedCoordinates or branch.axes!=self.axis:
                raise ValueError('Exact registered PDB branch and complete original axes required')
            return branch.objective(self.original_view,batch,split='train')

        return PreparedNativeTargets(observed,None,evaluate)
