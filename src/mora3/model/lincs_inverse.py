"""Full source LINCS profile-to-native-SMILES byte reconstruction reference.

Aliases existing LINCS numeric context and PubChem byte decoder parameters.
Native 978-feature L1000 stays distinct from Rosetta's 977-feature schema.
Source/split eligibility and campaign authorization remain external.
"""
import copy
import torch
from torch import nn
from .lincs_model import SharedLincsModel
from .pubchem_decoder import PubChemDecoder
from .mixed_payload_projection import exact_wire
from .source_byte_objective import validate_byte_targets,source_byte_objective


class SharedLincsInverseModel(nn.Module):
    def __init__(self,lincs,decoder,*,max_batch_size=64,max_total_target_bytes=64*1024*1024):
        super().__init__()
        if (type(lincs)is not SharedLincsModel or type(decoder)is not PubChemDecoder
                or lincs.parameterization_policy!='mup-lincs-v1' or decoder.get_extra_state()['version']!=2
                or lincs.shared is not decoder.shared):
            raise ValueError('exact same shared primary annotation/LINCS/byte decoder owners required')
        lincs.validate_parameterization_configuration()
        if any(type(n)is not int or n<1 for n in(max_batch_size,max_total_target_bytes)):
            raise ValueError('explicit positive complete target/resource bounds required')
        self.lincs=lincs;self.decoder=decoder;self._owners=(lincs,decoder)
        self.max_batch_size=max_batch_size;self.max_total_target_bytes=max_total_target_bytes
        self._configuration=self._local_state()

    def _local_state(self):
        return dict(schema='Mora3.SharedNativeLincsInverse/v1',
            condition='existing-strict-native-LINCS-full-axis-values+observed-mask+source-dose-time-metadata',
            target='complete-original-native-canonical-isomeric-SMILES-UTF8+EOS;no-normalization-crop-or-SELFIES-conversion',
            objective='all-next-byte-token+EOS-CE;global-target-token-mean;full-recurrent-history',
            source_admission='external-native-reader-current-role-and-global-campaign;none-issued-here',
            resource_bounds=dict(max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes))

    def get_extra_state(self):
        if (self.lincs is not self._owners[0]or self.decoder is not self._owners[1]
                or self.lincs.shared is not self.decoder.shared or not exact_wire(self._local_state(),self._configuration)):
            raise ValueError('native LINCS inverse shared owner/resource policy drift')
        return dict(**copy.deepcopy(self._configuration),lincs=self.lincs.get_extra_state(),decoder=self.decoder.get_extra_state())

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('native LINCS inverse checkpoint mismatch')

    def _views(self,views):
        views=self.lincs._validate(views)
        if len(views)>self.max_batch_size or any(v.direction!='profile_to_molecule'for v in views):
            raise ValueError('bounded native LINCS profile_to_molecule views required')
        return views

    def condition(self,views):
        self.get_extra_state();return self.lincs.condition(self._views(views))

    def objective(self,views,*,chunk_size=1024,recompute=False):
        self.get_extra_state();views=self._views(views);targets=[]
        for v in views:
            obj=v.targets()
            if (set(obj)!={'canonical_isomeric_smiles'}or type(obj['canonical_isomeric_smiles'])is not str
                    or not obj['canonical_isomeric_smiles']):
                raise ValueError('complete exact original native SMILES target required')
            targets.append(obj['canonical_isomeric_smiles'].encode('utf8'))
        targets=validate_byte_targets(targets,max_batch_size=self.max_batch_size,
            max_total_target_bytes=self.max_total_target_bytes,chunk_size=chunk_size,recompute=recompute)
        context=self.lincs.condition(views)
        return source_byte_objective(self.decoder,context,targets,chunk_size=chunk_size,recompute=recompute,
            max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes)


def lincs_inverse_parameter_roles(model,*,identifier_axis):
    """Every exact shared alias follows its already applied native owner rule."""
    from .assay_mup import assay_mup_parameter_roles
    from .sequence_mup import sequence_mup_parameter_roles
    from .parameterization import parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not SharedLincsInverseModel:
        raise ValueError('exact native LINCS inverse owner required')
    model.get_extra_state()
    if set(model._modules)!={'lincs','decoder'}or model._parameters or model._buffers:
        raise ValueError('exact native LINCS inverse alias-only wrapper required')
    roles={**{'lincs.'+n:r for n,r in assay_mup_parameter_roles(model.lincs,identifier_axis=identifier_axis).items()},
           **{'decoder.'+n:r for n,r in sequence_mup_parameter_roles(model.decoder,identifier_axis=identifier_axis).items()}}
    names=dict(model.named_parameters(remove_duplicate=False));by_id={}
    for name,r in roles.items():
        key=id(names[name])
        if key in by_id and by_id[key]!=r:raise ValueError('native inverse shared axis/role mismatch')
        by_id[key]=r
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
