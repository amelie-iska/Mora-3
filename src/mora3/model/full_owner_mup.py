"""Compose exact opt-in component μP rules over the literal registered owner.

Declarations merge by Parameter identity, never shape/name resemblance. Every
literal alias must agree and every actual Parameter must be covered. New source
extensions fail closed until their exact operator builder is registered here.
This is an optimizer family, not corpus admission, model quality or resource
qualification. The factory remains the authority for source-bound assembly.
"""
import hashlib
import json
from torch import nn
from .full_owner import SourceBoundOwner
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def _latent_roles(model,decoder,policy):
    from .profile_latent_decoder import ProfileLatentSelfiesDecoder
    from .profile_decoder import ProfileSelfiesDecoder
    from .profile_gfn import ProfileGFNPolicy
    if (type(model)is not ProfileLatentSelfiesDecoder or type(decoder)is not ProfileSelfiesDecoder
            or type(policy)is not ProfileGFNPolicy or model.decoder is not decoder
            or policy.profiles is not decoder.profiles or policy.parameterization_policy!='mup-profile-gfn-v1'
            or set(model._modules)!={'latent_projection'} or model._parameters or model._buffers
            or not exact_wire(model._policy,policy.get_extra_state())
            or (model.dimension,model.horizon)!=(policy.dimension,policy.horizon)):
        raise ValueError('exact new registered GFN/parent decoder/latent residual source ownership required')
    policy.validate_parameterization_configuration();decoder._validate_recurrent_parameterization()
    layer=model.latent_projection;w=policy.context_width;d=policy.dimension
    if (type(layer)is not nn.Linear or layer._modules or layer._buffers or layer.bias is not None
            or (layer.in_features,layer.out_features)!=(d,w) or tuple(layer.weight.shape)!=(w,d)):
        raise ValueError('exact bias-free fixed-latent to hidden residual input required')
    names=tuple('latent-coordinate:'+str(i) for i in range(d))
    identity=hashlib.sha256(json.dumps(['shared-latent-residual-coordinates.v2',names,d,policy.horizon,
            policy.get_extra_state()['source_binding']],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    roles={'latent_projection.weight':ParameterRole('input',(Axis('width','hidden'),Axis('finite',identity)))}
    parameter_ledger(model,roles)
    return roles


def full_owner_mup_parameter_roles(model,*,identifier_axis=('context','observation'),
        geometry_identifier_axis,color_identifier_axis):
    """Cover the actual registered complete owner, with explicit finite ID axes."""
    if type(model)is not SourceBoundOwner:raise ValueError('exact source-bound primary owner required')
    model.get_extra_state()
    from .cooperative_training import CooperativeModel,GeometryCooperativeModel
    from .morphology_mup import cooperative_mup_parameter_roles,_geometry_roles,morphology_mup_parameter_roles
    from .morphology import MorphologyInstance
    from .geometry_conditioning import GeometryConditioner
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    from .annotation_mup import annotation_mup_parameter_roles
    from .response_flow import ResponseFlowHead,ResponseFlowDenoiserV2
    from .flow_mup import flow_mup_parameter_roles
    from .residue_objective import SharedResidueModel
    from .residue_mup import residue_mup_parameter_roles
    from .molecule_profile import SharedMoleculeProfileModel
    from .sequence_mup import profile_mup_parameter_roles,sequence_mup_parameter_roles
    from .profile_decoder import ProfileSelfiesDecoder
    from .nucleotide_decoder import NucleotideDecoder
    from .rfam_family import RfamFamilyModel
    from .profile_gfn import ProfileGFNPolicy
    from .gfn_mup import gfn_mup_parameter_roles
    from .profile_latent_decoder import ProfileLatentSelfiesDecoder
    from .reconstruction_cooperative import ReconstructionCooperativeModel
    from .tox21 import SharedTox21Model
    from .lincs_model import SharedLincsModel
    from .assay_mup import assay_mup_parameter_roles
    from .uniprot_go import NativeGOAHead
    from .goa_mup import goa_mup_parameter_roles
    from .color_conditioning import ConditionedColorSemantic,ConditionedColorInstance
    from .color_mup import conditioned_color_mup_parameter_roles
    from .admet_model import SharedAdmetModel
    from .admet_mup import admet_mup_parameter_roles
    from .mcsa_role_list import SourceRoleListModel,source_role_list_parameter_roles
    from .pubchem_decoder import PubChemDecoder
    from .cartesian_gfn_policy import SharedCartesianGFNPolicy
    from .cartesian_mup import cartesian_mup_parameter_roles
    from .source_annotation_objectives import SourceAnnotationObjectives,source_annotation_objectives_parameter_roles
    from .cell_native import SharedNativeCellModel
    from .cell_mup import cell_mup_parameter_roles
    from .raw_count_cell_model import SharedRawCountCellModel,raw_cell_parameter_roles
    from .source_text import SharedSourceTextModel
    from .lincs_inverse import SharedLincsInverseModel,lincs_inverse_parameter_roles
    from .native_sequence_priors import NativeSequencePriorModel,native_sequence_prior_parameter_roles
    from .pdb_coordinate_model import PDBMaskedCoordinates,pdb_coordinate_parameter_roles
    from .rosetta_profile_flow import RosettaProfileFlow,rosetta_profile_flow_roles
    from .source_scalar_objectives import SharedNativeScalarModel,source_scalar_mup_parameter_roles
    from .native_grayscale_flow import NativeGrayscaleFlow,native_grayscale_flow_roles
    from .nci_assay_model import NativeNciAssayModel,nci_assay_parameter_roles
    from .bioelectric_outcomes import SharedBioelectricOutcomeModel,bioelectric_outcome_parameter_roles
    from .bioelectric_group_counts import SharedHcn4LateralityModel,hcn4_laterality_parameter_roles
    from .expansionrx_native_model import ExpansionRxNativeModel,expansionrx_parameter_roles
    from .reactzyme_product import NativeReactZymeProductModel,reactzyme_parameter_roles
    from .cema_trajectory_model import SharedCemaTrajectoryModel,cema_trajectory_parameter_roles
    from .betse_membrane import NativeBETSEMembraneHead,betse_membrane_parameter_roles
    from .tox21_current_model import NativeTox21SourceModel,tox21_current_model_parameter_roles
    from .reactome_static_model import NativeReactomeStaticModel,reactome_static_parameter_roles
    from .biogrid_chemical_model import NativeBioGRIDChemicalModel,biogrid_chemical_parameter_roles
    from .lincx_fetch_model import SharedLinCxFetchModel,lincx_fetch_parameter_roles
    from .string_score_model import NativeStringScoreModel,string_score_parameter_roles
    from .uniprot_native_objectives import SharedUniProtNativeAnnotationModel,uniprot_native_annotation_parameter_roles
    from .uniprot_full_record_model import SharedUniProtFullRecordModel,uniprot_full_record_parameter_roles
    for axis in (identifier_axis,geometry_identifier_axis,color_identifier_axis):
        if (type(axis)is not tuple or not axis or any(type(x)is not str or not x for x in axis)
                or len(set(axis))!=len(axis)):
            raise ValueError('explicit complete unique finite source identifier axes required')
    declarations={};seen=set();latent={}
    for module in model.modules():
        if isinstance(module,ReconstructionCooperativeModel):
            if not hasattr(module,'profile_gfn'):raise ValueError('actual registered profile GFN required')
            key=id(module.reconstruction);value=(module.decoder,module.profile_gfn)
            if key in latent and any(a is not b for a,b in zip(latent[key],value)):
                raise ValueError('ambiguous latent residual parent/policy binding')
            latent[key]=value
    def merge(module,roles):
        names=dict(module.named_parameters(remove_duplicate=False))
        if set(names)!=set(roles):raise ValueError('component declaration/alias completeness mismatch')
        for name,r in roles.items():
            key=id(names[name])
            if key in declarations and declarations[key]!=r:
                raise ValueError('conflicting shared source operator/axis/role declarations')
            declarations[key]=r
    from mora3.extensions.scri.native_owner import NativeSCRIModel,native_scri_roles
    from .forest_flow import NativeForestFlowHead,forest_flow_parameter_roles
    from .forest_analogy import NativeForestAnalogyHead,forest_analogy_parameter_roles
    from .forest_token_decoder import NativeForestTokenDecoder,forest_decoder_parameter_roles
    from .forest_quality import NativeForestQualityHead,forest_quality_parameter_roles
    def visit(module):
        if id(module)in seen:return
        seen.add(id(module));t=type(module);roles=None
        if t in (CooperativeModel,GeometryCooperativeModel):
            roles=cooperative_mup_parameter_roles(module,**(
                dict(geometry_identifier_axis=geometry_identifier_axis) if t is GeometryCooperativeModel else {}))
        elif t is MorphologyInstance:roles=morphology_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t in (SharedAnnotationModel,SharedPhysicalModel):roles=annotation_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t in (ResponseFlowHead,ResponseFlowDenoiserV2):roles=flow_mup_parameter_roles(module)
        elif t is SharedResidueModel:roles=residue_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedMoleculeProfileModel:roles=profile_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t in (NucleotideDecoder,ProfileSelfiesDecoder,RfamFamilyModel,PubChemDecoder):roles=sequence_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is ProfileGFNPolicy:roles=gfn_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is ProfileLatentSelfiesDecoder:
            if id(module)not in latent:raise ValueError('unbound latent residual source owner')
            roles=_latent_roles(module,*latent[id(module)])
        elif t in (SharedTox21Model,SharedLincsModel):roles=assay_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeGOAHead:roles=goa_mup_parameter_roles(module)
        elif t in (ConditionedColorSemantic,ConditionedColorInstance):roles=conditioned_color_mup_parameter_roles(module,identifier_axis=color_identifier_axis)
        elif t is GeometryConditioner:roles=_geometry_roles(module,identifier_axis=geometry_identifier_axis)
        elif t is SharedAdmetModel:roles=admet_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SourceRoleListModel:roles=source_role_list_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedCartesianGFNPolicy:roles=cartesian_mup_parameter_roles(module)
        elif t is SourceAnnotationObjectives:roles=source_annotation_objectives_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedNativeCellModel:roles=cell_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeSequencePriorModel:roles=native_sequence_prior_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is PDBMaskedCoordinates:roles=pdb_coordinate_parameter_roles(module)
        elif t is RosettaProfileFlow:roles=rosetta_profile_flow_roles(module)
        elif t is SharedNativeScalarModel:roles=source_scalar_mup_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeGrayscaleFlow:roles=native_grayscale_flow_roles(module,geometry_identifier_axis=geometry_identifier_axis)
        elif t is NativeNciAssayModel:roles=nci_assay_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedBioelectricOutcomeModel:roles=bioelectric_outcome_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedHcn4LateralityModel:roles=hcn4_laterality_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is ExpansionRxNativeModel:roles=expansionrx_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeReactZymeProductModel:roles=reactzyme_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedCemaTrajectoryModel:roles=cema_trajectory_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeBETSEMembraneHead:roles=betse_membrane_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedRawCountCellModel:roles=raw_cell_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeTox21SourceModel:roles=tox21_current_model_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeReactomeStaticModel:roles=reactome_static_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeBioGRIDChemicalModel:roles=biogrid_chemical_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedLinCxFetchModel:roles=lincx_fetch_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeStringScoreModel:roles=string_score_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedUniProtNativeAnnotationModel:roles=uniprot_native_annotation_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedUniProtFullRecordModel:roles=uniprot_full_record_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is NativeSCRIModel:roles=native_scri_roles(module,identifier_axis=identifier_axis)
        elif t is NativeForestFlowHead:roles=forest_flow_parameter_roles(module,identifier_axis=geometry_identifier_axis)
        elif t is NativeForestAnalogyHead:roles=forest_analogy_parameter_roles(module,identifier_axis=geometry_identifier_axis)
        elif t is NativeForestTokenDecoder:roles=forest_decoder_parameter_roles(module,identifier_axis=geometry_identifier_axis)
        elif t is NativeForestQualityHead:roles=forest_quality_parameter_roles(module,identifier_axis=geometry_identifier_axis)
        elif t is SharedLincsInverseModel:roles=lincs_inverse_parameter_roles(module,identifier_axis=identifier_axis)
        elif t is SharedSourceTextModel:
            module.get_extra_state()
            if set(module._modules)!={'decoder'} or module._parameters or module._buffers:
                raise ValueError('exact shared source-text decoder wrapper required')
            roles={'decoder.'+n:r for n,r in sequence_mup_parameter_roles(module.decoder,identifier_axis=identifier_axis).items()}
        if roles is not None:merge(module,roles)
        else:
            if module._parameters:raise ValueError('unqualified registered operator parameters: '+type(module).__name__)
            for child in module._modules.values():
                if child is not None:visit(child)
    visit(model)
    aliases=tuple(model.named_parameters(remove_duplicate=False))
    uncovered=[name for name,p in aliases if id(p)not in declarations]
    if uncovered:raise ValueError('unqualified registered parameter aliases: '+str(uncovered[:3]))
    roles={name:declarations[id(p)] for name,p in aliases}
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def _without_member_dimensions(value):
    if type(value)is dict:
        return {k:_without_member_dimensions(v) for k,v in value.items()
                if k not in ('width','feature_width','ffn_width','context_width','channels')}
    if type(value)is tuple:return tuple(_without_member_dimensions(v) for v in value)
    if type(value)is list:return [_without_member_dimensions(v) for v in value]
    return value


def full_owner_mup_width_family(base,delta,target,**axes):
    """Source-equal exact primary members; no inference of missing modalities."""
    ms=(base,delta,target)
    ledgers=[parameter_ledger(m,full_owner_mup_parameter_roles(m,**axes)) for m in ms]
    expected=_without_member_dimensions(base.get_extra_state())
    for m in ms:
        if not exact_wire(_without_member_dimensions(m.get_extra_state()),expected):
            raise ValueError('whole primary source/finite-axis/active-operator/policy configuration drift')
    if base.core.parent.parent.native.response.config['width']!=base.core.parent.parent.native.response.config['mup']['base_width']:
        raise ValueError('actual primary base must match declared base width')
    return width_family(*ledgers)
