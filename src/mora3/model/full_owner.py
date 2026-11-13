"""Source-bound construction of the cooperative owner and explicit extensions.

This replaces fixture constructors as the assembly entry point. Construction,
parameter counts and source identities are evidence, not training admission.
The full research campaign still requires its independent corpus/operator/
recovery/resource checks; this module never issues that approval itself.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re

import torch
from torch import nn

_SHA=re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class PrimaryConfiguration:
    width: int
    base_width: int
    source_snapshot_sha256: str
    readout_mult: float=1.0
    attn_mult: float=1.0
    encoder_var: float=1.0
    flow_version: int=2
    sigma_min: float=0.01

    def __post_init__(self):
        if any(type(v)is not int or v<2 or v%2 for v in(self.width,self.base_width)) or self.width<self.base_width:
            raise ValueError('even positive native widths with width>=base required')
        if type(self.source_snapshot_sha256)is not str or not _SHA.fullmatch(self.source_snapshot_sha256):
            raise ValueError('exact source snapshot SHA256 required')
        if type(self.flow_version)is not int or self.flow_version not in(1,2):
            raise ValueError('explicit native velocity1 or denoiser2 architecture required')
        for value in(self.readout_mult,self.attn_mult,self.encoder_var):
            if type(value)not in(int,float)or not math.isfinite(value)or value<=0:
                raise ValueError('finite positive declared multipliers required')
        if type(self.sigma_min)not in(int,float)or not math.isfinite(self.sigma_min)or not 0<self.sigma_min<1:
            raise ValueError('explicit finite denoiser sigma_min in(0,1)required')


@dataclass(frozen=True)
class SourceAssemblyInputs:
    """Already verified native axes/codecs; no path-based pickle/Hydra loading."""
    vocabulary: object
    annotation_registry: object
    response_axis: tuple
    residue_feature_axis: tuple
    physical_parameter_axis: tuple
    physical_method_axis: tuple
    profile_axes: dict
    lincs_axes: dict
    codec: object
    max_decoder_tokens: int
    rfam_family_axis: tuple
    rfam_manifest_sha256: str
    goa_packets: tuple
    reconstruction_dimension: int
    reconstruction_horizon: int
    reconstruction_hidden: int
    reconstruction_source: dict
    geometry_id_width: int
    color_id_width: int
    instance_slots: int
    instance_max_output_bytes: int
    payload_max_tokens: int
    color_input_policy: str
    instance_slot_policy: str
    admet_contracts: tuple=()
    mcsa_role_vocabulary: object=None
    residue_list_max_target_length: int | None=None
    cell_vocabulary: object=None
    cell_protocol: object=None
    cell_max_batch_size: int=128
    cell_max_positions: int=8_000_000
    source_text_max_target_bytes: int | None=None
    source_text_root_list_policy: str='disabled'
    lincs_inverse_max_target_bytes: int | None=None
    native_prior_max_target_bytes: int | None=None
    raster_source_envelope: object=None
    pubchem_source_root: object=None
    pubchem_manifest: object=None
    pubchem_manifest_sha256: str | None=None
    cartesian_budget: object=None
    cartesian_id_width: int | None=None
    cartesian_horizon: int | None=None
    cartesian_teacher_sha256: str | None=None
    cartesian_input_contract_sha256: str | None=None
    cartesian_backbone_identifier_axis: tuple=('context','observation')
    grayscale_source_envelope: object=None
    scalar_source_axis: object=None
    scalar_max_batch_size: int=128
    pdb_coordinate_axes: object=None
    pdb_max_attention_elements: int=32_000_000
    rosetta_flow_axes: tuple=()
    rosetta_flow_source_census_sha256: str | None=None
    rosetta_flow_native_view_policy_sha256: str | None=None
    grayscale_reverse_detector_dtype: str | None=None
    bioelectric_assay_axes: tuple=()
    bioelectric_source_contract_sha256: str | None=None
    nci_assay_source_axes: object | None=None
    nci_max_batch_size: int=128
    nci_max_total_target_bytes: int | None=None
    expansionrx_source_axes: object | None=None
    expansionrx_max_batch_size: int=128
    expansionrx_max_total_target_bytes: int | None=None
    cema_trajectory_source_sha256: str | None=None
    reactzyme_source_axes: object | None=None
    reactzyme_max_batch_size: int=64
    reactzyme_max_total_target_bytes: int | None=None
    tox21_current_axes: object | None=None
    tox21_max_batch_size: int=128
    tox21_max_total_target_bytes: int | None=None
    hcn4_laterality_source_sha256: str | None=None
    reactome_static_axes: object | None=None
    reactome_max_attention_elements: int=16_000_000
    biogrid_chemical_axes: object | None=None
    biogrid_chemical_max_batch_size: int=128
    lincx_fetch_source_sha256: str | None=None
    string_score_axes: object | None=None
    string_score_max_batch_size: int=64
    string_score_max_batch_sequence_bytes: int | None=None
    uniprot_native_annotation_guard_sha256: str | None=None
    uniprot_native_annotation_max_batch_size: int=128
    uniprot_native_annotation_max_target_bytes: int=64*1024*1024
    uniprot_full_record_axes: object | None=None
    uniprot_full_record_max_batch_size: int=16
    uniprot_full_record_max_target_bytes: int=8*1024*1024
    grayscale_forward_enabled: bool=False
    betse_source: object | None=None
    raw_cell_axis: object | None=None
    scri_axis: object | None=None
    scri_spatial_budget: object | None=None
    forest_flow_spec: object | None=None
    forest_analogy_spec: object | None=None
    forest_decoder_spec: object | None=None
    forest_decoder_codebook: object | None=None
    forest_quality_spec: object | None=None


def _exact(left,right):
    if type(left)is not type(right):return False
    if type(left)is dict:
        return set(left)==set(right)and all(_exact(value,right[key])for key,value in left.items())
    if type(left)in(list,tuple):return len(left)==len(right)and all(_exact(a,b)for a,b in zip(left,right))
    return type(left)in(str,int,float,bool,type(None))and left==right


def unique_parameter_inventory(model):
    """Count identity-tied parameters once; expose aliases and reject view ties.

    This is not an optimizer role assignment. It does not assert that an
    unqualified custom operator has a width-transfer rule.
    """
    if not isinstance(model,nn.Module):raise ValueError('native module required')
    groups={};storage={}
    for name,parameter in model.named_parameters(remove_duplicate=False):
        if type(parameter)is not nn.Parameter or parameter.layout!=torch.strided:
            raise ValueError('materialized dense native parameters required')
        key=id(parameter)
        if key not in groups:
            if parameter.numel():
                source=parameter.untyped_storage()
                storage_key=(str(parameter.device),source.data_ptr())
                if storage_key in storage and storage[storage_key]!=key:
                    raise ValueError('distinct parameter objects sharing storage are unsupported')
                storage[storage_key]=key
            groups[key]=dict(aliases=[],shape=tuple(parameter.shape),numel=parameter.numel(),
                             trainable=parameter.requires_grad,dtype=str(parameter.dtype),device=str(parameter.device))
        groups[key]['aliases'].append(name)
    rows=tuple(dict(row,aliases=tuple(row['aliases']))for row in groups.values())
    return dict(schema='mora3.constructed-owner-unique-counts-v1',parameters=rows,
                unique_parameter_tensors=len(rows),alias_names=sum(len(row['aliases'])for row in rows),
                total_parameters=sum(row['numel']for row in rows),
                trainable_parameters=sum(row['numel']for row in rows if row['trainable']),
                frozen_parameters=sum(row['numel']for row in rows if not row['trainable']),
                mup_operator_qualification=False)


class SourceBoundOwner(nn.Module):
    """One registered owner; extension branches reuse exact cooperative objects."""
    def __init__(self,core,*,source_snapshot_sha256,configuration,admet=None,mcsa=None,
                 pubchem=None,cartesian=None,source_annotations=None,cell=None,text=None,lincs_inverse=None,native_priors=None,scalar=None,pdb=None,rosetta_flow=None,gray_reverse=None,bioelectric=None,nci=None,expansionrx=None,cema_trajectory=None,reactzyme=None,tox21_source=None,hcn4_laterality=None,reactome_static=None,biogrid_chemical=None,lincx_fetch=None,string_score=None,uniprot_native_annotation=None,uniprot_full_record=None,gray_forward=None,betse=None,raw_cell=None,scri=None,forest_flow=None,forest_analogy=None,forest_decoder=None,forest_quality=None):
        super().__init__()
        from .instance_cooperative import InstanceCooperativeModel
        from .admet_model import SharedAdmetModel
        from .mcsa_role_list import SourceRoleListModel
        from .pubchem_decoder import PubChemDecoder
        from .cartesian_gfn_policy import SharedCartesianGFNPolicy
        from .source_annotation_objectives import SourceAnnotationObjectives
        from .cell_native import SharedNativeCellModel
        from .source_text import SharedSourceTextModel
        from .lincs_inverse import SharedLincsInverseModel
        from .native_sequence_priors import NativeSequencePriorModel
        from .source_scalar_objectives import SharedNativeScalarModel
        from .pdb_coordinate_model import PDBMaskedCoordinates
        from .rosetta_profile_flow import RosettaProfileFlow
        from .native_grayscale_flow import NativeGrayscaleFlow
        from .bioelectric_outcomes import SharedBioelectricOutcomeModel
        if (type(core)is not InstanceCooperativeModel or type(source_snapshot_sha256)is not str
                or not _SHA.fullmatch(source_snapshot_sha256)):
            raise ValueError('exact cooperative owner and source snapshot required')
        anchor=core.parent.parent
        if raw_cell is not None:
            from .raw_count_cell_model import SharedRawCountCellModel
            if type(raw_cell) is not SharedRawCountCellModel or raw_cell.shared is not anchor.annotation:
                raise ValueError('raw scCello must reuse the exact native annotation owner')
        if betse is not None:
            from .betse_membrane import NativeBETSEMembraneHead
            if (type(betse)is not NativeBETSEMembraneHead or betse.shared is not anchor.annotation
                    or betse.source.owner_snapshot_sha256!=source_snapshot_sha256):
                raise ValueError('BETSE must bind the exact composed source snapshot and shared annotation owner')
        if (getattr(anchor.native,'parameterization_policy',None)!='mup-cooperative-v1'
                or anchor.annotation.get_extra_state().get('version')!=2
                or getattr(anchor.residue,'parameterization_policy',None)!='mup-residue-v1'):
            raise ValueError('explicit applied native/annotation/residue primary policies required')
        if admet is not None and(type(admet)is not SharedAdmetModel or admet.shared is not anchor.annotation):
            raise ValueError('ADMET must reuse the same native annotation owner')
        if mcsa is not None and(type(mcsa)is not SourceRoleListModel or mcsa.residue is not anchor.residue):
            raise ValueError('MCSA must reuse the same native residue owner')
        if pubchem is not None and(type(pubchem)is not PubChemDecoder
                or pubchem.shared is not anchor.annotation or pubchem.get_extra_state().get('version')!=2):
            raise ValueError('PubChem must reuse the same native annotation owner with explicit sequence policy')
        if cartesian is not None and(type(cartesian)is not SharedCartesianGFNPolicy
                or cartesian.backbone is not anchor.annotation.backbone):
            raise ValueError('Cartesian GFN must reuse the same actual native TokenGT owner')
        if source_annotations is not None and(type(source_annotations)is not SourceAnnotationObjectives
                or source_annotations.annotation is not anchor.annotation
                or source_annotations.rfam is not anchor.rfam
                or source_annotations.residue is not anchor.residue):
            raise ValueError('source annotation objectives must reuse the same actual native owners')
        if cell is not None and(type(cell)is not SharedNativeCellModel or cell.shared is not anchor.annotation):
            raise ValueError('cell must reuse the same actual native annotation owner')
        if text is not None and(type(text)is not SharedSourceTextModel or text.decoder is not pubchem):
            raise ValueError('text must reuse the exact registered native byte decoder')
        if lincs_inverse is not None and(type(lincs_inverse)is not SharedLincsInverseModel
                or lincs_inverse.lincs is not anchor.lincs or lincs_inverse.decoder is not pubchem):
            raise ValueError('LINCS inverse must reuse the exact registered LINCS and byte decoder owners')
        if native_priors is not None and(type(native_priors)is not NativeSequencePriorModel
                or native_priors.nucleotide is not anchor.nucleotide or native_priors.decoder is not pubchem):
            raise ValueError('native priors must reuse the exact registered native-symbol and byte decoder owners')
        if scalar is not None and(type(scalar)is not SharedNativeScalarModel or scalar.shared is not anchor.annotation):
            raise ValueError('native scalar objectives must reuse the exact annotation owner')
        if pdb is not None and(type(pdb)is not PDBMaskedCoordinates or pdb.backbone is not anchor.annotation.backbone):
            raise ValueError('native PDB coordinates must reuse the exact TokenGT owner')
        if rosetta_flow is not None and(type(rosetta_flow)is not RosettaProfileFlow or rosetta_flow.backbone is not anchor.annotation.backbone):
            raise ValueError('native Rosetta profile flow must reuse the exact TokenGT owner')
        if gray_reverse is not None and(type(gray_reverse)is not NativeGrayscaleFlow or gray_reverse.geometry is not anchor.geometry_fusion.geometry):
            raise ValueError('native reverse detector flow must reuse the exact geometry owner')
        if gray_forward is not None:
            from .morphology import MorphologyInstance
            if (type(gray_forward) is not MorphologyInstance or gray_forward.source_envelope is None
                    or gray_forward.parameterization_policy != 'mup-image-v1'
                    or gray_forward.backbone is not anchor.annotation.backbone
                    or (gray_forward.width,gray_forward.base_width,gray_forward.output_mult)
                    != (configuration['width'],configuration['base_width'],configuration['readout_mult'])):
                raise ValueError('Exact source-bound grayscale morphology and shared TokenGT required')
            if gray_reverse is not None and gray_forward.source_envelope != gray_reverse.source_envelope:
                raise ValueError('Forward and reverse grayscale source envelopes differ')
        if bioelectric is not None and(type(bioelectric)is not SharedBioelectricOutcomeModel or bioelectric.shared is not anchor.annotation):
            raise ValueError('bioelectric readout must reuse the exact annotation owner')
        if nci is not None:
            from .nci_assay_model import NativeNciAssayModel
            if type(nci) is not NativeNciAssayModel or nci.decoder is not pubchem:
                raise ValueError('NCI readout must reuse the exact byte decoder')
        if expansionrx is not None:
            from .expansionrx_native_model import ExpansionRxNativeModel
            if (type(expansionrx) is not ExpansionRxNativeModel or expansionrx.admet is not admet
                    or expansionrx.decoder is not pubchem):
                raise ValueError('ExpansionRx must reuse exact ADMET and byte decoder owners')
        if cema_trajectory is not None:
            from .cema_trajectory_model import SharedCemaTrajectoryModel
            if type(cema_trajectory) is not SharedCemaTrajectoryModel or cema_trajectory.shared is not anchor.annotation:
                raise ValueError('CEMA trajectory must reuse the exact registered annotation owner')
        if reactzyme is not None:
            from .reactzyme_product import NativeReactZymeProductModel
            if type(reactzyme) is not NativeReactZymeProductModel or reactzyme.decoder is not pubchem:
                raise ValueError('ReactZyme must reuse the exact registered byte decoder')
        if tox21_source is not None:
            from .tox21_current_model import NativeTox21SourceModel
            if (type(tox21_source) is not NativeTox21SourceModel or tox21_source.tox21 is not anchor.tox21
                    or tox21_source.decoder is not pubchem):
                raise ValueError('native Tox21 must reuse exact registered toxicology and byte owners')
        if hcn4_laterality is not None:
            from .bioelectric_group_counts import SharedHcn4LateralityModel
            if type(hcn4_laterality) is not SharedHcn4LateralityModel or hcn4_laterality.shared is not anchor.annotation:
                raise ValueError('native HCN4 laterality must reuse exact registered annotation owner')
        if reactome_static is not None:
            from .reactome_static_model import NativeReactomeStaticModel
            if type(reactome_static) is not NativeReactomeStaticModel or reactome_static.shared is not anchor.annotation:
                raise ValueError('native Reactome must reuse exact registered annotation owner')
        if biogrid_chemical is not None:
            from .biogrid_chemical_model import NativeBioGRIDChemicalModel
            if type(biogrid_chemical) is not NativeBioGRIDChemicalModel or biogrid_chemical.shared is not anchor.annotation:
                raise ValueError('native BioGRID chemical must reuse exact registered annotation owner')
        if lincx_fetch is not None:
            from .lincx_fetch_model import SharedLinCxFetchModel
            if type(lincx_fetch) is not SharedLinCxFetchModel or lincx_fetch.shared is not anchor.annotation:
                raise ValueError('native LinCx FETCH must reuse exact registered annotation owner')
        if string_score is not None:
            from .string_score_model import NativeStringScoreModel
            if type(string_score) is not NativeStringScoreModel or string_score.shared is not anchor.annotation:
                raise ValueError('native STRING scores must reuse exact registered annotation owner')
        if uniprot_native_annotation is not None:
            from .uniprot_native_objectives import SharedUniProtNativeAnnotationModel
            if type(uniprot_native_annotation) is not SharedUniProtNativeAnnotationModel or uniprot_native_annotation.decoder is not pubchem:
                raise ValueError('native UniProt annotation must reuse exact registered byte decoder')
        if uniprot_full_record is not None:
            from .uniprot_full_record_model import SharedUniProtFullRecordModel
            if type(uniprot_full_record) is not SharedUniProtFullRecordModel or uniprot_full_record.decoder is not pubchem:
                raise ValueError('full-record UniProt must reuse exact registered byte decoder')
        if scri is not None:
            from mora3.extensions.scri.native_owner import NativeSCRIModel
            if (type(scri) is not NativeSCRIModel or scri.core.shared is not anchor.annotation
                    or scri.source_snapshot_sha256!=source_snapshot_sha256):
                raise ValueError('SCRI must reuse exact native annotation owner and source snapshot')
            scri.get_extra_state()
        if forest_flow is not None:
            from .forest_flow import NativeForestFlowHead
            if (type(forest_flow)is not NativeForestFlowHead
                    or forest_flow.geometry is not anchor.geometry_fusion.geometry
                    or forest_flow.spec.source_snapshot_sha256!=source_snapshot_sha256
                    or forest_flow.parameterization!='mup-forest-flow-v1'
                    or (forest_flow.width,forest_flow.base_width,forest_flow.readout_mult)
                    !=(configuration['width'],configuration['base_width'],configuration['readout_mult'])):
                raise ValueError('Forest flow must share the exact native context, source and primary width policy')
            forest_flow.get_extra_state()
        self.forest_flow=forest_flow
        if forest_analogy is not None:
            from .forest_analogy import NativeForestAnalogyHead
            if (forest_flow is None or type(forest_analogy)is not NativeForestAnalogyHead
                    or forest_analogy.field is not forest_flow):
                raise ValueError('Analogy head must share the exact registered forest field')
            forest_analogy.get_extra_state()
        self.forest_analogy=forest_analogy
        if forest_decoder is not None:
            from .forest_token_decoder import NativeForestTokenDecoder
            if (forest_flow is None or type(forest_decoder)is not NativeForestTokenDecoder
                    or forest_decoder.field is not forest_flow):
                raise ValueError('Token decoder must share the exact registered forest field')
            forest_decoder.get_extra_state()
        self.forest_decoder=forest_decoder
        if forest_quality is not None:
            from .forest_quality import NativeForestQualityHead
            if (forest_flow is None or type(forest_quality)is not NativeForestQualityHead or forest_quality.field is not forest_flow):
                raise ValueError('Quality head must share the exact registered forest field')
            forest_quality.get_extra_state()
        self.forest_quality=forest_quality
        self.scri=scri
        self.core=core
        self.admet=admet
        self.mcsa=mcsa
        self.pubchem=pubchem
        self.cartesian=cartesian
        self.source_annotations=source_annotations
        self.cell=cell
        self.text=text
        self.lincs_inverse=lincs_inverse
        self.native_priors=native_priors
        self.scalar=scalar
        self.pdb=pdb
        self.rosetta_flow=rosetta_flow
        self.gray_reverse=gray_reverse
        self.gray_forward=gray_forward
        self.betse=betse
        self.bioelectric=bioelectric
        self.nci=nci
        self.expansionrx=expansionrx
        self.cema_trajectory=cema_trajectory
        self.reactzyme=reactzyme
        self.tox21_source=tox21_source
        self.hcn4_laterality=hcn4_laterality
        self.reactome_static=reactome_static
        self.biogrid_chemical=biogrid_chemical
        self.lincx_fetch=lincx_fetch
        self.string_score=string_score
        self.uniprot_native_annotation=uniprot_native_annotation
        self.uniprot_full_record=uniprot_full_record
        self.raw_cell=raw_cell
        self._owners=(core,admet,mcsa,pubchem,cartesian,source_annotations,cell,text,lincs_inverse,native_priors,scalar,pdb,rosetta_flow,gray_reverse,gray_forward,bioelectric,nci,expansionrx,cema_trajectory,reactzyme,tox21_source,hcn4_laterality,reactome_static,biogrid_chemical,lincx_fetch,string_score,uniprot_native_annotation,uniprot_full_record,betse,raw_cell,scri,forest_flow,forest_analogy,forest_decoder,forest_quality)
        self._configuration=copy.deepcopy(dict(schema='mora3-source-bound-owner-v1',
            primary_arm='ordinary-attention-mup-primary-v1',source_snapshot_sha256=source_snapshot_sha256,
            primary_configuration=configuration,core=core.get_extra_state(),
            admet=None if admet is None else admet.get_extra_state(),
            mcsa=None if mcsa is None else mcsa.get_extra_state(),
            pairing='independent-source-pools-no-invented-crossmodality-joins',training_admission=False))
        if pubchem is not None or cartesian is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v2',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state())
        if source_annotations is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v3',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state(),
                source_annotations=source_annotations.get_extra_state())
        if cell is not None or text is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v4',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state(),
                source_annotations=None if source_annotations is None else source_annotations.get_extra_state(),
                cell=None if cell is None else cell.get_extra_state(),
                text=None if text is None else text.get_extra_state())
        if lincs_inverse is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v5',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state(),
                source_annotations=None if source_annotations is None else source_annotations.get_extra_state(),
                cell=None if cell is None else cell.get_extra_state(),
                text=None if text is None else text.get_extra_state(),lincs_inverse=lincs_inverse.get_extra_state())
        if native_priors is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v6',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state(),
                source_annotations=None if source_annotations is None else source_annotations.get_extra_state(),
                cell=None if cell is None else cell.get_extra_state(),
                text=None if text is None else text.get_extra_state(),
                lincs_inverse=None if lincs_inverse is None else lincs_inverse.get_extra_state(),
                native_priors=native_priors.get_extra_state())
        if scalar is not None or pdb is not None or rosetta_flow is not None or gray_reverse is not None or bioelectric is not None or nci is not None or expansionrx is not None or cema_trajectory is not None or reactzyme is not None or tox21_source is not None or hcn4_laterality is not None or reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v7',
                pubchem=None if pubchem is None else pubchem.get_extra_state(),
                cartesian=None if cartesian is None else cartesian.get_extra_state(),
                source_annotations=None if source_annotations is None else source_annotations.get_extra_state(),
                cell=None if cell is None else cell.get_extra_state(),text=None if text is None else text.get_extra_state(),
                lincs_inverse=None if lincs_inverse is None else lincs_inverse.get_extra_state(),
                native_priors=None if native_priors is None else native_priors.get_extra_state(),
                scalar=None if scalar is None else scalar.get_extra_state(),pdb=None if pdb is None else pdb.get_extra_state())
        if rosetta_flow is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v8',rosetta_flow=rosetta_flow.get_extra_state())
        if gray_reverse is not None or bioelectric is not None or nci is not None or expansionrx is not None or cema_trajectory is not None or reactzyme is not None or tox21_source is not None or hcn4_laterality is not None or reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v9',
                rosetta_flow=None if rosetta_flow is None else rosetta_flow.get_extra_state(),
                gray_reverse=None if gray_reverse is None else gray_reverse.get_extra_state(),
                bioelectric=None if bioelectric is None else bioelectric.get_extra_state())
        if nci is not None or expansionrx is not None or cema_trajectory is not None or reactzyme is not None or tox21_source is not None or hcn4_laterality is not None or reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v10',
                nci=None if nci is None else nci.get_extra_state(),
                expansionrx=None if expansionrx is None else expansionrx.get_extra_state())
        if cema_trajectory is not None or reactzyme is not None or tox21_source is not None or hcn4_laterality is not None or reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v11',
                cema_trajectory=None if cema_trajectory is None else cema_trajectory.get_extra_state(),
                reactzyme=None if reactzyme is None else reactzyme.get_extra_state())
        if tox21_source is not None or hcn4_laterality is not None or reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v12',
                tox21_source=None if tox21_source is None else tox21_source.get_extra_state(),
                hcn4_laterality=None if hcn4_laterality is None else hcn4_laterality.get_extra_state())
        if reactome_static is not None or biogrid_chemical is not None or lincx_fetch is not None or string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v13',
                reactome_static=None if reactome_static is None else reactome_static.get_extra_state(),
                biogrid_chemical=None if biogrid_chemical is None else biogrid_chemical.get_extra_state(),
                lincx_fetch=None if lincx_fetch is None else lincx_fetch.get_extra_state())
        if string_score is not None or uniprot_native_annotation is not None or uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v14',
                string_score=None if string_score is None else string_score.get_extra_state(),
                uniprot_native_annotation=None if uniprot_native_annotation is None else uniprot_native_annotation.get_extra_state())
        if uniprot_full_record is not None:
            self._configuration.update(schema='mora3-source-bound-owner-v15',
                uniprot_full_record=uniprot_full_record.get_extra_state())
        if gray_forward is not None:
            self._configuration['gray_forward']=gray_forward.get_extra_state()
        if betse is not None:
            self._configuration['betse']=betse.get_extra_state()
        if raw_cell is not None:
            self._configuration['raw_cell']=raw_cell.get_extra_state()
        if scri is not None:
            self._configuration['scri']=scri.get_extra_state()
        if forest_flow is not None:
            self._configuration['forest_flow']=forest_flow.get_extra_state()
        if forest_analogy is not None:
            self._configuration['forest_analogy']=forest_analogy.get_extra_state()
        if forest_decoder is not None:
            self._configuration['forest_decoder']=forest_decoder.get_extra_state()
        if forest_quality is not None:
            self._configuration['forest_quality']=forest_quality.get_extra_state()
        self._module_topology=tuple((name,type(module).__module__+'.'+type(module).__qualname__)
            for name,module in self.named_modules(remove_duplicate=False))
        self._parameter_topology=tuple((row['aliases'],row['shape'],row['trainable'])
            for row in unique_parameter_inventory(self)['parameters'])

    def get_extra_state(self):
        if any(a is not b for a,b in zip((self.core,self.admet,self.mcsa,self.pubchem,self.cartesian,self.source_annotations,self.cell,self.text,self.lincs_inverse,self.native_priors,self.scalar,self.pdb,self.rosetta_flow,self.gray_reverse,self.gray_forward,self.bioelectric,self.nci,self.expansionrx,self.cema_trajectory,self.reactzyme,self.tox21_source,self.hcn4_laterality,self.reactome_static,self.biogrid_chemical,self.lincx_fetch,self.string_score,self.uniprot_native_annotation,self.uniprot_full_record,self.betse,self.raw_cell,self.scri,self.forest_flow,self.forest_analogy,self.forest_decoder,self.forest_quality),self._owners)):
            raise ValueError('registered owner identity drift')
        if self._module_topology!=tuple((name,type(module).__module__+'.'+type(module).__qualname__)
                for name,module in self.named_modules(remove_duplicate=False)):
            raise ValueError('registered module topology drift')
        if self._parameter_topology!=tuple((row['aliases'],row['shape'],row['trainable'])
                for row in unique_parameter_inventory(self)['parameters']):
            raise ValueError('registered parameter/alias topology drift')
        current=dict(core=self.core.get_extra_state(),
                     admet=None if self.admet is None else self.admet.get_extra_state(),
                     mcsa=None if self.mcsa is None else self.mcsa.get_extra_state())
        if self._configuration['schema'] in('mora3-source-bound-owner-v2','mora3-source-bound-owner-v3','mora3-source-bound-owner-v4','mora3-source-bound-owner-v5','mora3-source-bound-owner-v6','mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(pubchem=None if self.pubchem is None else self.pubchem.get_extra_state(),
                           cartesian=None if self.cartesian is None else self.cartesian.get_extra_state())
        if self._configuration['schema'] in('mora3-source-bound-owner-v3','mora3-source-bound-owner-v4','mora3-source-bound-owner-v5','mora3-source-bound-owner-v6','mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current['source_annotations']=None if self.source_annotations is None else self.source_annotations.get_extra_state()
        if self._configuration['schema'] in('mora3-source-bound-owner-v4','mora3-source-bound-owner-v5','mora3-source-bound-owner-v6','mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(cell=None if self.cell is None else self.cell.get_extra_state(),
                           text=None if self.text is None else self.text.get_extra_state())
        if self._configuration['schema'] in('mora3-source-bound-owner-v5','mora3-source-bound-owner-v6','mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current['lincs_inverse']=None if self.lincs_inverse is None else self.lincs_inverse.get_extra_state()
        if self._configuration['schema'] in ('mora3-source-bound-owner-v6','mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current['native_priors']=None if self.native_priors is None else self.native_priors.get_extra_state()
        if self._configuration['schema'] in ('mora3-source-bound-owner-v7','mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(scalar=None if self.scalar is None else self.scalar.get_extra_state(),
                           pdb=None if self.pdb is None else self.pdb.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v8','mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current['rosetta_flow']=None if self.rosetta_flow is None else self.rosetta_flow.get_extra_state()
        if self._configuration['schema'] in ('mora3-source-bound-owner-v9','mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(gray_reverse=None if self.gray_reverse is None else self.gray_reverse.get_extra_state(),
                           bioelectric=None if self.bioelectric is None else self.bioelectric.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v10','mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(nci=None if self.nci is None else self.nci.get_extra_state(),
                           expansionrx=None if self.expansionrx is None else self.expansionrx.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v11','mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(cema_trajectory=None if self.cema_trajectory is None else self.cema_trajectory.get_extra_state(),
                           reactzyme=None if self.reactzyme is None else self.reactzyme.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v12','mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(tox21_source=None if self.tox21_source is None else self.tox21_source.get_extra_state(),
                           hcn4_laterality=None if self.hcn4_laterality is None else self.hcn4_laterality.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v13','mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(reactome_static=None if self.reactome_static is None else self.reactome_static.get_extra_state(),
                           biogrid_chemical=None if self.biogrid_chemical is None else self.biogrid_chemical.get_extra_state(),
                           lincx_fetch=None if self.lincx_fetch is None else self.lincx_fetch.get_extra_state())
        if self._configuration['schema'] in ('mora3-source-bound-owner-v14','mora3-source-bound-owner-v15'):
            current.update(string_score=None if self.string_score is None else self.string_score.get_extra_state(),
                           uniprot_native_annotation=None if self.uniprot_native_annotation is None else self.uniprot_native_annotation.get_extra_state())
        if self._configuration['schema']=='mora3-source-bound-owner-v15':
            current.update(uniprot_full_record=self.uniprot_full_record.get_extra_state())
        if self.gray_forward is not None:
            current['gray_forward']=self.gray_forward.get_extra_state()
        if self.betse is not None:
            current['betse']=self.betse.get_extra_state()
        if self.raw_cell is not None:
            current['raw_cell']=self.raw_cell.get_extra_state()
        if self.scri is not None:
            current['scri']=self.scri.get_extra_state()
        if self.forest_flow is not None:
            current['forest_flow']=self.forest_flow.get_extra_state()
        if self.forest_analogy is not None:
            current['forest_analogy']=self.forest_analogy.get_extra_state()
        if self.forest_decoder is not None:
            current['forest_decoder']=self.forest_decoder.get_extra_state()
        if self.forest_quality is not None:
            current['forest_quality']=self.forest_quality.get_extra_state()
        if any(not _exact(value,self._configuration[key])for key,value in current.items()):
            raise ValueError('source-bound assembly component configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not _exact(state,self.get_extra_state()):raise ValueError('source-bound assembly checkpoint mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        # A source-bound campaign resumes existing owners and exact ties. Assign
        # mode replaces Parameter objects and cannot preserve that contract.
        if strict is not True or assign is not False:
            raise ValueError('exact source-bound recovery requires strict=True and assign=False')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _preflight_checkpoint_state(self,state_dict,prefix):
        # Include persistent buffers and conditional policy entries emitted by
        # child _save_to_state_dict overrides, not just module _extra_state.
        # state_dict() does not copy into or mutate any owner tensor.
        expected=self.state_dict()
        if any(type(key)is not str for key in state_dict):
            raise ValueError('literal checkpoint state keys required')
        actual_keys={key for key in state_dict if key.startswith(prefix)}
        expected_keys={prefix+key for key in expected}
        if actual_keys!=expected_keys:
            raise ValueError('exact complete source-bound checkpoint key set required')
        for name,live in expected.items():
            incoming=state_dict[prefix+name]
            if isinstance(live,torch.Tensor):
                if (not isinstance(incoming,torch.Tensor) or incoming.is_meta
                        or incoming.shape!=live.shape or incoming.dtype!=live.dtype
                        or incoming.layout!=live.layout):
                    raise ValueError('complete checkpoint tensor shape/dtype/layout mismatch')
            elif not _exact(incoming,live):
                raise ValueError('complete checkpoint conditional policy state mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        # Validate every source/operator sidecar and tied parameter value before
        # PyTorch can copy a tensor into any shared owner.
        self.get_extra_state()
        if local_metadata.get('assign_to_params_buffers',False) is not False:
            raise ValueError('assignment cannot replace source-bound checkpoint owners')
        self._preflight_checkpoint_state(state_dict,prefix)
        # Validate immutable decoder buffers before any sibling/shared parameter
        # can be copied. A late child refusal would otherwise leave partial state.
        from .forest_token_decoder import NativeForestTokenDecoder
        for name,module in self.named_modules(remove_duplicate=False):
            if type(module)is NativeForestTokenDecoder:
                key=prefix+(name+'.'if name else '')+'codebook'
                if not torch.equal(state_dict[key],module.codebook):
                    raise ValueError('Frozen decoder codebook differs before full-owner recovery')
        for name,module in self.named_modules(remove_duplicate=False):
            if type(module).get_extra_state is nn.Module.get_extra_state:continue
            key=prefix+(name+'.'if name else '')+'_extra_state'
            if key not in state_dict or not _exact(state_dict[key],module.get_extra_state()):
                raise ValueError('complete source-bound checkpoint sidecars required')
        for row in unique_parameter_inventory(self)['parameters']:
            aliases=[prefix+name for name in row['aliases']]
            if any(name not in state_dict for name in aliases):
                raise ValueError('complete source-bound checkpoint parameters required')
            reference=state_dict[aliases[0]]
            if not isinstance(reference,torch.Tensor)or tuple(reference.shape)!=row['shape']:
                raise ValueError('checkpoint parameter shape/type mismatch')
            if any(not isinstance(state_dict[name],torch.Tensor)or not torch.equal(reference,state_dict[name])
                   for name in aliases[1:]):
                raise ValueError('checkpoint values disagree across tied aliases')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def authorize_primary_campaign_training(self, root, *, path, expected_sha256):
        # Only the separately reviewed, source-bound release manager may admit
        # an existing owner. Its approved-release set is currently empty.
        from .full_owner_training_admission import authorize
        authorize(self, root, path=path, expected_sha256=expected_sha256)

    def require_training_admission(self, *, campaign_sha256=None, trial_id=None, window_capacity_sha256=None):
        from .full_owner_training_admission import require
        require(self, campaign_sha256=campaign_sha256, trial_id=trial_id,
                window_capacity_sha256=window_capacity_sha256)


def construct_primary_owner(configuration,sources):
    """Construct the real cooperative chain from native source assets.

    No random seed is changed here. The caller owns and records the seed/RNG
    state. Unqualified heads remain registered/countable, never silently omitted
    or labeled MuP-qualified; the separate full-owner family must validate them.
    """
    if type(configuration)is not PrimaryConfiguration or type(sources)is not SourceAssemblyInputs:
        raise ValueError('exact primary configuration and native source assembly inputs required')
    configuration.__post_init__()
    if sources.forest_quality_spec is not None:
        from .forest_quality import ForestQualitySpec
        if type(sources.forest_quality_spec)is not ForestQualitySpec or sources.forest_flow_spec is None:
            raise ValueError('Native quality specification requires a registered forest field')
        sources.forest_quality_spec.__post_init__()
    if sources.forest_decoder_spec is not None or sources.forest_decoder_codebook is not None:
        from .forest_token_decoder import ForestDecoderSpec,ForestTokenCodebook
        book=sources.forest_decoder_codebook;decoder_spec=sources.forest_decoder_spec
        if type(book)is not ForestTokenCodebook or type(decoder_spec)is not ForestDecoderSpec or sources.forest_flow_spec is None:
            raise ValueError('Decoder requires both exact codebook/specification and forest field')
        book.__post_init__();decoder_spec.__post_init__()
        if (book.source_snapshot_sha256!=configuration.source_snapshot_sha256
                or book.coordinate_sha256!=sources.forest_flow_spec.coordinate_sha256
                or book.axis.base_specification_json!=sources.vocabulary.specification_json
                or book.values.dtype!=torch.float32
                or decoder_spec.max_score_elements<book.values.shape[0]
                or decoder_spec.max_slots*max(configuration.width,book.values.shape[1])>decoder_spec.max_embedding_elements):
            raise ValueError('Decoder exact source, coordinate and base vocabulary must match full owner FP32')
    if sources.forest_analogy_spec is not None:
        from .forest_analogy import ForestAnalogySpec
        if type(sources.forest_analogy_spec)is not ForestAnalogySpec or sources.forest_flow_spec is None:
            raise ValueError('Native analogy specification requires a registered forest flow specification')
        sources.forest_analogy_spec.__post_init__()
    if sources.forest_flow_spec is not None:
        from .forest_flow import ForestFlowSpec
        spec=sources.forest_flow_spec
        if type(spec)is not ForestFlowSpec:
            raise ValueError('Exact native forest source/coordinate specification required')
        spec.__post_init__()
        if (spec.source_snapshot_sha256!=configuration.source_snapshot_sha256
                or type(sources.geometry_id_width)is not int or spec.max_nodes>sources.geometry_id_width):
            raise ValueError('Forest source snapshot or exact identifier rank mismatch')
    if sources.raw_cell_axis is not None:
        from mora3.data.sccello_raw_cell_packets import RawFitAxis
        if (type(sources.raw_cell_axis) is not RawFitAxis
                or sources.raw_cell_axis.evidence_kind not in
                ('original-terminal-readback', 'native-selection-and-independent-fullsort')):
            raise ValueError('verified inactive scCello fitted source axis required')
    if sources.betse_source is not None:
        from mora3.data.betse_membrane import BETSESource
        if (type(sources.betse_source)is not BETSESource
                or sources.betse_source.owner_snapshot_sha256!=configuration.source_snapshot_sha256):
            raise ValueError('Exact opt-in BETSE source and composed owner snapshot required')
    if (sources.scri_axis is None)!=(sources.scri_spatial_budget is None):
        raise ValueError('SCRI source axis and spatial budget must be supplied together')
    if sources.scri_axis is not None:
        from mora3.extensions.scri.model import SCRITargetAxis
        from mora3.extensions.scri.native_owner import SpatialBudget
        if type(sources.scri_axis) is not SCRITargetAxis or type(sources.scri_spatial_budget) is not SpatialBudget:
            raise ValueError('exact SCRI source axis and spatial budget required')
        sources.scri_axis.__post_init__();sources.scri_spatial_budget.__post_init__()
    from mora3.data.annotation_axes import AnnotationAxes
    from .annotation_vocabulary import AnnotationVocabulary
    from .profile_decoder import SelfiesCodec
    if not isinstance(sources.annotation_registry,AnnotationAxes)or type(sources.vocabulary)is not AnnotationVocabulary:
        raise ValueError('verified complete annotation axes and fitted payload vocabulary required')
    if type(sources.codec)is not SelfiesCodec:
        raise ValueError('source-built SELFIES codec required')
    # SciPlex identities are source column positions; repeated labels stay distinct.
    if (type(sources.response_axis)is not tuple or not sources.response_axis
            or any(type(value)is not str or not value for value in sources.response_axis)):
        raise ValueError('complete ordered source response columns required')
    for axis in(sources.residue_feature_axis,sources.physical_parameter_axis,
                sources.physical_method_axis,sources.rfam_family_axis):
        if type(axis)is not tuple or not axis or len(axis)!=len(set(axis))or any(type(v)is not str or not v for v in axis):
            raise ValueError('complete ordered nonempty unique finite axes required')
    for value in(sources.max_decoder_tokens,sources.payload_max_tokens,sources.reconstruction_dimension,
                 sources.reconstruction_horizon,sources.reconstruction_hidden,sources.geometry_id_width,sources.color_id_width,
                 sources.instance_slots,sources.instance_max_output_bytes):
        if type(value)is not int or value<=0:raise ValueError('explicit positive source/runtime bounds required')
    if type(sources.rfam_manifest_sha256)is not str or not _SHA.fullmatch(sources.rfam_manifest_sha256):
        raise ValueError('exact Rfam source manifest SHA required')
    if sources.residue_list_max_target_length is not None and(
            type(sources.residue_list_max_target_length)is not int or sources.residue_list_max_target_length<1):
        raise ValueError('explicit positive complete residue source-list budget required')
    from .cell_native import NativeCellProtocol
    from mora3.data.sccello_native_rows import TokenVocabulary,TOKEN_COUNT
    cell_values=(sources.cell_vocabulary,sources.cell_protocol)
    if any(value is not None for value in cell_values):
        if (type(sources.cell_vocabulary)is not TokenVocabulary or type(sources.cell_protocol)is not NativeCellProtocol
                or any(type(value)is not int or value<1 for value in(sources.cell_max_batch_size,sources.cell_max_positions))):
            raise ValueError('complete cell vocabulary/protocol/resource bounds required')
        sources.cell_protocol.__post_init__()
        keys=sources.cell_vocabulary.gene_keys
        if (type(keys)is not tuple or len(keys)!=TOKEN_COUNT or keys[:2]!=('<pad>','<mask>')
                or any(type(x)is not str or not x for x in keys) or len(set(keys))!=TOKEN_COUNT
                or type(sources.cell_vocabulary.source_sha256)is not str
                or not _SHA.fullmatch(sources.cell_vocabulary.source_sha256)):
            raise ValueError('complete literal cell gene axis required')
    if sources.source_text_max_target_bytes is not None and(
            type(sources.source_text_max_target_bytes)is not int or sources.source_text_max_target_bytes<1
            or sources.pubchem_source_root is None):
        raise ValueError('text requires complete registered byte decoder and positive resource bound')
    if (sources.source_text_root_list_policy not in('disabled','observed-native-root-list-v1')
            or sources.source_text_root_list_policy!='disabled' and sources.source_text_max_target_bytes is None):
        raise ValueError('explicit source root-list policy requires registered text decoder')
    if sources.lincs_inverse_max_target_bytes is not None and(
            type(sources.lincs_inverse_max_target_bytes)is not int or sources.lincs_inverse_max_target_bytes<1
            or sources.pubchem_source_root is None):
        raise ValueError('native LINCS inverse requires registered byte decoder and positive full-target resource bound')
    if sources.native_prior_max_target_bytes is not None and(
            type(sources.native_prior_max_target_bytes)is not int or sources.native_prior_max_target_bytes<1
            or sources.pubchem_source_root is None):
        raise ValueError('native source priors require registered byte decoder and positive complete target budget')
    from .raster_source_envelope import RasterSourceEnvelope
    source_image_policies=('source-envelope-v3','source-layer-list-v4')
    if sources.raster_source_envelope is not None:
        if type(sources.raster_source_envelope)is not RasterSourceEnvelope:
            raise ValueError('exact complete native raster source envelope required')
        sources.raster_source_envelope.__post_init__()
        if (sources.color_input_policy!='source-envelope-v2'
                or sources.instance_slot_policy not in source_image_policies
                or sources.instance_slots!=sources.raster_source_envelope.max_instances):
            raise ValueError('complete source raster/slot policies and exact source layer capacity required')
    elif (sources.color_input_policy=='source-envelope-v2'
            or sources.instance_slot_policy in source_image_policies):
        raise ValueError('source raster and layer policies require complete source envelope')
    if sources.grayscale_source_envelope is not None:
        if type(sources.grayscale_source_envelope)is not RasterSourceEnvelope:
            raise ValueError('exact complete native grayscale source envelope required')
        sources.grayscale_source_envelope.binding()
    if sources.scalar_source_axis is not None:
        from .source_scalar_objectives import SourceScalarAxis
        if (type(sources.scalar_source_axis)is not SourceScalarAxis
                or type(sources.scalar_max_batch_size)is not int or sources.scalar_max_batch_size<1):
            raise ValueError('complete native scalar partition axis and resource bound required')
        sources.scalar_source_axis.__post_init__()
    if sources.pdb_coordinate_axes is not None:
        from mora3.data.pdb_masked_coordinates import PDBCoordinateAxes
        if type(sources.pdb_coordinate_axes)is not PDBCoordinateAxes:
            raise ValueError('exact complete native PDB coordinate axes required')
        sources.pdb_coordinate_axes.binding()
        if type(sources.pdb_max_attention_elements)is not int or sources.pdb_max_attention_elements<4*(sources.pdb_coordinate_axes.max_atoms+1)**2:
            raise ValueError('complete native PDB atom axis exceeds explicit attention budget')
    rosetta_refs=(sources.rosetta_flow_source_census_sha256,sources.rosetta_flow_native_view_policy_sha256)
    if type(sources.rosetta_flow_axes)is not tuple:
        raise ValueError('immutable Rosetta axis tuple required')
    if sources.rosetta_flow_axes or any(x is not None for x in rosetta_refs):
        from mora3.data.rosetta_unconditional_profiles import RosettaProfileAxis
        axes=sources.rosetta_flow_axes
        if (not axes or any(type(axis)is not RosettaProfileAxis for axis in axes)
                or any(type(x)is not str or not _SHA.fullmatch(x)for x in rosetta_refs)
                or len({a.sha256 for a in axes})!=len(axes)):
            raise ValueError('complete literal Rosetta finite axes and source census/view bindings required')
    if sources.grayscale_reverse_detector_dtype is not None and (
            type(sources.grayscale_reverse_detector_dtype)is not str or sources.grayscale_reverse_detector_dtype!='>u2'
            or sources.grayscale_source_envelope is None):
        raise ValueError('complete grayscale source envelope and literal DeepBacs >u2 detector dtype required')
    if type(sources.grayscale_forward_enabled) is not bool or (
            sources.grayscale_forward_enabled and sources.grayscale_source_envelope is None):
        raise ValueError('Explicit grayscale forward policy and exact source envelope required')
    if type(sources.bioelectric_assay_axes)is not tuple:
        raise ValueError('immutable native bioelectric assay axes required')
    if sources.bioelectric_assay_axes or sources.bioelectric_source_contract_sha256 is not None:
        from .bioelectric_outcomes import BioelectricAssayAxis
        axes=sources.bioelectric_assay_axes
        if (not axes or any(type(axis)is not BioelectricAssayAxis for axis in axes)
                or len({axis.key for axis in axes})!=len(axes)
                or type(sources.bioelectric_source_contract_sha256)is not str
                or not _SHA.fullmatch(sources.bioelectric_source_contract_sha256)):
            raise ValueError('complete native bioelectric assay axes and original source contract SHA required')
    from .cartesian_gfn_policy import CartesianBudget
    pubchem_values=(sources.pubchem_source_root,sources.pubchem_manifest,sources.pubchem_manifest_sha256)
    if any(value is not None for value in pubchem_values):
        if (any(value is None for value in pubchem_values)
                or type(sources.pubchem_manifest_sha256)is not str
                or not _SHA.fullmatch(sources.pubchem_manifest_sha256)):
            raise ValueError('complete source-bound PubChem root/manifest/SHA required')
    admet_contracts=sources.admet_contracts
    for axes,batch,total,kind in (
        (sources.nci_assay_source_axes,sources.nci_max_batch_size,sources.nci_max_total_target_bytes,'NCI'),
        (sources.expansionrx_source_axes,sources.expansionrx_max_batch_size,sources.expansionrx_max_total_target_bytes,'ExpansionRx'),
    ):
        if axes is None:
            if type(batch) is not int or batch!=128 or total is not None:
                raise ValueError(f'{kind} bounds require exact native source axes')
            continue
        if kind=='NCI':
            from mora3.data.nci_current_assays import NciAssaySourceAxes
            cls=NciAssaySourceAxes
        else:
            from mora3.data.expansionrx_current import ExpansionRxSourceAxes
            cls=ExpansionRxSourceAxes
        if type(axes) is not cls or sources.pubchem_source_root is None or type(batch) is not int or batch<1:
            raise ValueError(f'complete native {kind} axes, resource bounds and exact byte owner required')
        axes.__post_init__()
        axes.binding()
        minimum=axes.max_target_utf8_bytes if kind=='NCI' else axes.max_connectivity_bytes
        if total is not None and (type(total) is not int or total<minimum):
            raise ValueError(f'{kind} byte budget must admit the largest complete native target')
    if sources.expansionrx_source_axes is not None:
        from .expansionrx_native_model import expansionrx_official_contracts
        from mora3.model.admet_model import AdmetObjectiveContract
        if (type(admet_contracts) is not tuple or
                any(type(c) is not AdmetObjectiveContract for c in admet_contracts)):
            raise ValueError('immutable typed ADMET contracts required for shared ExpansionRx head merge')
        existing={c.partition_key:c for c in admet_contracts}
        if len(existing)!=len(admet_contracts):
            raise ValueError('duplicate existing ADMET partitions')
        for contract in expansionrx_official_contracts(sources.expansionrx_source_axes):
            same=existing.get(contract.partition_key)
            if same is None:
                existing[contract.partition_key]=contract
            elif same!=contract:
                raise ValueError('conflicting ExpansionRx and existing ADMET partition contract')
        admet_contracts=tuple(existing.values())
    if sources.cema_trajectory_source_sha256 is not None:
        from mora3.data.cema_native_trajectories import ARMS_SHA
        if type(sources.cema_trajectory_source_sha256) is not str or sources.cema_trajectory_source_sha256!=ARMS_SHA:
            raise ValueError('exact independent CEMA native source manifest required')
    rx=sources.reactzyme_source_axes;batch=sources.reactzyme_max_batch_size;target=sources.reactzyme_max_total_target_bytes
    if rx is None:
        if type(batch) is not int or batch!=64 or target is not None:
            raise ValueError('ReactZyme bounds require exact native source axes')
    else:
        from mora3.data.reactzyme_current import ReactZymeSourceAxes
        from .reactzyme_product import _stream_bound
        if type(rx) is not ReactZymeSourceAxes or sources.pubchem_source_root is None or type(batch) is not int or batch<1:
            raise ValueError('complete native ReactZyme axes, resource bounds and exact byte owner required')
        rx.__post_init__();rx.binding()
        if (type(sources.payload_max_tokens) is not int or sources.payload_max_tokens<_stream_bound(rx)
                or target is not None and (type(target) is not int or target<rx.max_source_bytes[2])):
            raise ValueError('ReactZyme bounds must admit complete native conditions and targets')
    if sources.hcn4_laterality_source_sha256 is not None:
        from mora3.data.hcn4_native_laterality import LEDGER_SHA
        if type(sources.hcn4_laterality_source_sha256) is not str or sources.hcn4_laterality_source_sha256!=LEDGER_SHA:
            raise ValueError('exact original HCN4 reported group count ledger required')
    tox=sources.tox21_current_axes;batch=sources.tox21_max_batch_size;target=sources.tox21_max_total_target_bytes
    if tox is None:
        if type(batch) is not int or batch!=128 or target is not None:
            raise ValueError('Tox21 bounds require exact native source axes')
    else:
        from mora3.data.tox21_current import Tox21CurrentAxes
        if type(tox) is not Tox21CurrentAxes or sources.pubchem_source_root is None or type(batch) is not int or batch<1:
            raise ValueError('complete native Tox21 axes, resource bounds and exact byte owner required')
        tox.__post_init__();tox.binding()
        if target is not None and (type(target) is not int or target<tox.max_full_smiles_bytes):
            raise ValueError('Tox21 target budget must admit the largest complete native target')
    if sources.lincx_fetch_source_sha256 is not None:
        from mora3.data.lincx_fetch_native_rows import SOURCE_SHA
        if type(sources.lincx_fetch_source_sha256) is not str or sources.lincx_fetch_source_sha256!=SOURCE_SHA:
            raise ValueError('exact original ordered LinCx source table manifest required')
    ra=sources.reactome_static_axes;attention=sources.reactome_max_attention_elements
    if ra is None:
        if type(attention) is not int or attention!=16_000_000:
            raise ValueError('Reactome resource bound requires exact native source axes')
    else:
        from mora3.data.reactome_static_pathways import ReactomeStaticAxes
        from .reactome_static_model import _stream_bound
        if type(ra) is not ReactomeStaticAxes or type(attention) is not int:
            raise ValueError('complete Reactome axes and resource bounds required')
        ra.binding()
        if (attention<4*(ra.max_entities+1)**2 or type(sources.payload_max_tokens) is not int
                or sources.payload_max_tokens<_stream_bound(ra)):
            raise ValueError('Reactome bounds must admit complete native graph and payload')
    bg=sources.biogrid_chemical_axes;bg_batch=sources.biogrid_chemical_max_batch_size
    if bg is None:
        if type(bg_batch) is not int or bg_batch!=128:
            raise ValueError('BioGRID resource bound requires exact native source axes')
    else:
        from mora3.data.biogrid_chemical_current import BioGRIDChemicalAxes
        from .biogrid_chemical_model import _entity_stream_bound
        if type(bg) is not BioGRIDChemicalAxes or type(bg_batch) is not int or bg_batch<1:
            raise ValueError('complete BioGRID native axes and batch bound required')
        bg.binding()
        if type(sources.payload_max_tokens) is not int or sources.payload_max_tokens<_entity_stream_bound(bg):
            raise ValueError('BioGRID bounds must admit complete native input')
    sa=sources.string_score_axes;s_batch=sources.string_score_max_batch_size;s_bytes=sources.string_score_max_batch_sequence_bytes
    if sa is None:
        if type(s_batch) is not int or s_batch!=64 or s_bytes is not None:
            raise ValueError('STRING resource bounds require exact native source axes')
    else:
        from mora3.data.string_score_current import StringScoreAxes
        from .string_score_model import _stream_bound
        if type(sa) is not StringScoreAxes or type(s_batch) is not int or s_batch<1:
            raise ValueError('complete STRING native axes and batch bound required')
        sa.binding()
        if (type(sources.payload_max_tokens) is not int or sources.payload_max_tokens<_stream_bound(sa)
                or s_bytes is not None and (type(s_bytes) is not int or s_bytes<2*sa.max_sequence_utf8_bytes)):
            raise ValueError('STRING bounds must admit complete original sequence pairs')
    ua=sources.uniprot_native_annotation_guard_sha256
    ub=sources.uniprot_native_annotation_max_batch_size;ut=sources.uniprot_native_annotation_max_target_bytes
    if ua is None:
        if type(ub) is not int or ub!=128 or type(ut) is not int or ut!=64*1024*1024:
            raise ValueError('UniProt native annotation resource bounds require exact source guard')
    else:
        from .uniprot_native_objectives import _GUARD,_TARGET_BYTES_BOUND
        if (type(ua) is not str or ua!=_GUARD or sources.pubchem_source_root is None
                or type(ub) is not int or ub<1 or type(ut) is not int or ut<_TARGET_BYTES_BOUND
                or type(sources.payload_max_tokens) is not int or sources.payload_max_tokens<37_000):
            raise ValueError('complete exact guarded UniProt source, decoder and native input/target capacity required')
    fr=sources.uniprot_full_record_axes;fb=sources.uniprot_full_record_max_batch_size
    ft=sources.uniprot_full_record_max_target_bytes
    if fr is None:
        if type(fb) is not int or fb!=16 or type(ft) is not int or ft!=8*1024*1024:
            raise ValueError('full-record UniProt bounds require exact native source axes')
    else:
        from mora3.data.uniprot_full_record import FullRecordUniProtAxes
        from .uniprot_full_record_model import full_record_stream_bound
        if type(fr) is not FullRecordUniProtAxes:
            raise ValueError('complete exact UniProt full-record source axes required')
        fr.binding()
        if (sources.pubchem_source_root is None or type(fb) is not int or fb<1 or type(ft) is not int
                or ft<max(field.max_wire_bytes for field in fr.fields) or type(sources.payload_max_tokens) is not int
                or sources.payload_max_tokens<full_record_stream_bound(fr)):
            raise ValueError('complete full-record UniProt shared decoder and native input/target capacity required')
    cartesian_values=(sources.cartesian_budget,sources.cartesian_id_width,sources.cartesian_horizon,
        sources.cartesian_teacher_sha256,sources.cartesian_input_contract_sha256)
    if any(value is not None for value in cartesian_values):
        if (any(value is None for value in cartesian_values) or type(sources.cartesian_budget)is not CartesianBudget
                or type(sources.cartesian_id_width)is not int or sources.cartesian_id_width<sources.cartesian_budget.max_atoms
                or type(sources.cartesian_horizon)is not int or sources.cartesian_horizon<=0
                or any(type(value)is not str or not _SHA.fullmatch(value)for value in
                    (sources.cartesian_teacher_sha256,sources.cartesian_input_contract_sha256))):
            raise ValueError('complete source Cartesian rank/horizon/teacher/resource contract required')
        sources.cartesian_budget.__post_init__()
        axis=sources.cartesian_backbone_identifier_axis
        if (type(axis)is not tuple or len(axis)!=2
                or any(type(value)is not str or not value for value in axis) or len(set(axis))!=2):
            raise ValueError('complete ordered source backbone identifier axis required')
    from .cooperative_training import CooperativeModel
    from .physical_trainer import SharedPhysicalModel
    from .lincs_cooperative import LincsCooperativeModel
    from .color_cooperative import ColorCooperativeModel
    from .instance_cooperative import InstanceCooperativeModel
    numeric=dict(base_width=configuration.base_width,readout_mult=float(configuration.readout_mult),
                 attn_mult=float(configuration.attn_mult),encoder_var=float(configuration.encoder_var))
    native=CooperativeModel(sources.response_axis,width=configuration.width,
                            parameterization_policy='mup-cooperative-v1',
                            grayscale_source_envelope=sources.grayscale_source_envelope,**numeric)
    annotation=SharedPhysicalModel.from_registry(sources.vocabulary,sources.annotation_registry,
        width=configuration.width,heads=2,layers=2,max_tokens=sources.payload_max_tokens,
        parameter_axis=sources.physical_parameter_axis,method_axis=sources.physical_method_axis,
        backbone=native.response.backbone,parameterization_policy='mup-annotation-v1',**numeric)
    parent=LincsCooperativeModel(native,annotation,sources.residue_feature_axis,sources.profile_axes,
        codec=sources.codec,max_tokens=sources.max_decoder_tokens,
        annotation_registry=sources.annotation_registry,family_axis=sources.rfam_family_axis,
        rfam_manifest_sha256=sources.rfam_manifest_sha256,goa_packets=sources.goa_packets,
        reconstruction_dimension=sources.reconstruction_dimension,
        reconstruction_horizon=sources.reconstruction_horizon,reconstruction_hidden=sources.reconstruction_hidden,
        reconstruction_source=sources.reconstruction_source,geometry_id_width=sources.geometry_id_width,
        lincs_axes=sources.lincs_axes,flow_version=configuration.flow_version,sigma_min=configuration.sigma_min,
        parameterization_policy='mup-cooperative-v1',**numeric)
    # New policies must propagate through the production color/instance owner
    # constructors; constructing legacy heads and relabeling is prohibited.
    color=ColorCooperativeModel(parent,color_id_width=sources.color_id_width,
                               input_policy=sources.color_input_policy,
                               source_envelope=sources.raster_source_envelope,
                               parameterization_policy='mup-cooperative-v1')
    core=InstanceCooperativeModel(color,slots=sources.instance_slots,
        max_output_bytes=sources.instance_max_output_bytes,slot_policy=sources.instance_slot_policy,
        source_envelope=sources.raster_source_envelope,
        parameterization_policy='mup-cooperative-v1')
    admet=mcsa=None
    if admet_contracts:
        from .admet_model import SharedAdmetModel
        admet=SharedAdmetModel(annotation,admet_contracts,parameterization_policy='mup-admet-v1')
    if sources.mcsa_role_vocabulary is not None:
        from .mcsa_role_list import SourceRoleListModel
        mcsa=SourceRoleListModel(parent.residue,sources.mcsa_role_vocabulary,max_target_length=75)
    pubchem=cartesian=None
    if sources.pubchem_source_root is not None:
        from .pubchem_decoder import PubChemDecoder
        pubchem=PubChemDecoder(annotation,source_root=sources.pubchem_source_root,
            manifest=sources.pubchem_manifest,expected_manifest_sha256=sources.pubchem_manifest_sha256,
            parameterization_policy='mup-sequence-v1')
    if sources.cartesian_budget is not None:
        from .cartesian_gfn_policy import SharedCartesianGFNPolicy
        cartesian=SharedCartesianGFNPolicy(annotation.backbone,id_width=sources.cartesian_id_width,
            horizon=sources.cartesian_horizon,budget=sources.cartesian_budget,
            teacher_sha256=sources.cartesian_teacher_sha256,
            input_condition_contract_sha256=sources.cartesian_input_contract_sha256,
            readout_mult=configuration.readout_mult,
            backbone_identifier_axis=sources.cartesian_backbone_identifier_axis)
    source_annotations=None
    if sources.residue_list_max_target_length is not None:
        from .source_annotation_objectives import SourceAnnotationObjectives
        source_annotations=SourceAnnotationObjectives(annotation,parent.rfam,parent.residue,
            residue_max_target_length=sources.residue_list_max_target_length)
    cell=text=None
    if sources.cell_vocabulary is not None:
        from .cell_native import SharedNativeCellModel
        cell=SharedNativeCellModel(annotation,sources.cell_vocabulary,sources.cell_protocol,
            max_batch_size=sources.cell_max_batch_size,max_positions=sources.cell_max_positions)
    if sources.source_text_max_target_bytes is not None:
        from .source_text import SharedSourceTextModel
        text=SharedSourceTextModel(pubchem,max_total_target_bytes=sources.source_text_max_target_bytes,
            root_list_policy=sources.source_text_root_list_policy)
    lincs_inverse=None
    if sources.lincs_inverse_max_target_bytes is not None:
        from .lincs_inverse import SharedLincsInverseModel
        lincs_inverse=SharedLincsInverseModel(parent.lincs,pubchem,
            max_total_target_bytes=sources.lincs_inverse_max_target_bytes)
    native_priors=None
    if sources.native_prior_max_target_bytes is not None:
        from .native_sequence_priors import NativeSequencePriorModel
        native_priors=NativeSequencePriorModel(parent.nucleotide,pubchem,
            max_total_target_bytes=sources.native_prior_max_target_bytes)
    scalar=pdb=None
    if sources.scalar_source_axis is not None:
        from .source_scalar_objectives import SharedNativeScalarModel
        scalar=SharedNativeScalarModel(annotation,sources.scalar_source_axis,max_batch_size=sources.scalar_max_batch_size)
    if sources.pdb_coordinate_axes is not None:
        from .pdb_coordinate_model import PDBMaskedCoordinates
        pdb=PDBMaskedCoordinates(annotation.backbone,sources.pdb_coordinate_axes,readout_mult=configuration.readout_mult,
            max_attention_elements=sources.pdb_max_attention_elements)
    rosetta_flow=None
    if sources.rosetta_flow_axes:
        from .rosetta_profile_flow import RosettaProfileFlow
        rosetta_flow=RosettaProfileFlow(annotation.backbone,sources.rosetta_flow_axes,
            source_census_sha256=sources.rosetta_flow_source_census_sha256,
            native_view_policy_sha256=sources.rosetta_flow_native_view_policy_sha256,
            readout_mult=configuration.readout_mult,sigma_min=configuration.sigma_min)
    gray_reverse=bioelectric=None
    if sources.grayscale_reverse_detector_dtype is not None:
        from .native_grayscale_flow import NativeGrayscaleFlow
        gray_reverse=NativeGrayscaleFlow(parent.geometry_fusion.geometry,sources.grayscale_source_envelope,
            detector_dtype=sources.grayscale_reverse_detector_dtype,readout_mult=configuration.readout_mult,
            sigma_min=configuration.sigma_min)
    if sources.bioelectric_assay_axes:
        from .bioelectric_outcomes import SharedBioelectricOutcomeModel
        bioelectric=SharedBioelectricOutcomeModel(annotation,sources.bioelectric_assay_axes,
            source_contract_manifest_sha256=sources.bioelectric_source_contract_sha256)
    nci=expansionrx=None
    if sources.nci_assay_source_axes is not None:
        from .nci_assay_model import NativeNciAssayModel
        nci=NativeNciAssayModel(pubchem,sources.nci_assay_source_axes,
            max_batch_size=sources.nci_max_batch_size,max_total_target_bytes=sources.nci_max_total_target_bytes)
    if sources.expansionrx_source_axes is not None:
        from .expansionrx_native_model import ExpansionRxNativeModel
        expansionrx=ExpansionRxNativeModel(admet,pubchem,sources.expansionrx_source_axes,
            max_batch_size=sources.expansionrx_max_batch_size,max_total_target_bytes=sources.expansionrx_max_total_target_bytes)
    cema_trajectory=reactzyme=None
    if sources.cema_trajectory_source_sha256 is not None:
        from .cema_trajectory_model import SharedCemaTrajectoryModel
        cema_trajectory=SharedCemaTrajectoryModel(annotation)
    if sources.reactzyme_source_axes is not None:
        from .reactzyme_product import NativeReactZymeProductModel
        reactzyme=NativeReactZymeProductModel(pubchem,sources.reactzyme_source_axes,
            max_batch_size=sources.reactzyme_max_batch_size,max_total_target_bytes=sources.reactzyme_max_total_target_bytes)
    tox21_source=hcn4_laterality=None
    if sources.tox21_current_axes is not None:
        from .tox21_current_model import NativeTox21SourceModel
        tox21_source=NativeTox21SourceModel(parent.tox21,pubchem,sources.tox21_current_axes,
            max_batch_size=sources.tox21_max_batch_size,max_total_target_bytes=sources.tox21_max_total_target_bytes)
    if sources.hcn4_laterality_source_sha256 is not None:
        from .bioelectric_group_counts import SharedHcn4LateralityModel
        hcn4_laterality=SharedHcn4LateralityModel(annotation)
    reactome_static=biogrid_chemical=lincx_fetch=None
    if sources.reactome_static_axes is not None:
        from .reactome_static_model import NativeReactomeStaticModel
        reactome_static=NativeReactomeStaticModel(annotation,sources.reactome_static_axes,
            max_attention_elements=sources.reactome_max_attention_elements)
    if sources.biogrid_chemical_axes is not None:
        from .biogrid_chemical_model import NativeBioGRIDChemicalModel
        biogrid_chemical=NativeBioGRIDChemicalModel(annotation,sources.biogrid_chemical_axes,
            max_batch_size=sources.biogrid_chemical_max_batch_size)
    if sources.lincx_fetch_source_sha256 is not None:
        from .lincx_fetch_model import SharedLinCxFetchModel
        lincx_fetch=SharedLinCxFetchModel(annotation)
    string_score=uniprot_native_annotation=None
    if sources.string_score_axes is not None:
        from .string_score_model import NativeStringScoreModel
        string_score=NativeStringScoreModel(annotation,sources.string_score_axes,
            max_batch_size=sources.string_score_max_batch_size,max_batch_sequence_bytes=sources.string_score_max_batch_sequence_bytes)
    if sources.uniprot_native_annotation_guard_sha256 is not None:
        from .uniprot_native_objectives import SharedUniProtNativeAnnotationModel
        uniprot_native_annotation=SharedUniProtNativeAnnotationModel(pubchem,max_batch_size=sources.uniprot_native_annotation_max_batch_size,
            max_total_target_bytes=sources.uniprot_native_annotation_max_target_bytes)
    uniprot_full_record=None
    if sources.uniprot_full_record_axes is not None:
        from .uniprot_full_record_model import SharedUniProtFullRecordModel
        uniprot_full_record=SharedUniProtFullRecordModel(pubchem,sources.uniprot_full_record_axes,
            max_batch_size=sources.uniprot_full_record_max_batch_size,
            max_total_target_bytes=sources.uniprot_full_record_max_target_bytes)
    gray_forward=None
    if sources.grayscale_forward_enabled:
        from .morphology import MorphologyInstance
        gray_forward=MorphologyInstance(configuration.width,normalization='unsigned-dtype-max',
            parameterization_policy='mup-image-v1',base_width=configuration.base_width,
            output_mult=configuration.readout_mult,attn_mult=configuration.attn_mult,
            encoder_var=configuration.encoder_var,backbone=annotation.backbone,
            source_envelope=sources.grayscale_source_envelope)
    scri=None
    if sources.scri_axis is not None:
        from mora3.extensions.scri.native_owner import NativeSCRIModel
        scri=NativeSCRIModel(annotation,sources.scri_axis,sources.scri_spatial_budget,configuration.source_snapshot_sha256)
    betse=None
    if sources.betse_source is not None:
        from .betse_membrane import NativeBETSEMembraneHead
        betse=NativeBETSEMembraneHead(annotation,sources.betse_source)
    raw_cell=None
    if sources.raw_cell_axis is not None:
        from .raw_count_cell_model import SharedRawCountCellModel
        raw_cell=SharedRawCountCellModel(annotation,sources.raw_cell_axis)
    forest_flow=None
    if sources.forest_flow_spec is not None:
        from .forest_flow import NativeForestFlowHead
        forest_flow=NativeForestFlowHead(parent.geometry_fusion.geometry,sources.forest_flow_spec,
            readout_mult=configuration.readout_mult)
    forest_analogy=None
    if sources.forest_analogy_spec is not None:
        from .forest_analogy import NativeForestAnalogyHead
        forest_analogy=NativeForestAnalogyHead(forest_flow,sources.forest_analogy_spec)
    forest_decoder=None
    if sources.forest_decoder_spec is not None:
        from .forest_token_decoder import NativeForestTokenDecoder
        forest_decoder=NativeForestTokenDecoder(forest_flow,sources.forest_decoder_codebook,sources.forest_decoder_spec)
    forest_quality=None
    if sources.forest_quality_spec is not None:
        from .forest_quality import NativeForestQualityHead
        forest_quality=NativeForestQualityHead(forest_flow,sources.forest_quality_spec)
    declared=dict(width=configuration.width,heads=2,layers=2,
        **numeric,flow_version=configuration.flow_version,sigma_min=configuration.sigma_min,
        payload_vocabulary_sha256=hashlib.sha256(sources.vocabulary.specification_json.encode()).hexdigest())
    return SourceBoundOwner(core,source_snapshot_sha256=configuration.source_snapshot_sha256,
                            configuration=declared,admet=admet,mcsa=mcsa,pubchem=pubchem,cartesian=cartesian,
                            source_annotations=source_annotations,cell=cell,text=text,lincs_inverse=lincs_inverse,native_priors=native_priors,scalar=scalar,pdb=pdb,rosetta_flow=rosetta_flow,
                            gray_reverse=gray_reverse,gray_forward=gray_forward,bioelectric=bioelectric,nci=nci,expansionrx=expansionrx,cema_trajectory=cema_trajectory,reactzyme=reactzyme,
                            tox21_source=tox21_source,hcn4_laterality=hcn4_laterality,
                            reactome_static=reactome_static,biogrid_chemical=biogrid_chemical,lincx_fetch=lincx_fetch,
                            string_score=string_score,uniprot_native_annotation=uniprot_native_annotation,uniprot_full_record=uniprot_full_record,betse=betse,raw_cell=raw_cell,scri=scri,forest_flow=forest_flow,forest_analogy=forest_analogy,forest_decoder=forest_decoder,forest_quality=forest_quality)
