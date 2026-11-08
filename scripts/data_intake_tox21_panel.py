"""Original Tox2112-axis observed-call priors; not clinicaltoxicity or dose-response truth."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask
ASSAYS=('NR-AhR','NR-AR','NR-AR-LBD','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53')

def panel(properties):
 values=[];observed=[]
 for assay in ASSAYS:
  value=properties.get(assay)
  if value in (None,'','x'):values.append(None);observed.append(False)
  elif value in ('0','1'):values.append(int(value));observed.append(True)
  else:raise ValueError('unexpected source assay call;retain/quarantine rather than clamp: '+str(value))
 return dict(axis=list(ASSAYS),values=values,observed_mask=observed,observed_calls=sum(observed),call_semantics='sourceoriginalHTSactive1_inactive0;notclinicaltoxicity;missingunknown')

def row_panel(row,final_labels=None):
 if row['source_split']=='source_final_evaluation_structures':
  sid=row['source_properties'].get('Sample ID')
  if final_labels is None or sid not in final_labels:raise ValueError('exact final source Sample ID missing from official label table')
  return panel(final_labels[sid])
 return panel(row['source_properties'])

def tasks(row,p,*,source_ref):
 smi=row.get('derived_canonical_isomeric_smiles')
 if not isinstance(smi,str) or not smi or not p['observed_calls']:return None,None
 common=dict(source_ref=source_ref,source_protocol_role=row['source_split'],source_role='sourceoriginalexperimentalHTSendpointcall_annotationprior;notmeasuredcontinuousresponseorclinicalrisk',source_coordinate_role='SDFdepictionnotexperimentalgeometry;no3Dinputs',physical_dose=None,exposure_time=None,cell_line_context='assayspecificnotprovidedperrow;notfabricated',source_missing_not_negative=True,source_fullproperties_and_molblock='immutable_originalnative_row_ref;FWandFormulaauditcomputedcontextonly',model_adapter_supported=False)
 missing=['perassayphysicaldose_exposuretime_cellline','replicate/batchcondition_resolution','sourcecalibration_thresholdcontext','native_model_Tox21_panelreader','clinical_or_deterministic_doseresponse_claim']
 f=CorpusTask('source_Tox21_molecule_to_observed_assay_calls',dict(canonical_isomeric_smiles=smi,known_assay_axis=list(ASSAYS)),dict(source_call_values=p['values'],source_call_mask=p['observed_mask']),common,missing)
 g=CorpusTask('source_Tox21_observed_call_pattern_to_molecule_prior',dict(known_assay_axis=list(ASSAYS),desired_source_call_values=p['values'],desired_source_call_mask=p['observed_mask']),dict(canonical_isomeric_smiles=smi),dict(common,condition_role='desired_source_observed_callpattern;notguaranteednewassayresponse',excluded_predictors=['molecule_SDFsourceIDs_names_FWFormula_stereoderiveddescriptors_targetlength','sourceoutcome_aggregateQC/citations']),missing)
 return f,g
