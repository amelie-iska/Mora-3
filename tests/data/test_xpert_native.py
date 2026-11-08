"""X-Pert native candidate contracts; synthetic core cases are explicitly labeled."""
from dataclasses import replace
import hashlib,importlib.util,json,os,struct,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(os.environ.get("MORA3_REPO_ROOT",Path.cwd()))
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def load_api():
    p=Path(os.environ.get("MORA3_XPERT_MODULE","src/mora3/data/xpert_native.py"))
    if not p.is_file():pytest.fail("X-Pert native candidate interface is not implemented")
    spec=importlib.util.spec_from_file_location("mora3.data.xpert_native",p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

@pytest.fixture(scope="module")
def api():return load_api()

def profile(api,words=(0x3f800000,0x80000000,0x00000000),encoding="dense"):
    # Synthetic IEEE754 oracle:1.0, explicit negative0.0, implicit positive0.0.
    stored=b"\1\1\1" if encoding=="dense" else b"\1\1\0"
    positions=struct.pack("<qqq",4,9,12 if encoding=="dense" else -1)
    return api.Float32Profile(struct.pack("<III",*words),b"\1"*3,b"\1"*3,stored,positions,encoding)

def cat(value,code=0):return dict(c=code,v=value)

def genetic_fields():
    # Synthetic context; all numeric QC values are withheld outcomes, not inputs.
    return dict(_index="hidden-barcode",guide_id=cat("GENE1_GENE2;GENE1_GENE2"),
        perturbation=cat("GENE1_GENE2"),nperts=2,perturbation_type=cat("CRISPR"),
        cell_line=cat("K562"),organism=cat("human"),celltype=cat("lymphoblasts"),
        tissue_type=cat("cell_line"),disease=cat("source context"),cancer=True,
        read_count=12,UMI_count=3,coverage={"t":"<f8","hex":"000000000000f03f","v":1.0},
        good_coverage=True,number_of_cells=1,ngenes=9,ncounts={"t":"<f8","hex":"0000000000002440","v":10.0},
        percent_mito={"t":"<f8","hex":"0000000000000000","v":0.0},
        percent_ribo={"t":"<f8","hex":"0000000000000000","v":0.0},gemgroup=2)

def chemical_fields(dose="10.0",unit="um"):
    return dict(_index="0",inst_id="hidden-inst",rna_plate=cat("hidden-plate"),rna_well=cat("hidden-well"),
        pert_id=cat("hidden-BRD"),pert_iname=cat("hidden-name"),pert_type=cat("trt_cp"),
        pert_dose=cat(dose),pert_dose_unit=cat(unit),pert_time=6,pert_time_unit=cat("h"),
        cell_id=cat("A375"),canonical_smiles=cat("CCO"),plate_id=cat("hidden-study"))

def context(api,article,fields):
    return api.XPertContext(article,canonical(fields))

def test_synthetic_float32_bits_are_immutable_not_float64_reserialized(api):
    p=profile(api)
    assert p.values_bytes==struct.pack("<III",0x3f800000,0x80000000,0)
    assert p.native_values().dtype.str=="<f4"
    assert p.native_values().flags.writeable is False
    assert p.native_values().tobytes()==p.values_bytes
    assert np.signbit(p.native_values()[1])
    with pytest.raises(ValueError):p.native_values()[0]=5

def test_synthetic_sparse_implicit_zero_is_observed_not_missing_and_explicit_negative_zero_kept(api):
    p=profile(api,encoding="csc_matrix")
    assert p.observed_mask==b"\1"*3
    assert p.explicit_storage_mask==b"\1\1\0"
    assert p.source_positions()==(4,9,-1)
    assert p.native_values().tobytes()==struct.pack("<III",0x3f800000,0x80000000,0)

@pytest.mark.parametrize("mutation",("dtypebytes","mask_length","mask_alias","missing_axis","finite","storage","implicit_negative_zero","positions","duplicate_positions","encoding"))
def test_synthetic_profile_axis_masks_positions_and_dtype_must_be_exact(api,mutation):
    p=profile(api,encoding="csc_matrix");kw={}
    if mutation=="dtypebytes":kw["values_bytes"]=p.values_bytes+b"\0"
    elif mutation=="mask_length":kw["observed_mask"]=b"\1"
    elif mutation=="mask_alias":kw["observed_mask"]=(True,True,True)
    elif mutation=="missing_axis":kw["observed_mask"]=b"\1\1\0"
    elif mutation=="finite":kw["finite_mask"]=b"\0\1\1"
    elif mutation=="storage":kw["explicit_storage_mask"]=b"\1\1\2"
    elif mutation=="implicit_negative_zero":kw["values_bytes"]=struct.pack("<III",0x3f800000,0x80000000,0x80000000)
    elif mutation=="positions":kw["source_position_bytes"]=struct.pack("<qqq",4,9,3)
    elif mutation=="duplicate_positions":kw["source_position_bytes"]=struct.pack("<qqq",4,4,-1)
    else:kw["matrix_encoding"]="csr_reordered"
    with pytest.raises(ValueError):replace(p,**kw)

def test_synthetic_nonfinite_native_bits_retained_with_distinct_finite_mask_not_imputed(api):
    words=(0x7fc01234,0x7f800000,0)
    p=api.Float32Profile(struct.pack("<III",*words),b"\1"*3,b"\0\0\1",b"\1"*3,
        struct.pack("<qqq",0,1,2),"dense")
    assert p.values_bytes==struct.pack("<III",*words)
    assert p.finite_mask==b"\0\0\1"
    assert p.observed_mask==b"\1"*3

@pytest.mark.parametrize("article", (True,32221275.0,0))
def test_context_source_article_types_are_exact(api,article):
    with pytest.raises(ValueError):context(api,article,genetic_fields())

def test_synthetic_genetic_combination_identity_is_known_but_placeholder_not_efficacy(api):
    c=context(api,32221275,genetic_fields())
    x=c.prediction_inputs()
    assert x["native_genetic_perturbation"]=="GENE1_GENE2"
    assert x["native_perturbation_count"]==2
    assert x["cell_line"]=="K562" and x["organism"]=="human"
    assert x["guide_efficacy"] is None and x["guide_efficacy_known"] is False
    assert "guide_id" not in x
    with pytest.raises(ValueError,match="placeholder|unavailable"):c.require_guide_efficacy_target()

@pytest.mark.parametrize("dose,unit,known",(("10.0","um",True),("0.1","%",True),("-666","-666",False),("","um",False)))
def test_synthetic_native_dose_units_and_sentinel_are_never_converted(api,dose,unit,known):
    c=context(api,32221302,chemical_fields(dose,unit));x=c.prediction_inputs()
    d=x["native_dose_coordinate"]
    assert d["value"]==dose and d["unit"]==unit
    assert d["coordinate_known"] is known
    assert d["physical_unit_confirmed"] is False and d["conversion_allowed"] is False
    assert x["native_exposure_time"]==dict(value=6,unit="h",coordinate_known=True)
    assert x["native_source_smiles"]=="CCO"

def test_synthetic_context_missing_categorical_value_is_distinct_from_empty_literal(api):
    a=chemical_fields();a["cell_id"]=cat(None,-1)
    b=chemical_fields();b["cell_id"]=cat("",0)
    assert context(api,32221302,a).prediction_inputs()["cell_line"] is None
    assert context(api,32221302,b).prediction_inputs()["cell_line"]==""
    with pytest.raises(ValueError):
        a["cell_id"]=cat("invented",False);context(api,32221302,a)

@pytest.mark.parametrize("field",("read_count","UMI_count","coverage","good_coverage","number_of_cells","ngenes","ncounts","percent_mito","percent_ribo","gemgroup","_index"))
def test_synthetic_postmeasurement_QC_and_source_ID_changes_do_not_enter_genetic_inputs(api,field):
    a=genetic_fields();b=genetic_fields();b[field]="hidden-mutated-target-or-audit"
    assert context(api,32221275,a).prediction_inputs()==context(api,32221275,b).prediction_inputs()

@pytest.mark.parametrize("field",("inst_id","rna_plate","rna_well","pert_id","pert_iname","plate_id","_index"))
def test_synthetic_chemical_identifiers_and_selection_audit_do_not_enter_inputs(api,field):
    a=chemical_fields();b=chemical_fields();b[field]="hidden-mutated-target-or-audit"
    assert context(api,32221302,a).prediction_inputs()==context(api,32221302,b).prediction_inputs()

@pytest.mark.parametrize("key",("native_expression","pvals","adjusted_pvals","scores","logfoldchanges","DE_gene_list","control_barcode","mean_untreated","sgRNA_new","target_length","target_mask","gold_context"))
def test_context_input_allowlist_refuses_placeholder_outcome_control_selection_and_target_fields(api,key):
    c=context(api,32221275,genetic_fields())
    with pytest.raises(ValueError):c.prediction_inputs(known_fields=(key,))

@pytest.mark.parametrize("purpose",("DE_statistics","gene_ranking","all_gene_DE_supervision"))
def test_protocol_point1_and_all_gene_DE_placeholders_never_become_targets(api,purpose):
    with pytest.raises(ValueError,match="placeholder"):api.require_DE_supervision(purpose)

def test_synthetic_control_roles_never_imply_samecell_pairing_or_measured_aggregate(api):
    for role in ("native_population_control","recipe_random_with_replacement_pool","recipe_synthetic_untreated_mean"):
        c=api.ControlSemantics(role)
        assert c.same_cell_before_after is False
        assert c.independent_experimental_measurement_count is None
        assert c.is_synthetic_aggregate is (role=="recipe_synthetic_untreated_mean")
    with pytest.raises(ValueError):api.ControlSemantics("same_cell_baseline")

@pytest.mark.parametrize("rows",((),(0,0),[0],(True,),(-1,),tuple(range(17))))
def test_native_selectors_are_exact_bounded_unique_not_splits(api,rows):
    with pytest.raises(ValueError):api.XPertSelector(32221275,rows)

@pytest.mark.parametrize("purpose",("training","evaluation","response_loss","retrieval","split_use"))
def test_no_generic_boolean_digest_or_schema_can_admit_native_candidate(api,purpose):
    for proof in (None,True,{"verified":True,"sha256":"0"*64,"schema":"admitted"}):
        with pytest.raises(ValueError,match="admission|opaque|unavailable"):
            api.require_XPert_model_use(purpose,admission=proof)

FIXTURE=Path("outputs/data-sources/20261101-XPert-source-v1/curated/bounded-native-reader-fixture-v1/handoff-bounded-native-reader-fixture-v2-role-masks.json")
FIXTURE_SHA="fbec896835fb84516447bebc4a928c4b06686796c1664e35835d4acf26f6388b"

@pytest.fixture(scope="module")
def native_source(api):
    if not (ROOT/FIXTURE).is_file():pytest.skip("source-owner safe native fixture unavailable")
    return api.XPertNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=FIXTURE_SHA)

@pytest.fixture(scope="module")
def real_cases(api,native_source):
    out={}
    for article,rows in ((32221275,(0,3,28)),(32221302,(0,1,590))):
        out[article]=native_source.read_profiles(api.XPertSelector(article,rows))
    return out

@pytest.mark.parametrize("article,n",((32221275,33694),(32221302,978)))
def test_real_safe_profiles_preserve_complete_gene_axis_float32_bits_native_rows_and_masks(
 api,native_source,real_cases,article,n):
    cases=real_cases[article]
    for v in cases:
        assert v.gene_axis.axis_length==v.profile.axis_length==n
        assert len(v.gene_axis.ordered_labels)==len(set(v.gene_axis.ordered_labels))==n
        assert v.profile.native_values().dtype.str=="<f4"
        assert v.profile.observed_mask==v.profile.finite_mask==b"\1"*n
        assert native_source.verify(v)==v
        assert v.status=="native_expression_candidate_no_task_admission"
        assert v.official_split is None
        assert v.audit()["biological_samecell_link"] is None
        assert v.audit()["independent_experimental_measurement_count"] is None
        assert "native_expression" not in v.prediction_inputs()

def test_real_source_roles_do_not_merge_single_combo_control_or_bulk_instances(real_cases):
    single,combo,control=real_cases[32221275]
    assert single.prediction_inputs()["native_genetic_perturbation"]=="ARID1A"
    assert combo.prediction_inputs()["native_genetic_perturbation"]=="SET_KLF1"
    assert control.prediction_inputs()["native_perturbation_count"]==0
    assert control.control_semantics.role=="native_population_control"
    a,b,vehicle=real_cases[32221302]
    assert a.prediction_inputs()==b.prediction_inputs()
    assert a.binding_sha256!=b.binding_sha256 and a.profile.values_bytes!=b.profile.values_bytes
    assert vehicle.prediction_inputs()["native_dose_coordinate"]==dict(value="-666.0",unit="-666",
        coordinate_known=False,physical_unit_confirmed=False,conversion_allowed=False)
    assert vehicle.prediction_inputs()["native_source_smiles"]=="CS(=O)C"
    assert vehicle.audit()["source_reference_graph"]["canonical_isomeric_SMILES"]=="CS(C)=O"
    assert vehicle.control_semantics.same_cell_before_after is False
    assert a.control_semantics is None

def test_real_sparse_zero_and_native_storage_positions_stay_distinct_from_missing(real_cases):
    for v in real_cases[32221275]:
        stored=np.frombuffer(v.profile.explicit_storage_mask,dtype=bool)
        values=v.profile.native_values()
        assert stored.any() and (~stored).any()
        assert np.all(values[~stored]==0) and not np.signbit(values[~stored]).any()
        assert np.all(np.asarray(v.profile.source_positions())[~stored]==-1)
        assert len(set(np.asarray(v.profile.source_positions())[stored].tolist()))==int(stored.sum())
    for v in real_cases[32221302]:
        assert v.profile.explicit_storage_mask==b"\1"*978
        assert v.profile.source_positions()==tuple(range(v.native_row_index*978,(v.native_row_index+1)*978))

@pytest.mark.parametrize("article", (32221275,32221302))
def test_real_target_and_QC_mutation_cannot_change_predictors_but_fresh_source_replay_refuses(
 api,native_source,real_cases,article):
    v=real_cases[article][0];fields=v.context.native_fields()
    fields["_index"]="hidden-mutated-audit"
    if article==32221275:fields["ngenes"]=999999
    altered_context=replace(v.context,native_fields_bytes=canonical(fields))
    assert altered_context.prediction_inputs()==v.prediction_inputs()
    with pytest.raises(ValueError):
        changed=replace(v,context=altered_context)
        native_source.verify(changed)
    raw=bytearray(v.profile.values_bytes)
    index=next(i for i,stored in enumerate(v.profile.explicit_storage_mask) if stored)
    raw[4*index:4*index+4]=struct.pack("<f",float(v.profile.native_values()[index])+1)
    p=replace(v.profile,values_bytes=bytes(raw))
    assert v.context.prediction_inputs()==v.prediction_inputs()
    with pytest.raises(ValueError):
        changed=replace(v,profile=p)
        native_source.verify(changed)

@pytest.mark.parametrize("article", (32221275,32221302))
@pytest.mark.parametrize("purpose", ("training","evaluation","response_loss","retrieval","split_use","DE_statistics","guide_efficacy"))
def test_real_source_native_candidates_never_admit_supervision_or_splits(real_cases,article,purpose):
    for v in real_cases[article]:
        with pytest.raises(ValueError):v.require_model_use(purpose,admission={"verified":True,"sha256":FIXTURE_SHA})

@pytest.mark.parametrize("field",("native_row_index","native_role","gene_axis","profile","provenance_bytes"))
def test_rehashed_row_axis_role_and_source_substitution_refuse(native_source,real_cases,field):
    v=real_cases[32221275][0]
    replacement=dict(native_row_index=3,native_role="measured_sgRNA_efficacy",
        gene_axis=real_cases[32221302][0].gene_axis,profile=real_cases[32221275][1].profile,
        provenance_bytes=b"{}")[field]
    with pytest.raises(ValueError):
        changed=replace(v,**{field:replacement})
        native_source.verify(changed)

def test_only_fixture_rows_available_no_raw_H5AD_fallback_or_cotrain_fabrication(api,native_source):
    with pytest.raises(ValueError,match="fixture|available"):native_source.read_profiles(api.XPertSelector(32221275,(1,)))
    with pytest.raises(ValueError):api.XPertSelector(32221386,(0,))

@pytest.mark.parametrize("artifact",("fixture","fullsource","QC","coordinates","graph","preparation",
 "gene_axis","obs","values","finite","storage","positions"))
def test_corrupted_consumed_native_values_masks_axes_and_proofs_refuse(api,native_source,artifact,monkeypatch):
    refs=native_source.source_references(32221275)
    target=(ROOT/refs[artifact]["path"]).resolve();read=Path.read_bytes;touched=[]
    def changed(path):
        raw=read(path)
        if path.resolve()==target:touched.append(True);return raw+b" "
        return raw
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError):
        source=api.XPertNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256=FIXTURE_SHA)
        source.read_profiles(api.XPertSelector(32221275,(0,)))
    assert touched

@pytest.mark.parametrize("bad",(True,0.0,"0"))
def test_native_gene_axis_row_type_and_mask_mutations_reject(api,real_cases,bad):
    axis=real_cases[32221302][0].gene_axis
    rows=list(axis.native_rows);first=json.loads(rows[0]);first["native_row_index"]=bad;rows[0]=canonical(first)
    with pytest.raises(ValueError):replace(axis,native_rows=tuple(rows))

def npy_bytes(array):
    buffer=__import__("io").BytesIO();np.save(buffer,array,allow_pickle=False);return buffer.getvalue()

@pytest.mark.parametrize("array,shape,dtype",((np.zeros((1,3),dtype=np.float64),(1,3),"<f4"),
 (np.zeros((1,3),dtype=np.float32),(3,),"<f4"),(np.zeros((1,3),dtype=np.int64),(1,3),"<f4")))
def test_safe_numeric_NPY_refuses_dtype_cast_reshape_or_integer_substitution(api,array,shape,dtype):
    with pytest.raises(ValueError):api.load_numeric_npy(npy_bytes(array),dtype=dtype,shape=shape)

def test_safe_numeric_NPY_refuses_object_payload_and_trailing_bytes_without_pickle(api):
    import io
    buf=io.BytesIO()
    np.lib.format.write_array_header_1_0(buf,dict(descr="|O",fortran_order=False,shape=(1,)))
    raw=buf.getvalue()+b"opaque-object-data-must-never-be-unpickled"
    with pytest.raises(ValueError):api.load_numeric_npy(raw,dtype="<f4",shape=(1,))
    raw=npy_bytes(np.zeros((1,3),dtype=np.float32))
    with pytest.raises(ValueError):api.load_numeric_npy(raw+b"trailing",dtype="<f4",shape=(1,3))

def test_runtime_never_opens_H5_ZIP_pickle_or_imports_inspection_environment(api,native_source,monkeypatch):
    read=Path.read_bytes
    def safe(path):
        assert path.suffix.lower() not in (".pkl",".pickle",".h5ad",".h5",".zip",".pt")
        return read(path)
    monkeypatch.setattr(Path,"read_bytes",safe)
    native_source.read_profiles(api.XPertSelector(32221302,(0,590)))

def test_desired_expression_is_separate_role_not_an_observed_native_label(api,real_cases):
    v=real_cases[32221302][0]
    query=api.XPertDesiredExpression(v.gene_axis,v.profile)
    assert query.role=="desired" and query.as_mapping()["evidence_role"]=="caller_desired_expression"
    with pytest.raises(ValueError):api.XPertDesiredExpression(v.gene_axis,v.profile,role="observed")
    with pytest.raises(ValueError):query.require_model_use("training")

def test_modified_or_wrong_handoff_refuses_before_numeric_source_io(api,tmp_path):
    (tmp_path/"manifest.json").write_bytes(b'{"task_admitted_records":1}')
    with pytest.raises(ValueError):api.XPertNativeSource(root=tmp_path,manifest="manifest.json",expected_manifest_sha256=FIXTURE_SHA)
    with pytest.raises(ValueError):api.XPertNativeSource(root=ROOT,manifest=FIXTURE,expected_manifest_sha256="0"*64)

@pytest.mark.parametrize("mutation",("article_numeric_alias","H5_source","value_role","native_role","admission_numeric_alias"))
def test_rehashed_provenance_type_source_and_scientific_role_changes_reject_at_construction(real_cases,mutation):
    v=real_cases[32221275][0];p=json.loads(v.provenance_bytes);kw={}
    if mutation=="article_numeric_alias":p["source_article_id"]=float(v.source_article_id)
    elif mutation=="H5_source":p["source_H5AD_SHA256"]="0"*64
    elif mutation=="value_role":p["source_value_role"]="computed_DE_statistics_as_observed"
    elif mutation=="native_role":
        p["native_role"]="measured_guide_efficacy";kw["native_role"]=p["native_role"]
    else:p["task_admitted"]=0
    with pytest.raises(ValueError):replace(v,provenance_bytes=canonical(p),**kw)
