"""Test-only identities; NOT original evidence or campaign receipts."""
import pytest
from mora3.research.cross_source_biogrid_rosetta import audit_gene_group_closure


def rec(source, ident, *groups):
    return {"source":source,"task_id":ident*64,"groups":tuple(groups)}


def test_gene_and_study_edges_are_separate_original_evidence():
    rows={
        "biogrid":(rec("biogrid","a","chemical_identifier:BIOGRID:-","gene:9606:468"),
                   rec("biogrid","b","chemical_identifier:BIOGRID:-","gene:9606:329")),
        "l1000":(rec("l1000","c","gene:9606:468","ncbi_gene:468","study:study-S"),
                 rec("l1000","d","gene:9606:7249","ncbi_gene:7249","study:study-S")),
        "cellpainting":(rec("cellpainting","e","gene:9606:468","study:study-S"),)}
    result=audit_gene_group_closure(rows)
    assert result["direct_pairwise_gene_tokens"]["biogrid:l1000"]==1
    assert result["direct_pairwise_gene_tokens"]["biogrid:cellpainting"]==1
    assert result["direct_pairwise_gene_tokens"]["l1000:cellpainting"]==1
    assert result["placeholder_rows"]==2
    assert result["global_literal_grouping"]["cross_source_components"]==1
    assert result["placeholder_omitted_sensitivity"]["cross_source_components"]==1
    assert result["literal_groups_are_true_chemical_equivalence"] is False
    assert result["training_admission"] is False


def test_placeholder_can_collapse_unrelated_genes_but_not_prove_ligand_identity():
    rows={
        "biogrid":(rec("biogrid","a","chemical_identifier:BIOGRID:-","gene:9606:468"),
                   rec("biogrid","b","chemical_identifier:BIOGRID:-","gene:9606:329")),
        "l1000":(rec("l1000","c","gene:9606:468"),),
        "cellpainting":(rec("cellpainting","d","gene:9606:999"),)}
    result=audit_gene_group_closure(rows)
    assert result["global_literal_grouping"]["cross_rows"]["biogrid"]==2
    assert result["placeholder_omitted_sensitivity"]["cross_rows"]["biogrid"]==1
    assert result["placeholder_omitted_sensitivity"]["cross_rows"].get("cellpainting",0)==0
    assert result["placeholder_rows"]==2


def test_reject_duplicate_record_and_unqualified_group():
    rows={
        "biogrid":(rec("biogrid","a","gene:9606:468"),),
        "l1000":(rec("l1000","b","gene:9606:468"),),
        "cellpainting":(rec("cellpainting","c","study:S"),)}
    assert audit_gene_group_closure(rows)["training_admission"] is False
    with pytest.raises(ValueError):
        audit_gene_group_closure({**rows,"biogrid":rows["biogrid"]*2})
    with pytest.raises(ValueError):
        audit_gene_group_closure({**rows,"biogrid":(rec("biogrid","a","gene:9606:468","gene:9606:468"),)})
    with pytest.raises(ValueError):
        audit_gene_group_closure({**rows,"biogrid":(rec("biogrid","a","unknown"),)})


def test_complete_pinned_original_biogrid_and_rosetta_train_groups():
    """Real originals; run with active-original root, not fixtures or target values."""
    from pathlib import Path
    from mora3.research.cross_source_biogrid_rosetta import audit_original_biogrid_rosetta
    result=audit_original_biogrid_rosetta(Path.cwd())
    assert result["original_shards"] == 173
    assert result["original_family_rows"] == {"biogrid":2916,"l1000":11862,"cellpainting":29060}
    assert result["original_excluded_rows"] == {"biogrid":210,"l1000":7770,"cellpainting":23934}
    assert result["source_original_sidecar_manifest_commitment"] == "c704c9c49874e9b66cc25d19f2168c6ddf2acab8e408fdd0c96454ac80e8b352"
    groups=result["source_group_closure"]
    assert groups["source_rows"] == {"biogrid":2706,"l1000":4092,"cellpainting":5126}
    assert groups["placeholder_rows"] == 2188
    assert groups["direct_pairwise_gene_tokens"] == {
        "biogrid:l1000":21, "biogrid:cellpainting":13,"l1000:cellpainting":142
    }
    assert groups["direct_pairwise_study_tokens"] == {
        "biogrid:l1000":0,"biogrid:cellpainting":0,"l1000:cellpainting":2
    }
    assert groups["global_literal_grouping"]["largest_cross_component_rows"] == 11711
    assert groups["placeholder_omitted_sensitivity"]["largest_cross_component_rows"] == 11613
    assert result["original_source_targets_decoded"] is False
    assert result["heldout_original_shards_read"] is False
    assert result["training_admission"] is False
    assert result["actual_proxy_trials"] == 0
