"""The fixtures test parsing only; real coverage is recorded in the output audit."""
import pytest
from mora3.data import gperturb_sciplex as source


def fixture():
    return dict(count_prefix=b',0,1\n0,0.0,2.0\n1,3.0,0.0\n2,1',
                perturbation=b',Dex,Nutlin,BMS,Vehicle,SAHA\n0,0.001,0,0,0,0\n1,0,0,0,1,0\n',
                cell_info=b',n_genes,n_counts\ncell-a,1,2\ncell-b,1,3\n',
                gene_names=b'0\nGENE_A\nGENE_B\n')


def parse(**changes):
    data=fixture(); data.update(changes)
    return source.parse_prefix(**data, limit=2, expected_genes=2, expected_cells=2)


def test_exact_pairing_complete_rows_and_control_semantics():
    rows, genes, audit = parse()
    assert genes == ('GENE_A','GENE_B')
    assert rows[0] == dict(source_index=0,cell_id='cell-a',intervention='Dex',
                           dose_coordinate=0.001,dose_units='source_scaled_coordinate',
                           counts=[0,2],n_genes=1,n_counts=2.0)
    assert rows[1]['intervention']=='Vehicle'
    assert rows[1]['dose_coordinate']==1  # source control code is NOT physical dose
    assert audit['partial_tail_discarded'] is True
    assert audit['complete_prefix_rows']==2
    assert audit['physical_dose_resolved'] is False


@pytest.mark.parametrize('changes',[
 {'count_prefix':b',0,1\n1,0,2\n0,3,0\n'},
 {'count_prefix':b',0,1\n0,0,2,4\n1,3,0\n'},
 {'count_prefix':b',0,1\n0,nan,2\n1,3,0\n'},
 {'count_prefix':b',0,1\n0,-1,2\n1,3,0\n'},
 {'count_prefix':b',0,1\n0,0.5,2\n1,3,0\n'},
 {'perturbation':b',Dex,Nutlin,BMS,Vehicle,SAHA\n0,1,1,0,0,0\n1,0,0,0,1,0\n'},
 {'perturbation':b',Dex,Nutlin,BMS,Vehicle,SAHA\n1,1,0,0,0,0\n0,0,0,0,1,0\n'},
 {'cell_info':b',n_genes,n_counts\ncell-a,1,2\ncell-a,1,3\n'},
 {'gene_names':b'0\nGENE_A\n'},
 {'count_prefix':b',1,0\n0,0,2\n1,3,0\n'},
])
def test_reject_ambiguous_or_corrupt_alignment(changes):
    with pytest.raises(ValueError): parse(**changes)


def test_bounded_selection_does_not_silently_shortfall():
    with pytest.raises(ValueError):
        source.parse_prefix(**fixture(),limit=3,expected_genes=2,expected_cells=2)


def test_expanded_512_cell_prefix_preserves_last_row_and_rejects_shortfall():
    n=512
    data=dict(count_prefix=(',0,1\n'+''.join(f'{i},{i},0\n' for i in range(n))).encode(),
        perturbation=(',Dex,Nutlin,BMS,Vehicle,SAHA\n'+''.join(f'{i},1,0,0,0,0\n' for i in range(n))).encode(),
        cell_info=(',n_genes,n_counts\n'+''.join(f'cell-{i},1,{i}\n' for i in range(n))).encode(),
        gene_names=b'0\nA\nA\n')
    rows,genes,report=source.parse_prefix(**data,limit=n,expected_cells=n,expected_genes=2)
    assert len(rows)==512 and rows[-1]['counts']==[511,0]
    assert rows[-1]['source_index']==511 and genes==('A','A')
    assert report['training_ready'] is False
    with pytest.raises(ValueError):
        source.parse_prefix(**data,limit=513,expected_cells=513,expected_genes=2)
    data['count_prefix']=data['count_prefix'].rsplit(b'511,',1)[0]
    with pytest.raises(ValueError):
        source.parse_prefix(**data,limit=n,expected_cells=n,expected_genes=2)
