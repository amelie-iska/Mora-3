from dataclasses import replace
import copy
from hashlib import sha256
from zipfile import ZipFile

import pytest
import torch

from mora3.data.bioelectric_evidence import (BioelectricWorkbookNumericCell,NativeBioelectricBaseline,
    NativeBioelectricEndpoint,NativeBioelectricEvent,NativeBioelectricObservation)
from mora3.data.source_scalar_native_rows import ScalarTarget
from mora3.model.bioelectric_outcomes import (BioelectricAssayAxis,SharedBioelectricOutcomeModel,
    bioelectric_outcome_parameter_roles)
from test_full_owner import actual_core_assets,cases,_construct


def source_fixture(tmp_path,*,formula=False):
    prefix=b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="3"><c r="B3">'
    sheet=prefix+(b'<f>1+1</f>' if formula else b'')+b'<v>2.00</v></c></row></sheetData></worksheet>'
    p=tmp_path/('formula.xlsx' if formula else 'numeric.xlsx');member='xl/worksheets/sheet1.xml'
    with ZipFile(p,'x')as z:z.writestr(member,sheet)
    b=p.read_bytes();return BioelectricWorkbookNumericCell(p.name,len(b),sha256(b).hexdigest(),member,sha256(sheet).hexdigest(),'B3',None,'2.00')


def fixture_row(cell):
    target=ScalarTarget('reported_point',2.0,None,None,None,None,None,'2.00','fixture-native-xml',b'{}',None,None,b'{}')
    return NativeBioelectricObservation(NativeBioelectricBaseline('fixture organism','cell culture',None,None,None),
        (NativeBioelectricEvent('other_reported_perturbation','fixture assay treatment','unknown',(),None,None,None,None),),
        NativeBioelectricEndpoint('fixture peak','fixture fluorescence',None,'native_scalar',None),
        'primary_observation','reported_comparator','fixture-work','fixture-assay','fixture-v1',(cell,),numeric_target=target)


def test_exact_native_cell_formula_cache_refused(tmp_path):
    assert source_fixture(tmp_path).read(tmp_path)==2.0
    with pytest.raises(ValueError,match='formula'):source_fixture(tmp_path,formula=True).read(tmp_path)


def test_real_shared_mup_source_outcome_and_pre_nn_proof(tmp_path,actual_core_assets):
    owner=_construct(tmp_path,actual_core_assets);shared=owner.core.parent.parent.annotation
    row=fixture_row(source_fixture(tmp_path));axis=BioelectricAssayAxis.from_endpoint(row.endpoint)
    model=SharedBioelectricOutcomeModel(shared,(axis,),source_contract_manifest_sha256='a'*64)
    got=model.reference_objective((row,),root=tmp_path)
    assert got['observed']==1 and got['loss'].dtype==torch.float64
    expected=-torch.distributions.Normal(got['mean'],got['scale']).log_prob(got['mean'].new_tensor([2.0])).mean()
    torch.testing.assert_close(got['loss'],expected,rtol=1e-14,atol=1e-14)
    g=torch.autograd.grad(got['loss'],(model.heads[axis.key].weight,shared.payload.embedding.weight))
    assert all(torch.count_nonzero(x)>0 and torch.isfinite(x).all() for x in g)
    roles=bioelectric_outcome_parameter_roles(model,identifier_axis=('context','observation'))
    assert set(roles)==set(dict(model.named_parameters(remove_duplicate=False)))
    assert roles['heads.'+axis.key+'.weight'].role=='output'
    state=copy.deepcopy(model.state_dict());copy.deepcopy(model).load_state_dict(state)
    state['_extra_state']['source_contract_manifest_sha256']='b'*64
    with pytest.raises(ValueError,match='checkpoint'):model.load_state_dict(state)
    bad=replace(row,numeric_target=replace(row.numeric_target,reported_value=3.))
    def cannot_run(*args):raise AssertionError('model must not execute with forged native coordinate')
    model._readout=cannot_run
    with pytest.raises(ValueError,match='Float64 bits'):model.reference_objective((bad,),root=tmp_path)
