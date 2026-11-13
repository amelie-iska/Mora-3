"""Browser mathematical reference versus the native PyTorch implementation."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import pytest
import torch
from scipy.special import ndtr
from mora3.reasoning.persistence_vectors import (exact_landscape_vectors, sampled_landscape_vectors, gaussian_persistence_images, pairwise_l2)
ROOT=Path(__file__).resolve().parents[2]
NODE=shutil.which('node')


def call(code):
    if not NODE:pytest.skip('Installed Node required; no install attempted')
    return json.loads(subprocess.check_output([NODE,'-e',code],cwd=ROOT,text=True))


def test_visual_normal_cdf_matches_independent_scipy():
    pairs=call('const m=require("./docs/project/persistence-vector-math.js");console.log(JSON.stringify(Array.from({length:161},(_,i)=>{let z=-8+i/10;return [z,m.cdf(z)];})));')
    x=np.array(pairs)
    np.testing.assert_allclose(x[:,1],ndtr(x[:,0]),rtol=0,atol=3e-14)


@pytest.mark.parametrize('shift,hidden,resolution,layers,pixels',[(0,True,5,3,16),(.25,True,17,2,8),(.5,False,65,1,32),(0,False,9,3,8)])
def test_same_interactive_math_as_native_cpu(shift,hidden,resolution,layers,pixels):
    state=dict(shift=shift,hidden=hidden,resolution=resolution,layers=layers,pixels=pixels)
    result=call('const m=require("./docs/project/persistence-vector-math.js");console.log(JSON.stringify(m.analysis('+json.dumps(state)+')));')
    diagrams=tuple(torch.tensor(result[k],dtype=torch.float64).reshape(-1,2)for k in('a','b'))
    actual=pairwise_l2(exact_landscape_vectors(diagrams)['vectors'])[0,1].item()
    assert result['exactL2']==pytest.approx(actual,abs=1e-13)
    padded=torch.zeros((2,3,2),dtype=torch.float64);mask=torch.zeros((2,3),dtype=torch.bool)
    for i,d in enumerate(diagrams):padded[i,:len(d)]=d;mask[i,:len(d)]=True
    sampled=sampled_landscape_vectors(padded,mask,torch.linspace(0,4,resolution,dtype=torch.float64),layers=layers)
    assert result['sampledL2']==pytest.approx(pairwise_l2(sampled['vectors'])[0,1].item(),abs=1e-13)
    edges=torch.linspace(0,4,pixels+1,dtype=torch.float64)
    image=gaussian_persistence_images(padded,mask,edges,edges,bandwidth=(.18,.18))
    np.testing.assert_allclose(result['imageB'],image['pixel_mass'][1],atol=3e-14,rtol=0)
    assert result['imageL2']==pytest.approx(pairwise_l2(image['density_l2_vectors'])[0,1].item(),abs=1e-12)


def test_static_paper_and_site_svg_match_live_renderer():
    if not NODE:pytest.skip('Installed Node required')
    raw=subprocess.check_output([NODE,'-e','process.stdout.write(require("./docs/project/persistence-vector-math.js").svg({}));'],cwd=ROOT)
    assert raw==(ROOT/'paper/figures/persistence-vectors.svg').read_bytes()
    assert raw==(ROOT/'docs/project/figures/persistence-vectors.svg').read_bytes()
    assert 'persistence-vectors.svg' in (ROOT/'paper/mora3_preprint.md').read_text()
    assert 'persistence-vectors.html' in (ROOT/'docs/paper.html').read_text()
