"""Public-site integration invariants; no datasets, weights, or optimizer calls."""
from pathlib import Path
from html.parser import HTMLParser
import unittest

ROOT=Path(__file__).resolve().parents[2]
FIGURES=('architecture','forest','reward','topology','decode','fewstep','quality','campaign')
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs):
        data=dict(attrs)
        if tag in ('a','img','iframe'):
            self.links.extend([data[k] for k in ('href','src') if k in data])
class MainSiteIntegration(unittest.TestCase):
    def test_home_and_paper_link_to_new_interactive_companion(self):
        for name in ('index.html','paper.html','training.html'):
            text=(ROOT/'docs'/name).read_text()
            self.assertIn('project/flow-forest.html',text,name)
    def test_live_paper_embeds_accessible_interactive_and_static_figures(self):
        text=(ROOT/'docs/paper.html').read_text()
        self.assertIn('title="Flow-generated complete reasoning forests',text)
        self.assertIn('id="flow-forest-methods"',text)
        for name in FIGURES:self.assertIn('project/figures/flow-forest-'+name+'.svg',text)
    def test_site_and_paper_figures_have_identical_bytes(self):
        for name in FIGURES:
            rel='flow-forest-'+name+'.svg'
            self.assertEqual((ROOT/'paper/figures'/rel).read_bytes(),(ROOT/'docs/project/figures'/rel).read_bytes())
    def test_current_training_page_does_not_authorize_superseded_extra_runs(self):
        text=(ROOT/'docs/training.html').read_text()
        self.assertNotIn('Keep eight standard-parameterization baseline runs separate',text)
        self.assertNotIn('Intermediate and target widths',text)
        self.assertIn('24 proxy',text);self.assertIn('two target',text)
    def test_new_deployed_relative_links_exist(self):
        for name in ('index.html','paper.html','training.html'):
            parser=Links();parser.feed((ROOT/'docs'/name).read_text())
            for link in parser.links:
                if link.startswith('project/'):
                    self.assertTrue((ROOT/'docs'/link.split('#')[0]).is_file(),link)
    def test_all_existing_public_visualizations_remain(self):
        for name in ('cartesian-walkthrough.js','layer-trace-engine.js','mechanism-models.js','viewer-scene.js'):
            self.assertTrue((ROOT/'docs/assets'/name).is_file())
        self.assertTrue((ROOT/'src/mora3/native_annealing.py').is_file())
if __name__=='__main__':unittest.main()
