"""Local isolated Chromium QA; no user browser, network, model or optimizer."""
from pathlib import Path
import json
import os
import shutil
import unittest
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
URL=(ROOT/'docs/project/flow-forest.html').as_uri()

class ExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable=shutil.which(os.environ.get('CHROMIUM_PATH','chromium'))
        if executable is None:
            raise unittest.SkipTest('Explicit installed Chromium required; this test does not install a browser')
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop()
    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1440,'height':1000},reduced_motion='reduce',accept_downloads=True)
        self.page=self.context.new_page();self.errors=[];self.network=[]
        self.page.on('pageerror',lambda err:self.errors.append(str(err)))
        self.page.on('request',lambda r:self.network.append(r.url) if r.url.startswith(('http://','https://')) else None)
        self.page.set_content((ROOT/'docs/project/flow-forest.html').read_text());self.page.wait_for_function('window.MoraExplorer && window.MoraExplorer.ready')
    def tearDown(self):
        self.assertEqual(self.errors,[]);self.assertEqual(self.network,[]);self.context.close()
    def choose(self,tab):self.page.locator(f'[role="tab"][data-tab="{tab}"]').click()
    def test_all_views_render_and_exportable_equation(self):
        for tab in ['architecture','forest','topology','decode','fewstep','quality','campaign']:
            self.choose(tab)
            self.assertEqual(self.page.locator('#diagram svg').count(),1)
            self.assertTrue(self.page.locator('#equation').inner_text())
            self.assertIn('tab='+tab,self.page.url)
            self.assertTrue(self.page.locator('#caveat').inner_text())
    def test_solver_and_probabilities_are_real_controls(self):
        self.choose('forest');self.page.locator('#flow-time').fill('1')
        self.assertEqual(self.page.evaluate('MoraExplorer.state().s'),1)
        self.assertIn('s = 1.00',self.page.locator('#diagram').inner_text())
        self.page.locator('#temperature').fill('0.5')
        self.assertAlmostEqual(self.page.evaluate('MoraExplorer.state().temperature'),.5)
        self.assertEqual(self.page.locator('#secondary svg').count(),1)
    def test_topology_switch_changes_relation_not_just_color(self):
        self.choose('topology');self.page.locator('#square').uncheck()
        self.assertIn('mapping refused',self.page.locator('#diagram').inner_text())
        self.page.locator('#filtration').fill('1.5')
        self.assertIn('H₁=0',self.page.locator('#diagram').inner_text())
    def test_typed_decoder_and_control_registry(self):
        self.choose('decode')
        self.assertIn('SOURCE_OBSERVED',self.page.locator('#diagram').inner_text())
        self.page.locator('#token-search').fill('HYPOTHESIS')
        self.assertEqual(self.page.locator('.token-item:not([hidden])').count(),1)
        self.page.locator('#latent-x').fill('1.7')
        self.assertAlmostEqual(self.page.evaluate('MoraExplorer.state().x'),1.7)
    def test_fewstep_and_quality_dispositions(self):
        self.choose('fewstep');self.page.locator('#steps').select_option('1')
        self.assertIn('0.71828',self.page.locator('#diagram').inner_text())
        self.choose('quality');self.page.locator('#modality').select_option('cell')
        self.assertIn('Communication',self.page.locator('#diagram').inner_text())
        self.assertIn('unevaluated',self.page.locator('#quality-result').inner_text())
        self.page.locator('[data-criterion="0"]').select_option('fail')
        self.assertIn('review-required',self.page.locator('#quality-result').inner_text())
    def test_hash_roundtrip_and_keyboard_tab_navigation(self):
        self.page.evaluate('location.hash = "#tab=topology&epsilon=1.15&square=0"');self.page.wait_for_function('MoraExplorer.state().tab === "topology"')
        self.assertIn('mapping refused',self.page.locator('#diagram').inner_text())
        self.page.locator('[data-tab="topology"]').focus();self.page.keyboard.press('ArrowRight')
        self.assertEqual(self.page.locator('[aria-selected="true"]').get_attribute('data-tab'),'decode')
    def test_svg_download_has_exact_current_view_and_no_remote_assets(self):
        self.choose('fewstep')
        with self.page.expect_download() as event:self.page.locator('#export-svg').click()
        raw=Path(event.value.path()).read_text()
        self.assertIn('A flow map is not a coarse Euler step',raw)
        self.assertNotIn('<script',raw);self.assertNotIn('href="http',raw)
    def test_mobile_page_width_and_targets(self):
        self.page.set_viewport_size({'width':390,'height':844})
        self.choose('forest')
        dimensions=self.page.evaluate('({scroll:document.documentElement.scrollWidth,width:innerWidth})')
        self.assertLessEqual(dimensions['scroll'],dimensions['width']+1)
        self.assertGreaterEqual(self.page.locator('#export-svg').bounding_box()['height'],42)
        self.assertEqual(self.page.locator('#diagram svg').count(),1)
    def test_figures_are_same_pure_svg_renderer(self):
        import xml.etree.ElementTree as ET
        for tab in ['architecture','forest','topology','decode','fewstep','quality','campaign']:
            expected=self.page.evaluate('(tab)=>MoraViews.renderSvg(tab,MoraMath.parseState(""))',tab)
            stored=(ROOT/f'paper/figures/flow-forest-{tab}.svg').read_text()
            self.assertEqual(stored,expected)
            ET.fromstring(stored)

if __name__=='__main__':unittest.main(verbosity=2)
