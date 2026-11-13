"""Isolated installed-browser workflow checks. No user profile or network use."""
from pathlib import Path
import json
import os
import shutil
import unittest
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
URL=(ROOT/'docs/project/persistence-vectors.html').as_uri()

class PersistenceBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable=shutil.which(os.environ.get('CHROMIUM_PATH','chromium'))
        if not executable:raise unittest.SkipTest('Installed browser required; no install')
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox','--disable-gpu'])
    @classmethod
    def tearDownClass(cls):cls.browser.close();cls.pw.stop()
    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1440,'height':1100},reduced_motion='reduce',accept_downloads=True)
        self.page=self.context.new_page();self.errors=[];self.requests=[]
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        self.page.on('request',lambda r:self.requests.append(r.url)if r.url.startswith(('http://','https://'))else None)
        self.page.goto(URL);self.page.wait_for_function('window.PersistenceVectorExplorer && PersistenceVectorExplorer.ready')
    def tearDown(self):
        self.assertEqual(self.errors,[]);self.assertEqual(self.requests,[]);self.context.close()
    def test_page_identity_and_real_sampling_counterexample(self):
        self.assertIn('Persistence vectors',self.page.title())
        self.assertEqual(self.page.locator('#figure svg').count(),1)
        self.assertGreater(float(self.page.locator('#exact-value').inner_text()),0)
        self.assertEqual(float(self.page.locator('#sampled-value').inner_text()),0)
        self.assertIn('misses a genuine difference',self.page.locator('#finding').inner_text())
    def test_sampling_control_and_native_interval_toggle(self):
        self.page.locator('#resolution').select_option('65')
        self.assertGreater(float(self.page.locator('#sampled-value').inner_text()),0)
        self.page.locator('#hidden-bar').uncheck()
        for id in('exact-value','sampled-value','image-value'):
            self.assertEqual(float(self.page.locator('#'+id).inner_text()),0)
    def test_shift_rank_and_image_resolution_are_functional(self):
        self.page.locator('#shift').fill('0.5')
        self.page.locator('#layers').select_option('1')
        self.page.locator('#pixels').select_option('32')
        self.assertEqual(self.page.evaluate('PersistenceVectorExplorer.state().pixels'),32)
        self.assertIn('32 × 32 pixels',self.page.locator('#figure').inner_text())
        self.assertIn('layers=1',self.page.url)
        self.assertGreater(float(self.page.locator('#exact-value').inner_text()),.1)
    def test_export_preserves_raw_intervals_and_explicit_approximation(self):
        with self.page.expect_download()as event:self.page.locator('#export-json').click()
        d=json.loads(Path(event.value.path()).read_text())
        self.assertEqual(d['b'][-1],[.21,.29]);self.assertFalse(d['imageIsLossless']);self.assertFalse(d['trained_result'])
        with self.page.expect_download()as event:self.page.locator('#export-svg').click()
        raw=Path(event.value.path()).read_text()
        self.assertIn('retained exact evidence',raw);self.assertNotIn('<script',raw)
    def test_hash_state_and_keyboard_access(self):
        self.page.evaluate('location.hash="#shift=0.25&resolution=17&layers=2&pixels=8&hidden=0"')
        self.page.wait_for_function('PersistenceVectorExplorer.state().resolution===17')
        self.assertFalse(self.page.locator('#hidden-bar').is_checked())
        self.page.locator('#resolution').focus();self.page.keyboard.press('ArrowDown')
        self.assertEqual(self.page.locator('#resolution').input_value(),'65')
    def test_mobile_document_does_not_overflow_and_graph_remains_scrollable(self):
        self.page.set_viewport_size({'width':390,'height':844})
        size=self.page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
        self.assertLessEqual(size['scroll'],size['width']+1)
        self.assertGreaterEqual(self.page.locator('#export-svg').bounding_box()['height'],44)
        self.assertEqual(self.page.locator('#figure').get_attribute('tabindex'),'0')
        self.assertIn('original intervals',self.page.locator('#export-json').inner_text().lower())

if __name__=='__main__':unittest.main()
