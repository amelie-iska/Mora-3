"""Browser mechanics for synthetic inference replay; no learned-quality claims."""
import json
import os
from pathlib import Path
import unittest
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


class RefinementBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=sync_playwright().start()
        cls.browser=cls.p.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH','/opt/google/chrome/chrome'),
            headless=True,args=['--no-sandbox','--disable-gpu'])
        cls.data=json.loads((ROOT/'tests/project/fixtures/flow-fot-refinement-example.json').read_text())
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.p.stop()
    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1440,'height':1100},reduced_motion='reduce',accept_downloads=True)
        self.page=self.context.new_page();self.errors=[];self.external=[]
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        self.page.on('request',lambda r:self.external.append(r.url)if r.url.startswith(('http:','https:'))else None)
        self.page.goto((ROOT/'docs/project/flow-fot-refinement.html').as_uri())
        self.page.wait_for_function('window.RefinementExplorer && RefinementExplorer.ready')
    def tearDown(self):
        self.assertEqual(self.errors,[]);self.assertEqual(self.external,[]);self.context.close()
    def test_actual_fixture_counts_and_selected_outputs(self):
        s=self.data['snapshots'][2]
        self.assertEqual(self.page.locator('#candidate-count').inner_text(),str(s['candidate_attempts']))
        self.assertEqual(self.page.locator('#output-count').inner_text(),'3')
        self.assertEqual(self.page.evaluate('RefinementExplorer.snapshot().outputs'),s['output_ids'])
        self.assertEqual(self.page.locator('#lineage [data-id]').count(),len(s['candidates']))
    def test_round_changes_replay_actual_controller_snapshots(self):
        for r in(0,1,2):
            self.page.locator('#rounds').select_option(str(r))
            s=self.data['snapshots'][r]
            self.assertEqual(self.page.evaluate('RefinementExplorer.snapshot().outputs'),s['output_ids'])
            self.assertEqual(self.page.locator('#lineage [data-id]').count(),len(s['candidates']))
            self.assertIn('r='+str(r),self.page.url)
    def test_keyboard_candidate_selection_and_full_branch_view(self):
        self.page.locator('#lineage [data-id="0"]').focus();self.page.keyboard.press('Enter')
        self.assertEqual(self.page.evaluate('RefinementExplorer.snapshot().candidate'),0)
        self.assertEqual(self.page.locator('#forest rect').count(),6)
        self.assertIn('Fixed condition',self.page.locator('#forest').inner_text())
        self.assertIn('0.',self.page.locator('#candidate-summary').inner_text())
    def test_export_is_live_svg_and_json_not_screenshot(self):
        with self.page.expect_download()as pending:self.page.locator('#export-svg').click()
        download=pending.value
        text=Path(download.path()).read_text()
        self.assertEqual(text,self.page.evaluate('RefinementExplorer.svg()'))
        self.assertIn('<svg',text);self.assertIn('data-id="14"',text)
        with self.page.expect_download()as pending:self.page.locator('#export-json').click()
        data=json.loads(Path(pending.value.path()).read_text())
        self.assertTrue(data['synthetic']);self.assertEqual(data['training_runs'],0)
        self.assertEqual(data['selected']['output_ids'],self.data['snapshots'][2]['output_ids'])
    def test_hash_restoration(self):
        self.page.goto((ROOT/'docs/project/flow-fot-refinement.html').as_uri()+'#r=1&c=6')
        self.page.reload();self.page.wait_for_function('RefinementExplorer.ready')
        state=self.page.evaluate('RefinementExplorer.snapshot()')
        self.assertEqual((state['round'],state['candidate']),(1,6))
    def test_mobile_page_bounds_and_touch_sized_controls(self):
        self.page.set_viewport_size({'width':390,'height':844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'))
        for selector in('#rounds','#candidate','#export-svg','#export-json'):
            box=self.page.locator(selector).bounding_box();self.assertGreaterEqual(box['height'],44)
        self.page.locator('#rounds').select_option('0')
        self.assertEqual(self.page.locator('#candidate-count').inner_text(),'6')


if __name__=='__main__':unittest.main(verbosity=2)
