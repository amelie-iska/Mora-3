"""Actual installed-browser workflows over retained untrained native computations."""
from pathlib import Path
import os,unittest,json
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


class NativeForestBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable=os.environ.get('CHROMIUM_PATH','/opt/google/chrome/chrome')
        if not Path(executable).is_file():raise unittest.SkipTest('Existing Chromium unavailable; no download')
        cls.p=sync_playwright().start();cls.browser=cls.p.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox','--disable-gpu'])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.p.stop()

    def setUp(self):
        self.context=self.browser.new_context(viewport={'width':1280,'height':1000},accept_downloads=True,reduced_motion='reduce')
        self.page=self.context.new_page();self.errors=[];self.external=[]
        self.page.on('pageerror',lambda error:self.errors.append(str(error)))
        self.page.on('request',lambda request:self.external.append(request.url)if request.url.startswith(('http:','https:'))else None)
        self.page.goto((ROOT/'docs/project/native-forest-heads.html').as_uri())
        self.page.wait_for_function('window.NativeForestExplorer && NativeForestExplorer.ready')

    def tearDown(self):
        self.assertEqual(self.errors,[]);self.assertEqual(self.external,[]);self.context.close()

    def test_real_fixture_has_explicit_untrained_status_and_all_four_heads(self):
        d=self.page.evaluate('NativeForestExplorer.fixture')
        self.assertTrue(d['synthetic']);self.assertFalse(d['trained']);self.assertEqual(d['optimizer_updates'],0)
        self.assertEqual(d['control_count'],40);self.assertEqual(len(d['rows']),10)
        self.assertEqual(self.page.locator('#velocity-values tr').count(),5)
        self.assertEqual(self.page.locator('#quality-values tr').count(),4)
        self.assertIn('0',self.page.locator('#hidden-difference').inner_text())

    def test_all_velocity_controls_match_retained_actual_values(self):
        d=self.page.evaluate('NativeForestExplorer.fixture')
        for row in d['rows']:
            self.page.select_option('#condition',str(int(row['condition'])))
            self.page.select_option('#time',str(row['time']).removesuffix('.0'))
            state=self.page.evaluate('NativeForestExplorer.getState()')
            self.assertEqual(state['condition'],row['condition']);self.assertEqual(state['time'],row['time'])
            for i,v in enumerate(row['velocity']):
                values=self.page.locator('#velocity-values tr').nth(i).locator('td').all_text_contents()
                self.assertAlmostEqual(float(values[1]),v[0],places=6);self.assertAlmostEqual(float(values[2]),v[1],places=6)
            self.assertEqual(row['velocity'][0],[0.,0.])

    def test_nearest_token_controls_use_full3d_distances(self):
        d=self.page.evaluate('NativeForestExplorer.fixture')
        for row in d['nearest_token_examples']:
            self.page.select_option('#node',str(row['node']));self.page.select_option('#slot',str(row['slot']))
            self.assertIn(f"{row['candidates'][0]['distance']:.4f}",self.page.locator('#nearest').inner_text())
            self.assertEqual(row['dimension'],3)

    def test_hash_restoration_and_invalid_state(self):
        self.page.evaluate("location.hash='condition=2&time=0.75&node=4&slot=3'")
        self.page.wait_for_function('NativeForestExplorer.getState().slot===3')
        self.assertEqual(self.page.evaluate('NativeForestExplorer.getState()'),dict(condition=2,time=.75,node=4,slot=3))
        self.page.reload();self.page.wait_for_function('NativeForestExplorer.ready')
        self.assertEqual(self.page.locator('#condition').input_value(),'2')
        self.page.evaluate("location.hash='condition=99&time=NaN&node=-1&slot=999'")
        self.page.wait_for_timeout(20)
        self.assertEqual(self.page.locator('#condition').input_value(),'2')

    def test_svg_and_json_exports_are_current(self):
        for control in('#export-field','#export-token','#export-data'):
            with self.page.expect_download()as event:self.page.click(control)
            download=event.value;raw=Path(download.path()).read_text()
            if control=='#export-data':self.assertEqual(json.loads(raw)['selection'],self.page.evaluate('NativeForestExplorer.getState()'))
            else:self.assertIn('<svg',raw);self.assertIn('viewBox=',raw)

    def test_mobile_keyboard_and_page_bounds(self):
        self.page.set_viewport_size({'width':390,'height':844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
        for element in self.page.locator('select,button').all():self.assertGreaterEqual(element.bounding_box()['height'],44)
        self.page.locator('#node').focus();self.page.keyboard.press('ArrowDown');self.page.keyboard.press('Enter')
        self.assertEqual(self.page.locator('#node').input_value(),'2')
        self.assertTrue(self.page.evaluate('document.querySelector("#vectors svg").getBoundingClientRect().right<=innerWidth'))


if __name__=='__main__':unittest.main(verbosity=2)
