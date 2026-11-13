from pathlib import Path
import json,unittest
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[2]

class GRHBridgeBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.engine=sync_playwright().start();cls.browser=cls.engine.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
 @classmethod
 def tearDownClass(cls):cls.browser.close();cls.engine.stop()
 def setUp(self):
  self.context=self.browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True,reduced_motion='reduce');self.page=self.context.new_page();self.errors=[];self.network=[]
  self.page.on('pageerror',lambda e:self.errors.append(str(e)))
  self.page.on('request',lambda r:self.network.append(r.url)if r.url.startswith(('http:','https:'))else None)
  self.page.goto((ROOT/'docs/project/grh-research-bridge.html').as_uri());self.page.wait_for_function('GRHBridge.ready')
 def tearDown(self):
  self.assertEqual(self.errors,[]);self.assertEqual(self.network,[]);self.context.close()
 def test_all_sources_all_prefixes_match_native_visibility(self):
  cases=self.page.evaluate('GRHBridge.data.cases')
  for i,c in enumerate(cases):
   self.page.select_option('#case',str(i))
   for k,p in enumerate(c['prefixes']):
    self.page.locator('#step').fill(str(k));self.page.locator('#step').dispatch_event('input')
    self.assertEqual(int(self.page.locator('#visible-count').inner_text()),len(p['visible']))
    expected={v[7:]for v in p['visible']if v.startswith('tensor:')}
    actual=set(self.page.locator('#facts tr td:first-child').all_text_contents())
    self.assertEqual(actual,expected)
    self.assertEqual(self.page.locator('#prefix-hash').inner_text(),p['condition_sha256'])
    if k<len(c['operations']):self.assertIn('Withheld',self.page.locator('#target-status').inner_text())
 def test_reveal_is_display_only_and_source_link_is_pinned(self):
  pin=self.page.locator('#prefix-hash').inner_text();self.assertFalse(self.page.locator('#target-values').is_visible())
  self.page.click('#reveal');self.assertTrue(self.page.locator('#target-values').is_visible())
  self.assertEqual(pin,self.page.locator('#prefix-hash').inner_text());self.assertEqual(self.page.evaluate('GRHBridge.state().step'),0)
  self.assertIn('dbd954eede479a5129831e4e7c30537389ae973d',self.page.locator('#source-link').get_attribute('href'))
  self.assertTrue(self.page.locator('#source-link').get_attribute('href').endswith('.json'))
 def test_url_state_and_invalid_state(self):
  self.page.evaluate("location.hash='case=2&step=3'");self.page.wait_for_function('GRHBridge.state().case===2')
  self.page.reload();self.page.wait_for_function('GRHBridge.ready');self.assertEqual(self.page.evaluate('GRHBridge.state().step'),3)
  self.page.evaluate("location.hash='case=999&step=-5'");self.page.wait_for_function('GRHBridge.state().case===0')
  self.assertEqual(self.page.evaluate('GRHBridge.state().step'),0)
 def test_svg_export(self):
  with self.page.expect_download()as event:self.page.click('#export')
  text=Path(event.value.path()).read_text();self.assertIn('<svg',text);self.assertIn('photonic-0',text)
 def test_mobile_keyboard_and_no_body_overflow(self):
  self.page.set_viewport_size({'width':390,'height':844});self.page.select_option('#case','2')
  self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'))
  self.page.locator('#step').focus();self.page.keyboard.press('ArrowRight');self.assertEqual(self.page.evaluate('GRHBridge.state().step'),1)
  for p in self.page.locator('button,select,input').all():self.assertGreaterEqual(p.bounding_box()['height'],44)

if __name__=='__main__':unittest.main(verbosity=2)
