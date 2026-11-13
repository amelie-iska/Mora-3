from pathlib import Path
from playwright.sync_api import sync_playwright
import json,os,shutil
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path=shutil.which(os.environ.get('CHROMIUM_PATH','chromium')),headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
 page.set_content((ROOT/'docs/project/flow-forest.html').read_text());page.wait_for_function('MoraExplorer.ready')
 for tab in ['architecture','forest','topology','decode','fewstep','quality','campaign']:
  page.locator(f'[data-tab="{tab}"]').click();page.screenshot(path=str(ROOT/f'qa/desktop-{tab}.png'),full_page=True)
 page.set_viewport_size({'width':390,'height':844});page.locator('[data-tab="forest"]').click()
 page.screenshot(path=str(ROOT/'qa/mobile-forest.png'),full_page=True)
 page.locator('[data-tab="decode"]').click();page.screenshot(path=str(ROOT/'qa/mobile-decode.png'),full_page=True)
 b.close()
