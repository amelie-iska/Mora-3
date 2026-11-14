"""Real offline manuscript UI, empty result semantics and vector-figure behavior."""
from pathlib import Path
import json
import re
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


def run():
    source=(ROOT/'paper/mora3_complete_methods.md').read_text()
    assert len(re.findall(r'^### (?:Proposition|Theorem) \d+\.',source,re.M))==12
    report=json.loads((ROOT/'paper/empirical_results_template.json').read_text())
    assert report['populated']is False and report['training_started']is False
    assert len(report['proxy'])==24 and len({r['trial_id']for r in report['proxy']})==24
    for row in report['proxy']:
        assert all(row[k]is None for k in('terminal_status','updates','validation_score','gpu_seconds','artifact_sha256'))
    assert len(report['targets'])==2
    assert all(not c['series']for c in report['charts'])
    results=[]
    with sync_playwright()as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
        for width in(1440,390):
            page=browser.new_page(viewport={'width':width,'height':1000},accept_downloads=True)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            requests=[];page.on('request',lambda r:requests.append(r.url)if r.url.startswith(('http:','https:'))else None)
            page.goto((ROOT/'docs/paper.html').as_uri());page.wait_for_function("document.documentElement.dataset.paperReady==='true'")
            assert page.locator('.formal-claim').count()==12
            for anchor in ('flow-forest-methods','native-forest-heads','evidence','source-forest-exports'):
                assert page.locator('[id="'+anchor+'"]').count()==1
            assert page.locator('.method-diagram').count()==6
            assert page.locator('.empty-chart').count()==4
            assert page.locator('.result-table').count()==5
            assert all(not s.strip()for s in page.locator('td[data-result="null"]').all_text_contents())
            assert page.evaluate('[...document.images].every(i=>i.complete&&i.naturalWidth>0)')
            for figure in page.locator('.method-diagram').all():
                button=figure.locator('.diagram-node').first
                button.focus();page.keyboard.press('Enter')
                assert 'Select a diagram'not in figure.locator('.diagram-detail').inner_text()
            topology=page.locator('#diagram-topology')
            for i in range(7):
                topology.locator('.step-topology').click()
                assert topology.get_attribute('data-step')==str(i+1)
            assert topology.locator('.step-topology').is_disabled()
            assert 'Complete ordered tree'in topology.locator('.diagram-detail').inner_text()
            topology.locator('.reset-topology').click();assert topology.get_attribute('data-step')=='0'
            assert topology.locator('.topology-node:visible').count()==1
            square=page.locator('#diagram-directed')
            square.locator('.toggle-square').click()
            assert 'not identified'in square.locator('.diagram-detail').inner_text()
            assert square.locator('.square-equivalence').text_content()=='distinct path words'
            square.locator('.toggle-square').click()
            assert 'Declared relation enabled'in square.locator('.diagram-detail').inner_text()
            with page.expect_download()as saved:
                page.locator('#diagram-campaign .export-diagram').click()
            raw=Path(saved.value.path()).read_text()
            assert '<svg'in raw and'24 existing proxy slots'in raw
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert not errors and not requests,(errors,requests)
            # Every local source hyperlink except the not-yet-printed PDF resolves.
            links=page.locator('a[href]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))')
            missing=[]
            for link in links:
                if link.startswith(('https:','http:','mailto:')):continue
                if link.startswith('#'):
                    if page.locator('[id="'+link[1:]+'"]').count()==0:missing.append(link)
                elif not link.startswith('mora3_methods.pdf')and not(ROOT/'docs'/link.split('#')[0]).exists():missing.append(link)
            assert not missing,missing
            results.append(dict(width=width,claims=12,diagrams=6,empty_charts=4,empty_tables=5,
                figures_keyboard=True,topology_replay=True,directed_relation_toggle=True,
                vector_export=True,no_console_errors=True,no_network_assets=True,no_body_overflow=True,local_links_valid=True))
            page.close()
        browser.close()
    print(json.dumps(dict(schema='Mora3.CompleteMethodsPaperBrowserQA/v1',results=results)))


if __name__=='__main__':run()
