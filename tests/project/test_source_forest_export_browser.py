"""Offline inspection of actual exported reference data, not trained quality."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]


def run():
    target=ROOT/'docs/project/source-forest-exports.html'
    assert target.is_file(),'Source export inspector has not been implemented'
    expected=json.loads((ROOT/'docs/project/source-forest-exports-data.json').read_text())
    assert expected['trained']is False and len(expected['examples'])==3
    results=[]
    with sync_playwright()as p:
        browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome',headless=True,args=['--no-sandbox','--disable-gpu'])
        for width in(1360,390):
            page=browser.new_page(viewport={'width':width,'height':900},accept_downloads=True)
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(target.as_uri());page.wait_for_function("document.documentElement.dataset.ready==='true'")
            for source in expected['examples']:
                page.select_option('#reference',source['trace_id'])
                index=len(source['nodes'])-1;page.select_option('#prefix',str(index))
                page.wait_for_function('(n)=>document.querySelectorAll("#coordinates tbody tr").length===n',arg=8)
                assert page.locator('#selected-title').inner_text()==source['nodes'][index]['id']
                values=[float(x)for x in page.locator('#coordinates tbody td:nth-child(2)').all_text_contents()]
                assert values==source['nodes'][index]['latent']
                assert len(page.locator('#facts li').all_text_contents())==len(source['nodes'][index]['visible_facts'])
                assert page.locator('#export-sha').inner_text()==source['export_sha256']
                assert page.locator('#projection circle.mark').count()==len(source['nodes'])
                page.select_option('#prefix','0')
                assert page.locator('#facts li').all_text_contents()==source['nodes'][0]['visible_facts']
            page.select_option('#x-axis','6');page.select_option('#y-axis','7')
            page.go_back();assert page.input_value('#y-axis')=='1'
            page.go_forward();assert page.input_value('#y-axis')=='7'
            before=page.url;page.reload();page.wait_for_function("document.documentElement.dataset.ready==='true'")
            assert page.input_value('#x-axis')=='6'and page.input_value('#y-axis')=='7'and page.url==before
            page.locator('#prefix').focus();page.keyboard.press('ArrowDown');page.keyboard.press('Enter')
            assert page.input_value('#prefix')=='1'
            with page.expect_download()as event:page.click('#export')
            downloaded=json.loads(Path(event.value.path()).read_text())
            assert downloaded['training_admission']is False and downloaded['trained']is False
            assert downloaded['node']['id']==expected['examples'][-1]['nodes'][1]['id']
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert not errors,errors
            results.append(dict(viewport=width,reference_checks=3,prefix_visibility=True,all_coordinates_exact=True,
                url_reload=True,keyboard=True,evidence_export=True,overflow=False,errors=errors))
            page.close()
        page=browser.new_page();page.goto(target.as_uri()+'#source=not-a-source&node=-1&x=999&y=no')
        page.wait_for_function("document.documentElement.dataset.ready==='true'")
        assert page.input_value('#reference')==expected['examples'][0]['trace_id']
        assert page.input_value('#prefix')=='0'
        page.close();browser.close()
    print(json.dumps(dict(schema='Mora3.SourceForestExportBrowserQA/v1',results=results,invalid_url_fallback=True)))


if __name__=='__main__':run()
