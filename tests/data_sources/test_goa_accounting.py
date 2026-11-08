import importlib.util,pathlib
ROOT=pathlib.Path.cwd()
def module():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    p=ROOT/'scripts/data_sources_goa_acquire_v2.py'
    assert p.is_file(), 'isolated incremental GOA byte accounting is not implemented'
    s=importlib.util.spec_from_file_location('data_sources_goa_acquire_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_budget_tracks_new_immutable_files_in_own_batches_without_rescanning_other_jobs(tmp_path):
    m=module();client=m.EfficientSkillClient.__new__(m.EfficientSkillClient);client.accounted_bytes=0;client.accounted_paths=set();client.batch_dir=None
    assert client.raw_bytes()==0
    a=tmp_path/'own-a';a.mkdir();b=tmp_path/'other-job';b.mkdir();(b/'foreign').write_bytes(b'X'*10000)
    client.batch_dir=a;(a/'page.json').write_bytes(b'123')
    assert client.raw_bytes()==3 and client.raw_bytes()==3
    (a/'receipt.json').write_bytes(b'4567');assert client.raw_bytes()==7
    c=tmp_path/'own-b';c.mkdir();client.batch_dir=c;(c/'page.json').write_bytes(b'8');assert client.raw_bytes()==8

def test_cumulative_budget_does_not_forget_previous_batch_bytes(tmp_path):
    m=module();client=m.EfficientSkillClient.__new__(m.EfficientSkillClient);client.accounted_bytes=0;client.accounted_paths=set()
    for i in range(5):
        directory=tmp_path/str(i);directory.mkdir();client.batch_dir=directory;(directory/'source').write_bytes(b'X'*10)
        assert client.raw_bytes()==10*(i+1)
