import importlib.util,pathlib,sys
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def test_revalidation_accepts_already_labelled_exited_process_without_duplicate_state():
    p=ROOT/'scripts/data_sources_quickgo_checkpoint.py';s=importlib.util.spec_from_file_location('checkpoint_reentry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);original=dict(pid=99999999,start_ticks=123,command='already-exited-owned-source',state='exited_before_checkpoint',process_group=99999999)
    one=m.inspect_owned(original);two=m.inspect_owned(one)
    assert one==two and two['state']=='exited_before_checkpoint' and original['state']=='exited_before_checkpoint'
