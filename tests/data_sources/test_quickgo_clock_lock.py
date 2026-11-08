import importlib.util,pathlib,multiprocessing
ROOT=pathlib.Path.cwd()
def test_actual_admission_clock_is_read_under_shared_lock():
    p=ROOT/'scripts/data_sources_quickgo_resume_v2.py';s=importlib.util.spec_from_file_location('resume_actualclock',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);g=m.SharedLimits(multiprocessing.get_context('fork'),3,{'annotation':(1000,0)})
    def clock():
        assert g.lock._semlock._is_mine(), 'actual admission time read before acquiring shared lock'
        return 25.0
    assert g.wait_for_start(clock=clock)==25.0
