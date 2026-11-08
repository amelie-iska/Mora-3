from pathlib import Path
import datetime,sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_downstream_launch import remaining_whole_seconds,last_completion_marker

def clock():return dict(created_utc='2026-11-01T00:00:00+00:00',whole_max_seconds=43200,deadline_epoch=1793534400.0)

def test_uses_remaining_original_clock_instead_of_restarting12hours():
 c=clock();start=datetime.datetime.fromisoformat(c['created_utc']).timestamp();c['deadline_epoch']=start+43200
 assert remaining_whole_seconds(c,start+3600)==39570


def test_deadline_cannot_silently_extend_or_run_after_remaining_time():
 c=clock();start=datetime.datetime.fromisoformat(c['created_utc']).timestamp();c['deadline_epoch']=start+43201
 assert remaining_whole_seconds(c,start+3600)==39571
 c['deadline_epoch']=start+43210
 with pytest.raises(ValueError):remaining_whole_seconds(c,start+3600)
 c['deadline_epoch']=start+43200
 with pytest.raises(ValueError):remaining_whole_seconds(c,c['deadline_epoch']-10)


def test_success_file_presence_is_not_normal_completion(tmp_path):
 p=tmp_path/'phase.log';p.write_text('COMPLETE_STAGE {}\nTraceback: finalguard\n')
 assert not last_completion_marker(p,'COMPLETE_STAGE')
 p.write_text('processing\nCOMPLETE_STAGE {"rows":1}\n')
 assert last_completion_marker(p,'COMPLETE_STAGE')
