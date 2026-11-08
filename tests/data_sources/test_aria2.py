from pathlib import Path
import hashlib
import http.server
import re
import shutil
import subprocess
import sys
import threading
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_aria2 import aria2_command, final_sha256

@pytest.fixture
def server():
    payload=bytes(range(256))*65536
    state={"mode":"ranges","requests":[]}
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def do_HEAD(self):
            self.send_response(200);self.send_header("Content-Length",str(len(payload)));self.send_header("ETag",'"fixed"');self.send_header("Accept-Ranges","bytes");self.end_headers()
        def do_GET(self):
            state["requests"].append(dict(range=self.headers.get("Range"),if_match=self.headers.get("If-Match")))
            if state["mode"]=="changed":
                self.send_response(412);self.send_header("Content-Length","0");self.end_headers();return
            start,end=0,len(payload)-1
            match=re.fullmatch(r"bytes=(\d+)-(\d*)",self.headers.get("Range", ""))
            ranged=bool(match) and state["mode"]=="ranges"
            if ranged:
                start=int(match[1]);end=int(match[2]) if match[2] else end
            self.send_response(206 if ranged else 200);self.send_header("Content-Length",str(end-start+1));self.send_header("ETag",'"fixed"')
            if ranged:self.send_header("Content-Range",f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            try:self.wfile.write(payload[start:end+1])
            except (BrokenPipeError,ConnectionResetError):pass
    class Server(http.server.ThreadingHTTPServer):
        def handle_error(self,*args):pass
    srv=Server(("127.0.0.1",0),Handler);thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    try:yield payload,state,f"http://127.0.0.1:{srv.server_port}/source"
    finally:srv.shutdown();srv.server_close();thread.join(timeout=2)

@pytest.mark.skipif(shutil.which("aria2c") is None,reason="optional user-installed aria2 unavailable")
def test_real_aria2_resumes_exact_prefix_with_sha256(tmp_path,server):
    payload,state,url=server;p=tmp_path/"target.bin";prefix=payload[:3*2**20];p.write_bytes(prefix)
    a=dict(url=url,expected_etag='"fixed"',expected_bytes=len(payload),expected_sha256=hashlib.sha256(payload).hexdigest())
    cmd=aria2_command(a,p,connections=4,min_split="1M",max_tries=1)
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    assert r.returncode==0,r.stdout.decode(errors="replace")[-1000:]
    assert p.read_bytes()==payload and final_sha256(p,a)==a["expected_sha256"]
    assert any(x["range"] for x in state["requests"])
    assert all(x["if_match"]=='"fixed"' for x in state["requests"])

@pytest.mark.skipif(shutil.which("aria2c") is None,reason="optional user-installed aria2 unavailable")
@pytest.mark.parametrize("mode",["changed","no_ranges"])
def test_real_aria2_refuses_entity_or_resume_change_and_keeps_prefix(tmp_path,server,mode):
    payload,state,url=server;state["mode"]=mode;p=tmp_path/"target.bin";prefix=payload[:3*2**20];p.write_bytes(prefix)
    a=dict(url=url,expected_etag='"fixed"',expected_bytes=len(payload),expected_sha256=hashlib.sha256(payload).hexdigest())
    r=subprocess.run(aria2_command(a,p,connections=4,min_split="1M",max_tries=1),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    assert r.returncode!=0
    assert p.read_bytes()==prefix


def test_final_checksum_failure_never_replaces_existing_bytes(tmp_path):
    p=tmp_path/"complete";p.write_bytes(b"wrong")
    with pytest.raises(ValueError):final_sha256(p,dict(expected_bytes=5,expected_sha256="0"*64))
    assert p.read_bytes()==b"wrong"
