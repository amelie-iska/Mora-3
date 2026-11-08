"""Conservative aria2 command policy plus independent immutable-source finalization."""
import pathlib
import shutil
from data_sources_fetch import digest


def aria2_command(asset,destination,connections=16,min_split="32M",max_tries=4):
    if not 1<=connections<=16:raise ValueError("tested per-host connection bound")
    binary=shutil.which("aria2c")
    if not binary:raise RuntimeError("user-installed aria2 unavailable")
    destination=pathlib.Path(destination)
    cmd=[binary,"--no-conf=true","--continue=true","--always-resume=true","--allow-overwrite=false","--auto-file-renaming=false","--remove-control-file=false","--force-save=true","--file-allocation=none",f"--split={connections}",f"--max-connection-per-server={connections}","--max-concurrent-downloads=1",f"--min-split-size={min_split}",f"--max-tries={max_tries}","--retry-wait=10","--timeout=60","--connect-timeout=30","--max-file-not-found=1","--check-integrity=false","--http-accept-gzip=false","--max-download-limit=0","--max-overall-download-limit=0","--auto-save-interval=5","--summary-interval=10","--enable-color=false","--show-console-readout=true","--console-log-level=warn","--download-result=hide","--dir="+str(destination.parent),"--out="+destination.name]
    if asset.get("expected_etag"):cmd.append("--header=If-Match: "+asset["expected_etag"])
    if asset.get("expected_sha256"):cmd.append("--checksum=sha-256="+asset["expected_sha256"])
    cmd.append(asset["url"])
    return cmd


def final_sha256(path,asset):
    path=pathlib.Path(path)
    if path.stat().st_size!=asset["expected_bytes"]:raise ValueError("final source length mismatch")
    sha=digest(path)
    if asset.get("expected_sha256") and sha!=asset["expected_sha256"]:raise ValueError("final source checksum mismatch; all bytes preserved")
    return sha
