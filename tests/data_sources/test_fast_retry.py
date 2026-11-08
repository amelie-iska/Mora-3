from pathlib import Path
import sys
import urllib.error
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_fetch_fast import is_retryable

def test_access_and_missing_sources_stop_while_rate_limit_can_retry():
    assert not is_retryable(urllib.error.HTTPError("https://example.invalid",401,"Unauthorized",{},None))
    assert not is_retryable(urllib.error.HTTPError("https://example.invalid",404,"Missing",{},None))
    assert is_retryable(urllib.error.HTTPError("https://example.invalid",429,"Rate limited",{},None))
    assert not is_retryable(ValueError("checksum mismatch"))
