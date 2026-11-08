import pathlib,sys,gzip
sys.path.insert(0,str(pathlib.Path.cwd()/'scripts'))
import data_sources_rfam as m

def test_legacy_seed_citation_bytes_are_lossless_and_never_utf8_replaced(tmp_path):
    raw=b'# STOCKHOLM 1.0\n#=GF AC RF00001\n#=GF ID tRNA\n#=GF AU M\xfcller R;\na/1-2 AC\n#=GC SS_cons ()\n//\n'
    p=tmp_path/'source.seed.gz';p.write_bytes(gzip.compress(raw))
    assert hasattr(m,'open_seed_text'), 'Rfam seed contains nonUTF8bytes; explicit reversible reader required'
    with m.open_seed_text(p) as f:decoded=f.read()
    assert decoded.encode('latin-1')==raw and '\ufffd' not in decoded
    with m.open_seed_text(p) as f:family=list(m.seed_families(f))[0]
    assert family['source_GF_fields']['AU']==['Müller R;']
