"""Source-encoding-specific view selection with immutable pilot conservation."""
def create_selection_schema(connection):
    connection.execute('CREATE TABLE selection(source_row INTEGER PRIMARY KEY,iso BLOB NOT NULL,emit_prior INTEGER NOT NULL,emit_codec INTEGER NOT NULL)')
def flag(value):
    if type(value) not in [bool,int] or value not in [0,1]:raise ValueError('explicit observed binary eligibility flag required')
    return int(value)
def queue_graph_views(connection,iso,prior_row,codec_row,pilot_prior,pilot_codec):
    if not isinstance(iso,bytes) or len(iso)!=32:raise ValueError('exact canonical graph SHA256 required')
    pilot_prior,pilot_codec=flag(pilot_prior),flag(pilot_codec)
    for row in [prior_row,codec_row]:
        if row is not None and (type(row) is not int or not 0<=row<8999964):raise ValueError('original native train row required')
    if prior_row is None:raise ValueError('source reference prior row required')
    def queue(row,prior,codec):
        result=connection.execute('INSERT INTO selection VALUES(?,?,?,?) ON CONFLICT(source_row) DO UPDATE SET emit_prior=MAX(emit_prior,excluded.emit_prior),emit_codec=MAX(emit_codec,excluded.emit_codec) WHERE selection.iso=excluded.iso',(row,iso,prior,codec))
        if result.rowcount!=1:raise ValueError('original source row cannot refer to multiple graphs')
    if not pilot_prior:queue(prior_row,1,0)
    if codec_row is not None and not pilot_codec:queue(codec_row,0,1)
def expected_view_mask(emit_prior,emit_codec):return flag(emit_prior)+6*flag(emit_codec)
def mark_view_seen(seen,source_row,direction):
    bits={'reference_prior':1,'smiles_to_selfies':2,'selfies_to_smiles':4}
    if direction not in bits or type(source_row) is not int or not 0<=source_row<len(seen):raise ValueError('exact source row and task direction required')
    bit=bits[direction]
    if seen[source_row]&bit:raise ValueError('duplicate source view direction')
    seen[source_row]|=bit
