"""Preserve native R rowname semantics and genuinely headerless source tables."""
import csv,io

def r_pdata_records(text):
    reader=csv.reader(io.StringIO(text),delimiter=' ',skipinitialspace=True);header=next(reader)
    if not header or header[0]!='Cell':raise ValueError('explicit native Cell field required')
    result=[]
    for row in reader:
        if not row:continue
        if len(row)!=len(header)+1:raise ValueError('native R implicit rowname width mismatch')
        data=dict(zip(['native_row_name']+header,row))
        if data['native_row_name']!=data['Cell']:raise ValueError('native rowname and Cell identity disagree')
        result.append(data)
    return result

def headerless_tab(text,min_columns):
    rows=[]
    for row in csv.reader(io.StringIO(text),delimiter='\t'):
        if not row:continue
        if len(row)<min_columns:raise ValueError('incomplete native headerless record')
        rows.append(row)
    return rows
