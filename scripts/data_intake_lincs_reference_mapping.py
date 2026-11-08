"""Exact source identifier bindings, not assay pairing or alias resolution."""
def exact_perturbation_map(rows):
    found={}
    for row in rows:
        key=row['pert_id']
        if not isinstance(key,str) or not key:
            continue
        value=found.setdefault(key,dict(graphs=set(),source_rows=[],unresolved_rows=[]))
        value['source_rows'].append(row['source_row'])
        if row.get('graph_sha') is not None:value['graphs'].add(row['graph_sha'])
        else:value['unresolved_rows'].append(row['source_row'])
    result={}
    for key,value in found.items():
        graphs=sorted(value['graphs'])
        result[key]=dict(eligible_reference_graph_sha=graphs[0] if len(graphs)==1 and not value['unresolved_rows'] else None,
                         graph_candidates=graphs,source_rows=sorted(value['source_rows']),unresolved_source_rows=sorted(value['unresolved_rows']),
                         reason='unresolved_source_graphs' if graphs and value['unresolved_rows'] else 'unique_literal_source_reference_graph' if len(graphs)==1 else 'multiple_literal_reference_graphs' if graphs else 'no_eligible_source_graph')
    return result
