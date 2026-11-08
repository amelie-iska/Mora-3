"""RNAcentral sequence/annotation projection, verified against saved public API bytes.

This is not a lossless ToricGT graph conversion. Original graphs remain sidecars;
proxy latents, chunks, generated reasoning and rewards are not biological targets.
Observed means supplied by the database, not experimentally validated. No network IO.
"""
import hashlib
import json
import re
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import Provenance, LeakageKey, require_text


def convert(row, api_bytes):
    if row.get('dataset') != 'toricblm_rnacentral_8192':
        raise ValueError('unsupported source')
    identity = row.get('entry_id')
    if not isinstance(identity, str) or not re.fullmatch(r'URS[0-9A-F]{10}', identity):
        raise ValueError('invalid RNAcentral ID')
    sequence = row.get('sequence')
    if not isinstance(sequence, str) or not re.fullmatch('[ACGTURYSWKMBDHVN]+', sequence):
        raise ValueError('missing or invalid IUPAC sequence')
    if row.get('split') not in ('train', 'validation', 'test'):
        raise ValueError('missing original split')
    require_text(row.get('split_cluster'), 'upstream split cluster')
    cluster = json.loads(row['split_cluster'])
    if not isinstance(cluster, dict) or cluster.get('schema') != 'toricblm.nonprotein_similarity_split.v1':
        raise ValueError('unsupported cluster schema')
    require_text(cluster.get('cluster_id'), 'cluster ID')
    api = json.loads(api_bytes)
    normalized = sequence.replace('T', 'U')
    if (api.get('rnacentral_id') != identity or api.get('is_active') is not True
            or api.get('sequence') != normalized or type(api.get('length')) is not int
            or api['length'] != len(normalized)
            or api.get('md5') != hashlib.md5(normalized.replace('U', 'T').encode()).hexdigest()):
        raise ValueError('RNAcentral identity/sequence verification failed')
    sha = hashlib.sha256(api_bytes).hexdigest()
    # Content-addressed API snapshot, deliberately not an invented release number.
    provenance = Provenance('RNAcentral', 'api-sha256:' + sha, identity, sha, 'CC0-1.0')
    def annotation(name):
        value = api.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError('annotation must be text')
        value = value or None
        return Measurement(name, value, None, 'observed' if value is not None else 'missing', value is not None)
    nodes = (Node('rna', (Measurement('sequence', sequence, None, 'observed', True),
                          Measurement('structure', None, 'angstrom', 'missing', False))),
             Node('rna_annotation', (annotation('description'), annotation('rna_type'))))
    keys = (LeakageKey('sequence:rna-T-to-U:sha256:v1', hashlib.sha256(normalized.encode()).hexdigest()),
            LeakageKey('toricblm.nonprotein_similarity_split.v1:rna', cluster['cluster_id']))
    return GraphRecord(row['record_id'], nodes, (Edge(0, 1, 'has_annotation', True),), (provenance,), keys)
