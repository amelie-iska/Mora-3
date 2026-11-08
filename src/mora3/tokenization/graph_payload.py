"""Lossless scalar graph payload packing; not a tensorizer or neural encoder.

One container per node/edge retains TokenGT incidence. ConvexTok shortest paths
are solved independently inside each measurement. Strings use raw UTF-8; numbers
and booleans use canonical JSON; null is an explicitly typed empty payload.
Record identity/provenance/leakage keys stay in an audit sidecar, not features.
No conditioning, target masks, field pooling or graph-token expansion is implied.
"""
from dataclasses import asdict
import hashlib
import json

from mora3.data.schema import GraphRecord
from .shortest_path import ByteTokenizer


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


_TYPES = {str: 'str', int: 'int', float: 'float', bool: 'bool', type(None): 'null'}


class GraphPayloadCodec:
    def __init__(self, tokenizer: ByteTokenizer):
        self.tokenizer = tokenizer
        try:
            self.roles = {role: tokenizer.control_id(role) for role in ('pad', 'node', 'edge')}
        except KeyError as exc:
            raise ValueError('Graph vocabulary requires pad/node/edge controls') from exc
        self.vocabulary_sha256 = _sha(_json_bytes({
            'payload_hex': [t.hex() for t in tokenizer.vocabulary],
            'roles': self.roles, 'size': tokenizer.size}))

    def encode(self, record: GraphRecord) -> dict:
        if not isinstance(record, GraphRecord):
            raise TypeError('Expected validated GraphRecord')
        data = asdict(record)
        metadata = {k: v for k, v in data.items() if k not in ('nodes', 'edges')}
        entities = []
        for role, items in (('node', record.nodes), ('edge', record.edges)):
            for index, item in enumerate(items):
                fields = []
                for mi, m in enumerate(item.measurements):
                    payload = (b'' if m.value is None else m.value.encode('utf-8')
                               if type(m.value) is str else _json_bytes(m.value))
                    fields.append({'index': mi, 'name': m.name, 'units': m.units,
                        'status': m.status, 'available': m.available,
                        'value_type': _TYPES[type(m.value)], 'byte_count': len(payload),
                        'token_ids': list(self.tokenizer.encode(payload))})
                entities.append({'role': role, 'role_id': self.roles[role], 'index': index,
                    'kind': item.kind,
                    'endpoints': [index, index] if role == 'node' else [item.source, item.target],
                    'directed': None if role == 'node' else item.directed,
                    'measurements': fields})
        # JSON-native result avoids tuple/list ambiguity after persistence.
        return json.loads(_json_bytes({'version': 1,
            'vocabulary_sha256': self.vocabulary_sha256,
            'record_sha256': _sha(record.canonical_bytes()),
            'canonical_record_bytes': record.canonical_byte_count,
            'metadata': metadata, 'entities': entities}))

    def decode(self, packed: dict) -> GraphRecord:
        """Validate a persisted artifact, including vocabulary and record identity.

        Re-encoding requires canonical boundaries, indices and shortest paths;
        alternate encodings and unknown fields fail closed. Hashes detect damage,
        not malicious edits by someone who can also rewrite the checksums.
        """
        try:
            if packed['vocabulary_sha256'] != self.vocabulary_sha256:
                raise ValueError('vocabulary fingerprint mismatch')
            data = dict(packed['metadata'])
            data['nodes'], data['edges'] = [], []
            for entity in packed['entities']:
                role = entity['role']
                if role not in ('node', 'edge'):
                    raise ValueError('invalid entity role')
                fields = []
                for field in entity['measurements']:
                    raw = self.tokenizer.decode(field['token_ids'])
                    value_type = field['value_type']
                    if value_type == 'str':
                        value = raw.decode('utf-8')
                    elif value_type == 'null':
                        if raw:
                            raise ValueError('null must have empty payload')
                        value = None
                    else:
                        value = json.loads(raw)
                    if _TYPES.get(type(value)) != value_type:
                        raise ValueError('scalar type mismatch')
                    fields.append({k: field[k] for k in ('name', 'units', 'status', 'available')}
                                  | {'value': value})
                item = {'kind': entity['kind'], 'measurements': fields}
                if role == 'edge':
                    source, target = entity['endpoints']
                    item.update(source=source, target=target, directed=entity['directed'])
                data[role + 's'].append(item)
            record = GraphRecord.from_bytes(_json_bytes(data))
            if _json_bytes(self.encode(record)) != _json_bytes(packed):
                raise ValueError('noncanonical or corrupted graph payload artifact')
            return record
        except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
            raise ValueError('malformed graph payload artifact') from exc
