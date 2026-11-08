"""Fixed-vocabulary minimum-token encoding, ConvexTok 2605.22821v1 §2–3.

This is the encoding component, not the convex vocabulary learner. Payload
boundaries are caller-supplied: encode each graph measurement separately.
No Unicode normalization, pretokenization, or special-string recognition occurs.
"""
from collections.abc import Iterable


class ByteTokenizer:
    """All 256 bytes, ordered learned byte strings, then explicit control IDs.

    Controls cannot be decoded as bytes or inferred from payload spelling.
    Equal-cost paths prefer the longest first token. Runtime is O(n * L) trie
    transitions in the worst case, for payload length n and longest token L;
    DP storage is O(n). This CPU reference makes no extreme-context claim.
    """

    def __init__(self, learned: Iterable[bytes] = (), *, controls: Iterable[str] = ()):
        learned = tuple(learned)
        controls = tuple(controls)
        if any(type(token) is not bytes or len(token) < 2 for token in learned):
            raise ValueError('Learned tokens must be byte strings of length >= 2')
        if len(set(learned)) != len(learned):
            raise ValueError('Duplicate learned tokens')
        if any(type(name) is not str or not name for name in controls):
            raise ValueError('Control names must be nonempty strings')
        if len(set(controls)) != len(controls):
            raise ValueError('Duplicate control names')
        self._vocabulary = tuple(bytes([i]) for i in range(256)) + learned
        self._controls = {name: len(self._vocabulary) + i for i, name in enumerate(controls)}
        self._trie = {}
        for index, token in enumerate(self._vocabulary):
            node = self._trie
            for byte in token:
                node = node.setdefault(byte, {})
            node[None] = index

    @property
    def vocabulary(self) -> tuple[bytes, ...]:
        return self._vocabulary

    @property
    def size(self) -> int:
        return len(self._vocabulary) + len(self._controls)

    def control_id(self, name: str) -> int:
        return self._controls[name]

    def encode(self, payload: bytes) -> tuple[int, ...]:
        """Solve d[i] = min_{t matching at i} (1 + d[i + len(t)])."""
        if type(payload) is not bytes:
            raise TypeError('Payload must be bytes; encode text explicitly as UTF-8')
        n = len(payload)
        costs = [n + 1] * n + [0]
        choices = [None] * n
        for start in range(n - 1, -1, -1):
            node = self._trie
            end = start
            while end < n and payload[end] in node:
                node = node[payload[end]]
                end += 1
                if None in node:
                    cost = 1 + costs[end]
                    # Increasing ends make equality choose the longer first token.
                    if cost <= costs[start]:
                        costs[start] = cost
                        choices[start] = (node[None], end)
        ids = []
        start = 0
        while start < n:
            index, start = choices[start]
            ids.append(index)
        return tuple(ids)

    def decode(self, ids: Iterable[int]) -> bytes:
        chunks = []
        for index in ids:
            if type(index) is not int or not 0 <= index < len(self._vocabulary):
                raise ValueError('Expected a payload token ID, not a control or invalid ID')
            chunks.append(self._vocabulary[index])
        return b''.join(chunks)
