"""New literal-wire controls; never imports or invokes NumPy or pickle.load."""
import struct
import unittest

from mora3.data import sccello_numeric_dictionary as wire


def text(value):
    raw = value.encode('utf8')
    return b'\x8c' + bytes([len(raw)]) + raw


def integer(value):
    return b'J' + struct.pack('<i', value)


def literal(dtype='f8', order='<', payload=None, state_version=3, symbol='scalar', key='SYNTHETIC_ONLY_GENE'):
    if payload is None:
        payload = struct.pack('<d', 1.25)
    state = b'(' + integer(state_version) + text(order) + b'NNN' + integer(-1) * 2 + integer(0) + b'tb'
    scalar = (text('numpy.core.multiarray') + text(symbol) + b'\x93' + text('numpy') + text('dtype') +
              b'\x93' + text(dtype) + b'\x89\x88\x87R' + state + b'C' + bytes([len(payload)]) + payload + b'\x86R')
    return b'\x80\x04}(' + text(key) + scalar + b'u.'


class NumericWireTests(unittest.TestCase):
    def decode(self, **kwargs):
        return wire.decode_numeric_mapping(literal(**kwargs))['pairs'][0]['value']

    def test_float64_preserves_bits_and_value(self):
        value = self.decode()
        self.assertEqual(value['value'], 1.25)
        self.assertEqual(value['payload_hex'], struct.pack('<d', 1.25).hex())
        self.assertEqual((value['numpy_dtype'], value['byte_order']), ('f8', '<'))

    def test_signed_int16_negative_and_big_endian(self):
        value = self.decode(dtype='i2', order='>', payload=struct.pack('>h', -32001))
        self.assertEqual(value['value'], -32001)
        self.assertEqual(value['numpy_dtype'], 'i2')

    def test_unsigned_integer_is_not_signed(self):
        self.assertEqual(self.decode(dtype='u2', payload=b'\xff\xff')['value'], 65535)

    def test_negative_zero_float_bits_remain_distinct(self):
        value = self.decode(payload=struct.pack('<d', -0.0))
        self.assertEqual(value['payload_hex'], '0000000000000080')

    def test_nan_bits_preserved_without_nonfinite_JSON_value(self):
        value = self.decode(payload=bytes.fromhex('010000000000f87f'))
        self.assertIsNone(value['value'])
        self.assertFalse(value['finite'])
        self.assertEqual(value['payload_hex'], '010000000000f87f')

    def test_unknown_or_object_dtype_refuses(self):
        for dtype in ('O8', 'c16', 'V8', 'f16', '<f8'):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                self.decode(dtype=dtype)

    def test_wrong_byte_count_refuses(self):
        with self.assertRaises(ValueError):
            self.decode(dtype='f8', payload=b'\0' * 7)

    def test_native_implicit_order_refuses(self):
        with self.assertRaises(ValueError):
            self.decode(order='=')

    def test_wrong_dtype_state_version_refuses(self):
        with self.assertRaises(ValueError):
            self.decode(state_version=4)

    def test_unknown_serialized_global_refuses(self):
        with self.assertRaises(ValueError):
            self.decode(symbol='arbitrary_callable')

    def test_duplicate_gene_key_refuses_instead_of_overwrite(self):
        raw = b'\x80\x04}(' + text('fixture') + integer(1) + text('fixture') + integer(2) + b'u.'
        with self.assertRaises(ValueError):
            wire.decode_numeric_mapping(raw)

    def test_trailing_bytes_after_STOP_refuse(self):
        with self.assertRaises(ValueError):
            wire.decode_numeric_mapping(literal() + b'garbage')

    def test_complete_framed_recipe(self):
        raw = literal()
        body = raw[2:]
        framed = raw[:2] + b'\x95' + struct.pack('<Q', len(body)) + body
        self.assertEqual(wire.decode_numeric_mapping(framed)['pairs'][0]['value']['value'], 1.25)

    def test_cross_frame_operand_refuses(self):
        raw = literal()
        with self.assertRaises(ValueError):
            wire.decode_numeric_mapping(raw[:2] + b'\x95' + struct.pack('<Q', 5) + raw[2:])

    def test_memoized_scalar_references_preserve_original_order(self):
        raw = literal()
        # Memoize the already reduced scalar, then reuse as a second source value.
        raw = raw[:-2] + b'\x94' + text('SECOND_SYNTHETIC_ONLY_GENE') + b'h\0u.'
        result = wire.decode_numeric_mapping(raw)
        self.assertEqual([x['key'] for x in result['pairs']], ['SYNTHETIC_ONLY_GENE', 'SECOND_SYNTHETIC_ONLY_GENE'])
        self.assertEqual([x['value']['value'] for x in result['pairs']], [1.25, 1.25])


class NewRefusalDomain(unittest.TestCase):
    def root(self):
        return b'\x80\x04}\x94(' + text('SYNTHETIC_ONLY')

    def dtype_symbol(self):
        return text('numpy') + text('dtype') + b'\x93'

    def test_memoized_root_as_dtype_argument_refuses(self):
        raw = self.root() + self.dtype_symbol() + b'(h\x00\x89\x88tRu.'
        with self.assertRaises(wire.Refusal):
            wire.decode_numeric_mapping(raw)

    def test_memoized_root_as_dtype_state_member_refuses(self):
        raw = (self.root() + self.dtype_symbol() + text('i2') + b'\x89\x88\x87R' +
               b'(K\x03' + text('<') + b'h\x00NNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tbu.')
        with self.assertRaises(wire.Refusal):
            wire.decode_numeric_mapping(raw)

    def test_memoized_root_as_scalar_dtype_refuses(self):
        raw = (self.root() + text('numpy.core.multiarray') + text('scalar') + b'\x93' +
               b'(h\x00C\x02\x01\x00tRu.')
        with self.assertRaises(wire.Refusal):
            wire.decode_numeric_mapping(raw)

    def test_surrogate_unicode_refuses_cleanly(self):
        raw = b'\x80\x04}(X\x03\x00\x00\x00\xed\xa0\x80K\x01u.'
        with self.assertRaises(wire.Refusal):
            wire.decode_numeric_mapping(raw)


if __name__ == '__main__':
    unittest.main()
