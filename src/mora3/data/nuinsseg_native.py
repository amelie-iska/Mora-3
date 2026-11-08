"""Authenticated RGB and ordered NuInsSeg layers from the immutable author ZIP."""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat, whosmat

from .finite_union_native_rows import IDENTITY_FIELDS, full9, read_exact


_SOURCE = 'outputs/data-sources/'
_NATIVE = _SOURCE + 'readiness-20261106-nuinsseg-native-axes-v1/'
_CENSUS = (_NATIVE + 'manifest.json',
           '6d5fc2a842f6ebc7d732049a702204ef37786a7c1c4ba37a933bdafd12ce2e6a')
_ORACLE = (_NATIVE + 'independent-complete-original-native-axes-verification.json',
           'a794b40af73780426250601aead9ac1a42c299a568deb141510a0ca766475a01')
_ALPHA = (_SOURCE + 'readiness-20261106-nuinsseg-opaque-alpha-reversibility-v1/'
          'exact-original-alpha-and-reversible-RGB-view.json',
          'a172988951c426073d4f014b4e6dae135236b80f89452916b538c5c3afd39f62')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_json(root, pin):
    path, digest = pin
    reference = {'path': path, 'bytes': (root / path).stat().st_size, 'sha256': digest}
    return json.loads(read_exact(reference, root=root)), reference


def _decode_stack(raw, metadata, *, maximum_declared_stack_bytes):
    """Use the source-qualified v2 MAT rules with exact observed uint8 storage."""
    shape = tuple(metadata['native_shape'])
    if (type(raw) is not bytes or len(shape) not in (2, 3)
            or any(type(n) is not int or n <= 0 for n in shape)
            or metadata['native_MAT_declared_class'] != 'double'
            or metadata['native_decoded_storage_dtype'] != '|u1'):
        raise ValueError('exact primitive native MAT declaration and stored dtype required')
    if type(maximum_declared_stack_bytes) is not int or math.prod(shape) * 8 > maximum_declared_stack_bytes:
        raise ValueError('complete native MAT exceeds explicit declared array budget')
    if whosmat(io.BytesIO(raw), appendmat=False, squeeze_me=False, struct_as_record=True) != [
            ('stacked_mask', shape, 'double')]:
        raise ValueError('single exact native MAT declaration required')
    result = loadmat(io.BytesIO(raw), appendmat=False, variable_names=['stacked_mask'],
        mat_dtype=False, squeeze_me=False, chars_as_strings=False, struct_as_record=True,
        verify_compressed_data_integrity=True, simplify_cells=False)
    if set(result) != {'__header__', '__version__', '__globals__', 'stacked_mask'}:
        raise ValueError('unexpected native MAT variables')
    array = result['stacked_mask']
    if (type(array) is not np.ndarray or array.dtype.str != '|u1' or array.shape != shape
            or array.nbytes != metadata['native_decoded_bytes']):
        raise ValueError('exact original primitive array shape, dtype and bytes required')
    raw_values = array.tobytes(order='F')
    if _sha(raw_values) != metadata['decoded_native_Fortran_scalar_stream_sha256']:
        raise ValueError('complete native array digest differs')
    original = np.frombuffer(raw_values, dtype='u1').reshape(shape, order='F')
    layers = original[None, :, :] if original.ndim == 2 else np.moveaxis(original, 2, 0)
    if layers.shape[0] != metadata['complete_native_plane_axis']:
        raise ValueError('complete native ordered plane axis differs')
    return layers


def _decode_image(raw, metadata, opaque_alpha):
    shape = tuple(metadata['native_shape'])
    mode = metadata['native_mode']
    if (metadata['native_format'] != 'PNG' or mode not in ('RGB', 'RGBA')
            or len(shape) != 3 or shape[2] != len(mode)
            or metadata['native_dtype'] != '|u1' or metadata['native_bands'] != list(mode)):
        raise ValueError('exact observed native PNG RGB/RGBA declaration required')
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != 'PNG' or image.mode != mode or image.size != (shape[1], shape[0])
                or getattr(image, 'n_frames', 1) != 1 or image.getbands() != tuple(mode)):
            raise ValueError('exact original PNG dimensions and native channels required')
        decoded = np.asarray(image)
        if decoded.dtype.str != '|u1' or decoded.shape != shape:
            raise ValueError('exact original PNG dtype and pixel axis differ')
        values = decoded.tobytes(order='C')
    if (len(values) != metadata['native_decoded_bytes']
            or _sha(values) != metadata['native_pixel_C_order_sha256']):
        raise ValueError('complete native PNG pixel digest differs')
    original = np.frombuffer(values, dtype='u1').reshape(shape)
    if mode == 'RGB':
        return original, original, None
    rgb = original[:, :, :3]
    alpha = original[:, :, 3]
    if (opaque_alpha is None or opaque_alpha.get('alpha_all255') is not True
            or opaque_alpha.get('exact_bytewise_reconstruction') is not True or not np.all(alpha == 255)
            or _sha(alpha.tobytes()) != opaque_alpha.get('original_alpha_C_order_SHA256')
            or _sha(rgb.tobytes()) != opaque_alpha.get('derived_RGB_C_order_SHA256')
            or metadata['native_pixel_C_order_sha256'] != opaque_alpha.get('native_RGBA_C_order_SHA256')
            or metadata['native_pixel_C_order_sha256'] != opaque_alpha.get('constant_alpha_reconstruction_RGBA_SHA256')):
        raise ValueError('exact opaque alpha proof required for reversible native RGB view')
    return original, rgb, 255


@dataclass(frozen=True)
class NuInsSegOriginalFrame:
    """Complete authenticated pixels and ordered observed source mask layers."""
    original_stack_member_ordinal: int
    original_image_member_ordinal: int
    original_stack_member_name: str
    original_image_member_name: str
    original_archive_sha256: str
    original_image_mode: str
    original_stack_shape: tuple[int, ...]
    native_image_hwc: np.ndarray
    rgb_hwc: np.ndarray
    ordered_mask_planes_nhw: np.ndarray
    constant_alpha_for_rgb_reconstruction: int | None
    empty_native_plane_ordinals: tuple[int, ...]
    original_source_role: None = None


class NuInsSegOriginalReader:
    """Open exact retained source metadata once and read only requested members.

    The selected stack ordinal is the original ZIP directory coordinate, not a
    synthetic sample or biological identity. Use the reader as a context manager.
    A frame returns source evidence and carries no assigned experiment role.
    """

    def __init__(self, root):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        census, census_ref = _source_json(self.root, _CENSUS)
        oracle, _ = _source_json(self.root, _ORACLE)
        alpha, _ = _source_json(self.root, _ALPHA)
        inv = json.loads(read_exact(census['complete_original_inventory'], root=self.root))
        counts = oracle['counts']
        if (oracle['producer_manifest'] != census_ref
                or not oracle['every_original_plane_scalar_bit_and_native_order_checked']
                or not oracle['all_original_raster_pixels_and_ROI_headers_checked']
                or oracle['original_full9_unchanged'] is not True
                or census['original'] != inv['original']
                or census['original_full9'] != inv['original_full9']
                or inv['all_member_CRC_and_uncompressed_SHA_checked'] is not True
                or counts['matrix/rows'] != 665 or counts['matrix/planes'] != 30704
                or counts['matrix/empty_planes'] != 1
                or alpha['original_archive_SHA256'] != census['original']['sha256']):
            raise ValueError('exact completed native source census and independent oracle required')
        self._original = census['original']
        self._archive_path = self.root / self._original['path']
        self._identity = tuple(census['original_full9'])
        self._archive_members = inv['archive_members']
        self._alpha = alpha
        self._lease = None
        matrices = pq.read_table(io.BytesIO(read_exact(census['matrices'], root=self.root)), use_threads=False)
        self._frames = {}
        needed = set()
        for matrix in matrices.to_pylist():
            ordinal = matrix['original_archive_member_ordinal']
            image = json.loads(matrix['exact_source_companion_contexts_json'])['raw_image']
            if (ordinal in self._frames or matrix['original_source_role'] is not None
                    or image['status'] != 'EXACT_NATIVE_FILENAME_CONTEXT'
                    or matrix['native_shape'][:2] != image['native_shape'][:2]
                    or any(matrix[key] != 0 for key in ('finite_nonbinary_values', 'nonfinite_values', 'negative_zero_values'))):
                raise ValueError('exact original stack/image source identity differs')
            self._frames[ordinal] = (matrix, image)
            needed.update((matrix['native_stack_name'], image['native_name']))
        members = pq.read_table(io.BytesIO(read_exact(inv['all_members'], root=self.root)), use_threads=False,
            columns=['original_archive_member_ordinal', 'native_name', 'source_file_bytes', 'source_compressed_bytes',
                     'source_CRC32', 'header_offset', 'compress_type', 'source_flag_bits', 'directory',
                     'uncompressed_whole_sha256'])
        self._members = {row['native_name']: row for row in members.to_pylist() if row['native_name'] in needed}
        if (len(matrices) != len(self._frames) or len(self._frames) != 665 or len(needed) != 2 * len(self._frames)
                or set(self._members) != needed or len(members) != self._archive_members
                or sum(m['complete_native_plane_axis'] for m, _ in self._frames.values()) != 30704):
            raise ValueError('complete original frame and layer census differs')
        for ordinal, (matrix, image) in self._frames.items():
            source_stack = self._members[matrix['native_stack_name']]
            source_image = self._members[image['native_name']]
            if (source_stack['original_archive_member_ordinal'] != ordinal
                    or source_stack['uncompressed_whole_sha256'] != matrix['original_numeric_member_sha256']
                    or source_image['uncompressed_whole_sha256'] != image['encoded_sha256']):
                raise ValueError('original exact stack/image members differ')

    @property
    def stack_member_ordinals(self):
        return tuple(sorted(self._frames))

    def _require_original_archive(self):
        stat = os.fstat(self._file.fileno())
        if (tuple(getattr(stat, key) for key in IDENTITY_FIELDS) != self._identity
                or full9(self._archive_path) != self._identity):
            raise ValueError('immutable NuInsSeg original ZIP identity differs')

    def __enter__(self):
        if self._lease is not None:
            raise ValueError('NuInsSeg source reader already open')
        with ExitStack() as stack:
            self._file = stack.enter_context(self._archive_path.open('rb'))
            self._require_original_archive()
            self._archive = stack.enter_context(zipfile.ZipFile(self._file, mode='r'))
            self._infos = self._archive.infolist()
            if len(self._infos) != self._archive_members:
                raise ValueError('exact original ZIP directory count differs')
            self._lease = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lease.close()
        self._lease = None

    def _read_member(self, name, maximum_member_bytes):
        expected = self._members[name]
        info = self._infos[expected['original_archive_member_ordinal']]
        if (info.filename != name or info.is_dir() or expected['directory']
                or info.file_size != expected['source_file_bytes']
                or info.compress_size != expected['source_compressed_bytes']
                or info.CRC != expected['source_CRC32'] or info.header_offset != expected['header_offset']
                or info.compress_type != expected['compress_type'] or info.flag_bits != expected['source_flag_bits']
                or info.flag_bits & 1):
            raise ValueError('exact authenticated NuInsSeg original ZIP member differs')
        if info.file_size > maximum_member_bytes:
            raise ValueError('original NuInsSeg member exceeds explicit encoded byte budget')
        with self._archive.open(info, mode='r') as handle:
            raw = handle.read(info.file_size + 1)
        if len(raw) != info.file_size or _sha(raw) != expected['uncompressed_whole_sha256']:
            raise ValueError('complete original NuInsSeg ZIP member digest differs')
        return raw

    def read_frame(self, stack_member_ordinal, *, maximum_member_bytes, maximum_declared_stack_bytes):
        """Read a complete authenticated source frame within explicit byte budgets."""
        if (self._lease is None or type(stack_member_ordinal) is not int
                or stack_member_ordinal not in self._frames
                or type(maximum_member_bytes) is not int or maximum_member_bytes <= 0
                or type(maximum_declared_stack_bytes) is not int or maximum_declared_stack_bytes <= 0):
            raise ValueError('open source reader, exact original stack coordinate and positive budgets required')
        self._require_original_archive()
        matrix, image = self._frames[stack_member_ordinal]
        name = image['native_name']
        image_ordinal = self._members[name]['original_archive_member_ordinal']
        alpha = self._alpha if (image_ordinal == self._alpha['original_archive_member_ordinal']
                                 and name == self._alpha['native_name']) else None
        pixels, rgb, constant = _decode_image(self._read_member(name, maximum_member_bytes), image, alpha)
        stack = _decode_stack(self._read_member(matrix['native_stack_name'], maximum_member_bytes), matrix,
                              maximum_declared_stack_bytes=maximum_declared_stack_bytes)
        if stack.shape[1:] != rgb.shape[:2]:
            raise ValueError('exact original stack and image pixel axes differ')
        empty = tuple(matrix['empty_native_plane_positions'])
        if any(i < 0 or i >= len(stack) or stack[i].any() for i in empty):
            raise ValueError('exact observed empty original source plane differs')
        self._require_original_archive()
        return NuInsSegOriginalFrame(stack_member_ordinal, image_ordinal, matrix['native_stack_name'], name,
            self._original['sha256'], image['native_mode'], tuple(matrix['native_shape']),
            pixels, rgb, stack, constant, empty)
