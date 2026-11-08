"""Complete original BBBC038 stage1 RGB and mask evidence."""
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .finite_union_native_rows import IDENTITY_FIELDS, full9, read_exact


_BASE = 'outputs/data-sources/'
_NATIVE = (_BASE + 'readiness-20261106-bbbc038-v1-complete-native-source-axes-v2/manifest.json',
           'e949949e9c4d562ae7e1052493dfeeb7ab58b2b10b2a8fb8b81a8f053ae02604')
_INDEPENDENT = (_BASE + 'readiness-20261106-bbbc038-v1-independent-native-source-proof/manifest.json',
                '40b67e15ff11986578bcacc005eb8f620e67d442fcaf7becfab7fc66d9e18574')
_ALPHA = (_BASE + 'readiness-20261106-bbbc038-v1-stage1-train-original-RGBA-alpha/manifest.json',
          '6b54d778345ae4a23fda1f8a9ba636e4b957baa9eafb28030932a1727b8eebe2')
_STAGE = 'stage1_train'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _document(root, pin):
    path, expected = pin
    ref = {'path': path, 'bytes': (root / path).stat().st_size, 'sha256': expected}
    return json.loads(read_exact(ref, root=root)), ref


def _png_values(raw, *, mode, shape, expected_pixels):
    if (type(raw) is not bytes or raw[:8] != b'\x89PNG\r\n\x1a\n'
            or raw[12:16] != b'IHDR' or raw[24:26] != (b'\x08\x06' if mode == 'RGBA' else b'\x08\x00')):
        raise ValueError('exact original 8-bit source PNG header required')
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != 'PNG' or image.mode != mode or image.size != (shape[1], shape[0])
                or getattr(image, 'n_frames', 1) != 1):
            raise ValueError('exact original PNG dimensions and native mode required')
        values = np.asarray(image)
        if values.dtype != np.uint8 or values.shape != shape:
            raise ValueError('exact original PNG pixel dtype and shape differ')
        decoded = values.tobytes(order='C')
    if _sha(decoded) != expected_pixels:
        raise ValueError('complete original PNG pixel digest differs')
    return decoded


def _original_rgba_and_rgb(raw, image, alpha):
    shape = tuple(image['original_decoded_shape'])
    decoded = _png_values(raw, mode='RGBA', shape=shape,
                          expected_pixels=image['original_decoded_C_order_sha256'])
    rgba = np.frombuffer(decoded, dtype='u1').reshape(shape)
    rgb = rgba[:, :, :3]
    alpha_plane = rgba[:, :, 3]
    if (alpha['reversible_constant_alpha'] != 255 or not np.all(alpha_plane == 255)
            or _sha(decoded) != alpha['reconstructed_RGBA_C_order_SHA256']
            or _sha(alpha_plane.tobytes()) != alpha['original_alpha_C_order_SHA256']
            or _sha(rgb.tobytes()) != alpha['exact_reversible_RGB_C_order_SHA256']):
        raise ValueError('exact reversible RGB and opaque 255 source proof required')
    return rgba, rgb


@dataclass(frozen=True)
class BBBC038OriginalTrainFrame:
    original_ImageId: str
    original_image_member_ordinal: int
    original_image_member_path: str
    original_mask_member_ordinals: tuple[int, ...]
    original_train_archive_sha256: str
    native_image_hwc: np.ndarray
    rgb_hwc: np.ndarray
    ordered_mask_planes_nhw: np.ndarray
    reversible_original_alpha: int = 255
    original_source_stage: str = _STAGE


class BBBC038OriginalTrainReader:
    """Read complete native stage1 source frames and original ordered mask PNGs."""

    def __init__(self, root):
        import pyarrow.parquet as pq

        self.root = Path(root).resolve()
        native, native_ref = _document(self.root, _NATIVE)
        independent, independent_ref = _document(self.root, _INDEPENDENT)
        alpha, _ = _document(self.root, _ALPHA)
        if (independent['producer_manifest'] != native_ref
                or alpha['original_dataset_source_proofs'] != [native_ref, independent_ref]
                or native['all_archive_members_CRC_and_SHA_verified'] is not True
                or alpha['counts']['EXACT_NATIVE_OPAQUE_255_RGB_RECONSTRUCTION'] != 670
                or alpha['counts']['nonopaque_original_frames'] != 0):
            raise ValueError('verified exact original stage1 masks and RGB source evidence required')
        self._original = alpha['original_train_ZIP']
        self._path = self.root / self._original['path']
        self._lease = None

        def table(ref, *, stage_filter=False):
            return pq.read_table(io.BytesIO(read_exact(ref, root=self.root)), use_threads=False,
                filters=[('official_archive_stage', '=', _STAGE)] if stage_filter else None).to_pylist()

        self._images = {r['original_image_member_ordinal']: r for r in
            table(native['derived']['images'], stage_filter=True)}
        self._alpha = {r['original_image_member_ordinal']: r for r in table(alpha['original_train_alpha_native_positions'])}
        self._members = {r['original_archive_member_ordinal']: r for r in
            table(native['derived']['archive_members'], stage_filter=True)}
        grouped = defaultdict(list)
        for record in table(native['derived']['train_mask_PNGs']):
            grouped[record['original_ImageId']].append(record)
        self._masks = {key: tuple(sorted(rows, key=lambda r: r['original_mask_member_ordinal']))
                       for key, rows in grouped.items()}
        if (len(self._images) != len(self._alpha) or len(self._images) != len(self._masks)
                or len(self._images) != 670 or set(self._images) != set(self._alpha)
                or len(self._members) != 32141 or sum(map(len, self._masks.values())) != 29461):
            raise ValueError('complete exact original stage1 frame and mask axes differ')

    @property
    def image_member_ordinals(self):
        return tuple(sorted(self._images))

    def _check_source_identity(self):
        st = os.fstat(self._file.fileno())
        if (tuple(getattr(st, field) for field in IDENTITY_FIELDS) != self._identity
                or full9(self._path) != self._identity):
            raise ValueError('original BBBC038 ZIP identity changed')

    def __enter__(self):
        if self._lease is not None:
            raise ValueError('BBBC038 original reader already open')
        self._identity = full9(self._path)
        if self._identity[6] != self._original['bytes']:
            raise ValueError('exact original BBBC038 archive length required')
        with ExitStack() as lease:
            self._file = lease.enter_context(self._path.open('rb'))
            self._check_source_identity()
            self._zip = lease.enter_context(zipfile.ZipFile(self._file, 'r'))
            self._infos = self._zip.infolist()
            if len(self._infos) != len(self._members):
                raise ValueError('complete original BBBC038 ZIP directory axis differs')
            self._lease = lease.pop_all()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lease.close()
        self._lease = None

    def _read(self, ordinal, maximum_member_bytes, expected_name, expected_sha):
        info = self._infos[ordinal]
        source = self._members[ordinal]
        if (info.filename != source['original_native_path'] or info.filename != expected_name
                or info.file_size != source['original_source_bytes'] or info.is_dir() or info.flag_bits & 1
                or info.compress_size != source['original_source_compressed_bytes']
                or info.header_offset != source['original_member_header_offset']
                or info.CRC != source['original_source_crc32'] or expected_sha != source['original_source_member_sha256']):
            raise ValueError('exact original BBBC038 ZIP member coordinates differ')
        if info.file_size > maximum_member_bytes:
            raise ValueError('BBBC038 original member exceeds explicit encoded byte budget')
        with self._zip.open(info) as handle:
            raw = handle.read(info.file_size + 1)
        if len(raw) != info.file_size or _sha(raw) != expected_sha:
            raise ValueError('complete original BBBC038 ZIP member digest differs')
        return raw

    def _mask_values(self, masks, shape, maximum_member_bytes):
        for record in masks:
            if (record['original_decoded_shape'] != list(shape) or record['original_raster_mode'] != 'L'
                    or record['original_decoded_dtype'] != '|u1' or record['native_mask_values'] != [0, 255]
                    or record['native_aligned_binary_mask_geometry_checked'] is not True
                    or not record['native_mask_foreground_pixels']):
                raise ValueError('exact native BBBC038 mask geometry and nonempty uint8 values required')
            raw = self._read(record['original_mask_member_ordinal'], maximum_member_bytes,
                             record['original_mask_member_path'], record['original_mask_member_sha256'])
            yield _png_values(raw, mode='L', shape=shape, expected_pixels=record['original_decoded_C_order_sha256'])

    def read_frame(self, image_member_ordinal, *, maximum_member_bytes, maximum_native_stack_bytes):
        """Return the full immutable original frame within explicit byte bounds."""
        if (self._lease is None or type(image_member_ordinal) is not int or image_member_ordinal not in self._images
                or type(maximum_member_bytes) is not int or maximum_member_bytes <= 0
                or type(maximum_native_stack_bytes) is not int or maximum_native_stack_bytes <= 0):
            raise ValueError('open original BBBC038 reader, exact image ordinal and positive bounds required')
        self._check_source_identity()
        image = self._images[image_member_ordinal]
        alpha = self._alpha[image_member_ordinal]
        key = image['original_ImageId']
        masks = self._masks[key]
        shape = image['native_height'], image['native_width']
        if (image['original_individual_mask_PNG_count'] != len(masks) or alpha['original_ImageId'] != key
                or alpha['original_encoded_source_SHA256'] != image['original_image_member_sha256']):
            raise ValueError('exact original BBBC038 image, alpha and mask identity differs')
        if len(masks) * shape[0] * shape[1] > maximum_native_stack_bytes:
            raise ValueError('complete original BBBC038 mask stack exceeds explicit native stack budget')
        raw = self._read(image_member_ordinal, maximum_member_bytes, image['original_image_member_path'],
                         image['original_image_member_sha256'])
        rgba, rgb = _original_rgba_and_rgb(raw, image, alpha)
        values = b''.join(self._mask_values(masks, shape, maximum_member_bytes))
        layers = np.frombuffer(values, dtype='u1').reshape((len(masks), *shape))
        self._check_source_identity()
        return BBBC038OriginalTrainFrame(key, image_member_ordinal, image['original_image_member_path'],
            tuple(record['original_mask_member_ordinal'] for record in masks), self._original['sha256'], rgba, rgb, layers)
