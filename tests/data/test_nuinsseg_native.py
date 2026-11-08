"""Native NuInsSeg reader tests; fixtures confer no source or training admission."""
import hashlib
import io

import numpy as np
from PIL import Image
import pytest
from scipy.io import savemat

from mora3.data import nuinsseg_native as native


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stack(planes):
    original = np.asarray(planes,dtype='u1',order='F')
    stream = io.BytesIO()
    # SciPy writes the primitive class itself. Exercise the frozen double-header
    # validation by replacing the native class byte (mxUINT8_CLASS -> mxDOUBLE).
    savemat(stream,{'stacked_mask':original},do_compression=False)
    raw = bytearray(stream.getvalue())
    assert raw[144]==9
    raw[144]=6
    proof = dict(native_shape=list(original.shape),native_decoded_storage_dtype='|u1',
        native_MAT_declared_class='double',native_decoded_bytes=original.nbytes,
        decoded_native_Fortran_scalar_stream_sha256=sha(original.tobytes(order='F')),
        complete_native_plane_axis=original.shape[2] if original.ndim==3 else 1,
        finite_nonbinary_values=0,nonfinite_values=0,negative_zero_values=0)
    return bytes(raw),proof,original


def png(pixels):
    image = Image.fromarray(pixels)
    stream = io.BytesIO()
    image.save(stream,format='PNG')
    info = dict(native_mode=image.mode,native_bands=list(image.getbands()),native_format='PNG',
        native_dtype='|u1',native_shape=list(pixels.shape),native_decoded_bytes=pixels.nbytes,
        native_pixel_C_order_sha256=sha(pixels.tobytes()))
    return stream.getvalue(),info


def test_complete_ordered_stack_with_empty_plane_is_immutable():
    source = np.zeros((3,4,4),dtype='u1')
    source[2,1,0] = source[2,1,2] = source[1,3,3] = 1
    raw,proof,original = stack(source)
    actual = native._decode_stack(raw,proof,maximum_declared_stack_bytes=384)
    assert actual.shape==(4,3,4) and actual.dtype==np.uint8
    np.testing.assert_array_equal(actual,np.moveaxis(original,2,0))
    assert not actual[1].any() and actual[0,2,1]==actual[2,2,1]==1
    with pytest.raises(ValueError): actual.setflags(write=True)


def test_native_rank2_singleton_preserves_plane_and_requires_exact_budget():
    raw,proof,original = stack([[0,1,1],[1,0,0]])
    with pytest.raises(ValueError,match='declared array budget'):
        native._decode_stack(raw,proof,maximum_declared_stack_bytes=47)
    actual = native._decode_stack(raw,proof,maximum_declared_stack_bytes=48)
    assert actual.shape==(1,2,3)
    np.testing.assert_array_equal(actual[0],original)
    with pytest.raises(ValueError): actual[0,0,0]=1


def test_stack_digest_and_source_declaration_refuse_wrong_shape():
    raw,proof,_ = stack([[0,1,1],[1,0,0]])
    with pytest.raises(ValueError,match='complete native array digest'):
        native._decode_stack(raw,{**proof,'decoded_native_Fortran_scalar_stream_sha256':'a'*64},
                             maximum_declared_stack_bytes=48)
    with pytest.raises(ValueError,match='native MAT declaration'):
        native._decode_stack(raw,{**proof,'native_shape':[3,2]},
                             maximum_declared_stack_bytes=48)


def test_native_rgb_bytes_without_repair():
    expected = np.arange(36,dtype='u1').reshape(3,4,3)
    raw,proof = png(expected)
    image,rgb,alpha = native._decode_image(raw,proof,None)
    assert alpha is None and image is rgb
    np.testing.assert_array_equal(rgb,expected)
    with pytest.raises(ValueError): rgb.setflags(write=True)


def test_constant_opaque_alpha_is_explicit_and_reversible():
    original = np.empty((2,3,4),dtype='u1')
    original[:,:,:3]=np.arange(18,dtype='u1').reshape(2,3,3)
    original[:,:,3]=255
    raw,proof = png(original)
    alpha_proof = dict(alpha_all255=True,exact_bytewise_reconstruction=True,
        native_RGBA_C_order_SHA256=sha(original.tobytes()),
        original_alpha_C_order_SHA256=sha(original[:,:,3].tobytes()),
        derived_RGB_C_order_SHA256=sha(original[:,:,:3].tobytes()),
        constant_alpha_reconstruction_RGBA_SHA256=sha(original.tobytes()))
    native_rgba,rgb,alpha = native._decode_image(raw,proof,alpha_proof)
    assert alpha==255 and rgb.shape==(2,3,3)
    np.testing.assert_array_equal(native_rgba,original)
    np.testing.assert_array_equal(rgb,original[:,:,:3])
    with pytest.raises(ValueError,match='exact opaque alpha proof'):
        native._decode_image(raw,proof,None)
    original[0,0,3]=254
    wrong_raw,wrong_proof = png(original)
    with pytest.raises(ValueError,match='exact opaque alpha proof'):
        native._decode_image(wrong_raw,wrong_proof,alpha_proof)
