"""Native BBBC038 source format boundaries on synthetic PNG fixtures."""
import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from mora3.data import bbbc038_native as native


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def png(values):
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format='PNG')
    return stream.getvalue()


def test_exact_native_uint8_mask_bytes_are_preserved():
    values = np.array([[0,255,0],[255,255,0]],dtype='u1')
    result = native._png_values(png(values), mode='L', shape=(2,3),
                                expected_pixels=sha(values.tobytes()))
    assert result == values.tobytes()
    assert set(result) == {0,255}


@pytest.mark.parametrize('shape,mode,error', [
    ((3,2),'L','dimensions'),
    ((2,3,4),'RGBA','8-bit source PNG header'),
])
def test_png_axes_reject_wrong_shape_and_mode(shape,mode,error):
    values = np.zeros((2,3),dtype='u1')
    with pytest.raises(ValueError,match=error):
        native._png_values(png(values),mode=mode,shape=shape,expected_pixels=sha(values.tobytes()))


def test_png_does_not_accept_unverified_decoded_content():
    values = np.zeros((2,3),dtype='u1')
    with pytest.raises(ValueError,match='pixel digest'):
        native._png_values(png(values),mode='L',shape=(2,3),expected_pixels='1'*64)


def test_reversible_exact_opaque_RGBA_and_immutable_RGB():
    rgba = np.empty((2,3,4),dtype='u1')
    rgba[:,:,:3] = np.arange(18,dtype='u1').reshape(2,3,3)
    rgba[:,:,3] = 255
    image = dict(original_decoded_shape=list(rgba.shape),original_decoded_C_order_sha256=sha(rgba.tobytes()))
    alpha = dict(reversible_constant_alpha=255,reconstructed_RGBA_C_order_SHA256=sha(rgba.tobytes()),
        original_alpha_C_order_SHA256=sha(rgba[:,:,3].tobytes()),
        exact_reversible_RGB_C_order_SHA256=sha(rgba[:,:,:3].tobytes()))
    original,rgb = native._original_rgba_and_rgb(png(rgba),image,alpha)
    np.testing.assert_array_equal(original,rgba)
    np.testing.assert_array_equal(rgb,rgba[:,:,:3])
    for result in (original,rgb):
        with pytest.raises(ValueError): result.setflags(write=True)
    rgba[0,0,3] = 254
    image['original_decoded_C_order_sha256'] = sha(rgba.tobytes())
    with pytest.raises(ValueError,match='opaque 255'):
        native._original_rgba_and_rgb(png(rgba),image,alpha)
