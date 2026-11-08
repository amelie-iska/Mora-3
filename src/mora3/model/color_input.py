"""Explicit lossless RGB input preparation for the queued nuclear semantic head.

Alpha is source packaging, not a biological channel. Only verified opaque RGBA
is accepted under this version's explicit policy; no compositing, resizing,
grayscale projection, mask access or training admission occurs here.
"""
import numpy as np
import torch


def color_tensor(pixels, *, alpha_policy, device='cpu', dtype=torch.float32,
                 input_policy='native-max512-v1',source_envelope=None):
    """Copy bounded uint8 HWC pixels into normalized NCHW RGB model inputs.

    This input-only primitive does not authorize the legacy brightfield head to
    consume color records. Source pixels remain immutable; float conversion is
    explicit. The default retains the historical 512-pixel side limit; the
    explicit native-max1000-v1 policy permits complete 1000-pixel inputs without
    resizing, cropping or changing any channel. Nonopaque alpha requires a separately specified future policy.
    """
    from .raster_source_envelope import RasterSourceEnvelope
    full=input_policy=='source-envelope-v2'
    if(input_policy not in('native-max512-v1','native-max1000-v1','source-envelope-v2')
            or(full and type(source_envelope)is not RasterSourceEnvelope)or(not full and source_envelope is not None)):
        raise ValueError('explicit supported complete source native input policy required')
    if full:source_envelope.binding()
    max_height=source_envelope.max_height if full else(512 if input_policy=='native-max512-v1'else 1000)
    max_width=source_envelope.max_width if full else max_height
    if alpha_policy != 'require-opaque':
        raise ValueError('explicit require-opaque alpha policy required')
    if (not isinstance(pixels,np.ndarray) or pixels.dtype!=np.uint8
            or pixels.ndim!=3 or pixels.shape[2] not in (3,4)
            or not 1<=pixels.shape[0]<=max_height or not 1<=pixels.shape[1]<=max_width):
        raise ValueError('bounded uint8 HWC RGB/RGBA pixels required')
    if dtype not in (torch.float32,torch.float64):
        raise ValueError('explicit float32 or float64 computation required')
    if pixels.shape[2]==4 and not np.all(pixels[:,:,3]==255):
        raise ValueError('RGBA must be fully opaque under require-opaque policy')
    return torch.from_numpy(pixels[:,:,:3].copy()).to(device=device,dtype=dtype).permute(2,0,1).unsqueeze(0)/255
