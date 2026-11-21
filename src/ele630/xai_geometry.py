"""XAI reference-grid validation and explicit frozen-layer centre mapping."""
from __future__ import annotations

import numpy as np

from .geometry import _affine

STRIDE_ZYX = (4, 16, 16)


def validate_labels(labels, affine, shape_xyz, image_affine):
    """No silent registration repair; original audited 0/1/2/3 label schema."""
    a, b = np.asarray(_affine(affine)), np.asarray(_affine(image_affine))
    if (not isinstance(labels, np.ndarray) or labels.shape != tuple(shape_xyz)
            or labels.ndim != 3 or not np.isfinite(labels).all()
            or not np.isin(labels, (0, 1, 2, 3)).all()
            or not np.allclose(a, b, atol=1e-5, rtol=0)):
        raise ValueError("reference mask shape, labels or physical registration invalid")
    return labels.astype(np.uint8, copy=True)


def feature_labels(labels_xyz, affine, feature_shape_zyx):
    """Nearest native mask centres at frozen conv layer centres (offset zero).

    Affine validation is mandatory even though aligned sampling reduces to
    integer strides. This is equivalent to a nearest-neighbour physical transfer
    to image_affine @ diag(16,16,4,1), not a resize to arbitrary endpoints.
    """
    _affine(affine)
    shape = tuple(feature_shape_zyx)
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("positive feature-grid shape required")
    expected = tuple((n+s-1)//s for n, s in zip(reversed(labels_xyz.shape), STRIDE_ZYX))
    if shape != expected:
        raise ValueError("feature grid disagrees with frozen layer stride")
    return labels_xyz.transpose(2, 1, 0)[::4, ::16, ::16].copy()


def project_native(feature_map, shape_zyx):
    """Linear feature-centre interpolation, boundary clamp, no normalization."""
    result = np.asarray(feature_map, dtype=np.float64)
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError("finite 3D feature map required")
    shape = tuple(shape_zyx)
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("native shape required")
    if result.shape != tuple((n+s-1)//s for n, s in zip(shape, STRIDE_ZYX)):
        raise ValueError("feature/native geometry disagrees")
    for axis, (count, stride) in enumerate(zip(shape, STRIDE_ZYX)):
        coords = np.minimum(np.arange(count)/stride, result.shape[axis]-1)
        low = coords.astype(int)
        high = np.minimum(low+1, result.shape[axis]-1)
        weight_shape = [1, 1, 1]
        weight_shape[axis] = count
        weight = (coords-low).reshape(weight_shape)
        result = np.take(result, low, axis=axis)*(1-weight) + np.take(result, high, axis=axis)*weight
    return result
