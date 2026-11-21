"""Stage7-only CPU frame cache. No patient lookup, disk cache or CUDA graphs.

The caller authorizes membership before supplying a series. NIfTI sources are
read in acquisition order through ONE open stream; the frozen adapter/scaling
and model interfaces are unchanged. Each cache MUST have a scoped lifetime.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np

from .ctp_input import CTPFrameInput, prepare_ctp_frame_input, validate_frame_index

MAX_CACHE_BYTES = 8 * 1024**3
CACHE_VERSION = 1


def prepared_bytes(shape_xyz):
    if len(shape_xyz) != 3 or any(type(n) is not int or n <= 0 for n in shape_xyz):
        raise ValueError("positive native CTP dimensions required")
    return math.prod(shape_xyz) * 40 * 4


def validate_cache_record(record):
    if (set(record) != {"schema", "backend", "decoded_frames", "frame_hits", "prepared_bytes", "limit_bytes"}
            or record["schema"] != CACHE_VERSION or record["backend"] not in ("prepared_series", "sequential_nifti_proxy")
            or any(type(record[k]) is not int for k in ("decoded_frames", "frame_hits", "prepared_bytes", "limit_bytes"))
            or record["decoded_frames"] != 40 or record["frame_hits"] < 0
            or not 0 < record["prepared_bytes"] <= record["limit_bytes"] <= MAX_CACHE_BYTES):
        raise ValueError("bounded CTP cache metadata rejected")
    return record


@contextmanager
def _sequential_source(series):
    proxy = getattr(series, "_proxy", None)
    if proxy is None:
        if hasattr(series, "_proxy"):
            raise ValueError("missing validated NIfTI proxy")
        # Fictional in-memory sources only; real repositories always have a proxy.
        yield series
        return
    from nibabel.arrayproxy import ArrayProxy
    from nibabel.openers import ImageOpener
    if (not isinstance(proxy, ArrayProxy) or proxy.order != "F"
            or tuple(proxy.shape) != (*series.metadata.spatial_shape, 40)
            or np.dtype(proxy.dtype).kind != "f" or np.dtype(proxy.dtype).itemsize != 4):
        raise ValueError("validated float32 Fortran-order NIfTI proxy required")
    # Passing a file handle prevents ArrayProxy from reopening gzip per frame.
    # A3D proxy per contiguous frame also avoids4D fileslice read-ahead
    # heuristics (which can rewind gzip for exceptionally small frames).
    with ImageOpener(proxy.file_like, "rb", keep_open=True) as stream:
        frame_bytes = math.prod(series.metadata.spatial_shape)*proxy.dtype.itemsize
        def load_frame(i):
            spec = (series.metadata.spatial_shape, proxy.dtype, proxy.offset+i*frame_bytes, proxy.slope, proxy.inter)
            frame = np.asanyarray(ArrayProxy(stream, spec, mmap=False, order="F"))
            return SimpleNamespace(input=prepare_ctp_frame_input(frame, series.metadata.affine, i))
        yield SimpleNamespace(metadata=series.metadata, load_frame=load_frame)


class CachedCTPFrames:
    """At most8GiB of prepared float32 CPU arrays; deliberately not serializable."""
    def __init__(self, series, *, max_bytes=MAX_CACHE_BYTES, deadline=None):
        shape = tuple(series.metadata.spatial_shape)
        size = prepared_bytes(shape)
        if (series.metadata.frame_count != 40 or type(max_bytes) is not int
                or not 0 < max_bytes <= MAX_CACHE_BYTES or size > max_bytes):
            raise ValueError("CTP frame cache memory/frame-count bound exceeded")
        self.metadata = copy.deepcopy(series.metadata)
        self.metadata.affine.setflags(write=False)
        self._frames, self._hits, self._decoded = [], 0, 0
        self._size, self._limit, self._closed = size, max_bytes, False
        self._backend = "sequential_nifti_proxy" if getattr(series, "_proxy", None) is not None else "prepared_series"
        try:
            with _sequential_source(series) as source:
                for i in range(40):
                    if deadline is not None:
                        deadline()
                    frame = source.load_frame(i).input
                    if (not isinstance(frame, CTPFrameInput) or frame.frame_index != i
                            or frame.native_shape != shape or frame.tensor.shape != (1, *reversed(shape))
                            or frame.tensor.dtype != np.float32 or not frame.tensor.flags.c_contiguous
                            or not np.array_equal(frame.native_affine, self.metadata.affine)
                            or not np.isfinite(frame.tensor).all() or np.any((frame.tensor < 0) | (frame.tensor > 255))):
                        raise ValueError("cached CTP frame violates frozen input contract")
                    if not frame.tensor.flags.owndata:
                        raise ValueError("independently owned prepared CTP frame required")
                    frame.tensor.setflags(write=False)
                    frame.native_affine.setflags(write=False)
                    self._frames.append(SimpleNamespace(input=frame))
                    self._decoded += 1
        except BaseException:
            self.close()
            raise

    def load_frame(self, index):
        index = validate_frame_index(index)
        if self._closed:
            raise ValueError("CTP frame cache is closed")
        self._hits += 1
        return self._frames[index]

    def record(self):
        return {"schema": CACHE_VERSION, "backend": self._backend, "decoded_frames": self._decoded,
                "frame_hits": self._hits, "prepared_bytes": self._size, "limit_bytes": self._limit}

    def close(self):
        self._frames.clear()
        self._closed = True

    def __reduce_ex__(self, protocol):
        raise TypeError("patient CTP frame caches must not be serialized")


@contextmanager
def cached_series(series, *, resources=None, resource_phase="cache_creation"):
    cache = None
    try:
        with resources.phase(resource_phase) if resources is not None else nullcontext():
            cache = CachedCTPFrames(series, deadline=None if resources is None else resources._check_deadline)
        yield cache
    finally:
        if cache is not None:
            try:
                if resources is not None:
                    resources.add_cache(cache.record())
            finally:
                cache.close()
