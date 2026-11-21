"""Exclusive/atomic completed-case records for deliberate same-run resumption."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from . import fusion_runtime as original
from .ncct_private_output import open_private_binary
from .xai_runtime import record_digest
from .xai_schema import validate_values


@contextmanager
def run_lock(directory):
    """Kernel advisory lock; process death releases it without stale-lock deletion."""
    import fcntl
    path = directory/"analysis.lock"
    if not path.exists() and not path.is_symlink():
        with open_private_binary(path):
            pass
    original.private_file(path)
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def case_path(directory, index):
    if type(index) is not int or index not in range(23):
        raise ValueError("development case ordinal required")
    return directory/f"case_{index:03d}.json"


def validate_case(record):
    if type(record) is not dict or set(record) != {"values", "reasons", "audit"}:
        raise ValueError("private case schema rejected")
    validate_values(record["values"], record["reasons"])
    a = record["audit"]
    if set(a) != {"feature_shape", "native_shape", "spacing_mm_zyx", "mask_reason", "interventions",
                  "frame_count", "blur_sigma_voxels_zyx", "blur_radius_voxels", "blur_sigma_mm_zyx", "patch_shape_zyx"}:
        raise ValueError("private audit schema rejected")
    if a["mask_reason"] not in (None, "missing_mask", "invalid_mask"):
        raise ValueError("private mask audit rejected")
    for key in ("feature_shape", "native_shape"):
        if type(a[key]) is not list or len(a[key]) != 3 or any(type(n) is not int or n < 1 for n in a[key]):
            raise ValueError("private grid audit rejected")
    import math
    if len(a["spacing_mm_zyx"]) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in a["spacing_mm_zyx"]):
        raise ValueError("private spacing audit rejected")
    if (type(a["frame_count"]) is not int or a["frame_count"] not in (1, 40)
            or a["blur_sigma_voxels_zyx"] != [2., 2., 2.] or a["blur_radius_voxels"] != 6
            or a["patch_shape_zyx"] != [4, 16, 16]
            or a["blur_sigma_mm_zyx"] != [2*s for s in a["spacing_mm_zyx"]]):
        raise ValueError("private perturbation definition audit rejected")
    if not set(a["interventions"]).issubset({"gradcam", "hirescam"}):
        raise ValueError("private intervention audit rejected")
    for item in a["interventions"].values():
        if item and (set(item) != {"selected_voxels", "changed_voxels"} or any(type(v) is not int or v < 0 for v in item.values())):
            raise ValueError("private intervention counts rejected")
        if item and item["changed_voxels"] > item["selected_voxels"]*a["frame_count"]:
            raise ValueError("private changed-volume invariant failed")
    return record


def write_case(directory, index, signature, record):
    validate_case(record)
    path = case_path(directory, index)
    pending = path.with_suffix(".pending")
    if path.exists() or path.is_symlink() or pending.exists() or pending.is_symlink():
        raise ValueError("case already exists or incomplete; preserve for review")
    envelope = {"signature": signature, "index": index, "record": record, "sha256": record_digest(record)}
    with open_private_binary(pending) as stream:
        stream.write((json.dumps(envelope, sort_keys=True, allow_nan=False)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    # link is exclusive, unlike replace/rename-overwrite. A crash between link
    # and unlink is detected as an incomplete artifact, never silently repaired.
    os.link(pending, path)
    pending.unlink()


def read_case(directory, index, signature):
    path = case_path(directory, index)
    if path.with_suffix(".pending").exists() or path.with_suffix(".pending").is_symlink():
        raise ValueError("incomplete case artifact requires private review")
    if not path.exists() and not path.is_symlink():
        return None
    data = original.read_private_json(path)
    if (set(data) != {"signature", "index", "record", "sha256"} or data["index"] != index
            or data["signature"] != signature or data["sha256"] != record_digest(data["record"])):
        raise ValueError("resume case identity or checksum mismatch")
    return validate_case(data["record"])
