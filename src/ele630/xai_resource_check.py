"""Bounded native-size synthetic CUDA/actual-gzip-proxy calibration; no cohort I/O."""
from __future__ import annotations

import gzip
from types import SimpleNamespace

import numpy as np
import torch

from .xai_resources import ResourceProfile, MAX_CHECK_SECONDS, planning_seconds, ctp_workload, RESOURCE_VERSION


def _nifti_fixture(path, shape_zyx, profile, on_created):
    """Write40 fictional frames incrementally, owner-private and exclusive."""
    import nibabel as nib
    from nibabel._compression import HAVE_INDEXED_GZIP
    from .ncct_private_output import _validate_base, open_private_binary
    from .ctp_input import prepare_ctp_frame_input
    _validate_base(path.parent)
    header = nib.Nifti1Header()
    shape_xyz = tuple(reversed(shape_zyx))
    header.set_data_shape((*shape_xyz, 40))
    header.set_data_dtype(np.float32)
    header.set_sform(np.eye(4), code=1)
    header.set_qform(np.eye(4), code=1)
    header["vox_offset"] = 352
    header.set_slope_inter(1., 0.)
    with open_private_binary(path) as raw:
        on_created()  # Cleanup ownership begins ONLY after exclusive creation.
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream:
            stream.write(header.binaryblock)
            stream.write(b"\0"*4)
            for i in range(40):
                profile._check_deadline()
                # z/y/x C bytes equal x/y/z Fortran bytes in NIfTI storage.
                frame = np.random.default_rng(630+i).random(shape_zyx, dtype=np.float32)*255
                stream.write(frame.tobytes(order="C"))
    image = nib.load(str(path))  # Same installed ArrayProxy/gzip backend as real input.
    class FictionalProxy:
        metadata = SimpleNamespace(frame_count=40, spatial_shape=shape_xyz, affine=np.eye(4))
        @property
        def _proxy(self):
            return image.dataobj
        def load_frame(self, i):
            profile._check_deadline()
            return SimpleNamespace(input=prepare_ctp_frame_input(np.asanyarray(image.dataobj[:, :, :, i]), np.eye(4), i))
    return FictionalProxy(), bool(HAVE_INDEXED_GZIP)


def synthetic_resource_check(shape_zyx=(72, 512, 512), *, device="cuda", fixture_path=None):
    """CPU/in-memory entry supports tiny local tests; it NEVER satisfies UiS gate.

    Production caller supplies an exclusive owner-private fixture path. Deadline
    checked between operations; Slurm is the hard bound on an in-flight kernel.
    """
    from .xai_ctp_cache import cached_series, prepared_bytes, MAX_CACHE_BYTES
    from .ctp_model import CompactCTPMean
    from .fusion_features import freeze
    from .fusion_synthetic import FictionalCTP
    from .xai_spatial import frame_cam
    from .xai_temporal import replacement_frames
    from .xai_perturbation import replace_blur
    from .xai_randomization import randomized_copy
    from .xai_synthetic import fictional_state
    from .xai_fused_score import torch_state, image_gradient
    from .ncct_training import _parameter_digest
    shape = tuple(shape_zyx)
    if len(shape) != 3 or any(type(n) is not int or not 4 <= n <= 1024 for n in shape) or np.prod(shape) > 64*1024*1024:
        raise ValueError("bounded synthetic resource shape required")
    if prepared_bytes(tuple(reversed(shape))) > MAX_CACHE_BYTES:
        raise ValueError("synthetic shape exceeds per-patient CPU cache bound")
    profile = ResourceProfile(device, max_seconds=MAX_CHECK_SECONDS)
    torch.manual_seed(630)
    model = freeze(CompactCTPMean().to(device))
    before = _parameter_digest(model)
    state = torch_state(fictional_state(), device)
    created = False
    series = None
    try:
        if fixture_path is None:
            series, indexed = FictionalCTP(630, shape=shape), None
        else:
            # Mark only a previously absent, workflow-owned fixture as ours.
            if fixture_path.exists() or fixture_path.is_symlink():
                raise ValueError("new fictional resource fixture required")
            def mark_created():
                nonlocal created
                created = True
            with profile.phase("resource_fixture"):
                series, indexed = _nifti_fixture(fixture_path, shape, profile, mark_created)
        with cached_series(series, resources=profile, resource_phase="resource_cache_creation") as cached:
            with profile.attach(model):
                with torch.no_grad(), profile.phase("resource_warmup"):
                    model.encoder.forward_features(model._frame_tensor(cached, 0))
                resident_frame = model._frame_tensor(cached, 0)
                with torch.no_grad(), profile.phase("resource_forward"):
                    model.encoder.forward_features(resident_frame)
                del resident_frame
                with torch.no_grad(), profile.phase("resource_baseline"):
                    model._mean_embedding(cached)  # All40 frames, actual streaming semantics.
                with profile.phase("resource_cam"):
                    frame_cam(model.encoder, model._frame_tensor(cached, 0), image_gradient(state)/40)
                with profile.phase("resource_upper_cam"):
                    copied, copied_state = randomized_copy(model, state, upper_block=True)
                    frame_cam(copied.encoder, copied._frame_tensor(cached, 0), image_gradient(copied_state)/40)
                    del copied, copied_state
                with torch.no_grad(), profile.phase("resource_spatial"):
                    mask = np.zeros(shape, dtype=bool)
                    mask.ravel()[:int(np.ceil(.1*mask.size))] = True
                    perturbed, _ = replace_blur(model._frame_tensor(cached, 0), mask)
                    model.encoder.forward_features(perturbed)
                    del perturbed, mask
                with torch.no_grad(), profile.phase("resource_temporal"):
                    for _, replacement in replacement_frames(lambda i: model._frame_tensor(cached, i), 34):
                        model.encoder.forward_features(replacement)
            cache_record = cached.record()
        if before != _parameter_digest(model) or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError("resource check altered frozen synthetic model")
        record = profile.record()
        estimate = planning_seconds(record)
        return {"schema": RESOURCE_VERSION, "resource_version": RESOURCE_VERSION, "shape_zyx": list(shape),
                "device": torch.device(device).type, "patient_data_access": False,
                "io_backend": "nifti_gzip_proxy" if fixture_path is not None else "in_memory_fictional",
                "indexed_gzip_available": indexed, "cache": cache_record, "profile": record, "workload": ctp_workload(),
                "projected_23_case_seconds_with_2x_margin": estimate,
                "within_24h_with_20pct_headroom": estimate <= .8*24*3600}
    finally:
        series = None  # Release ArrayProxy before deleting this fictional file.
        if created and fixture_path.exists():
            fixture_path.unlink()
