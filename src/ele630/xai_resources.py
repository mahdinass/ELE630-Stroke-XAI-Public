"""Bounded timing/counter instrumentation; no images, scores or identifiers."""
from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager

PHASES = ("feature_extraction", "case_total", "baseline_cam", "decomposition", "spatial_intervention",
          "randomized_head", "randomized_upper", "temporal", "resource_baseline", "resource_cam",
          "resource_upper_cam", "resource_spatial", "resource_temporal", "resource_fixture", "resource_warmup", "resource_forward",
          "cache_creation", "resource_cache_creation")
RESOURCE_VERSION = 2
MAX_CHECK_SECONDS = 1800


def ctp_workload():
    """Upper bounds for valid maps/masks and distinct targeted/boundary masks."""
    return {"training_feature_forwards": 92*40, "validation_feature_forwards": 23*40,
            "original_forwards_per_case": 1335, "optimized_forwards_per_case": 815,
            "cam_gradient_evaluations_per_case": 120, "spatial_forwards_per_case": 13*40,
            "temporal_replacement_forwards_per_case": 19*5,
            "original_total_forwards": 115*40+23*1335,
            "optimized_total_forwards": 115*40+23*815,
            "feature_cache_builds": 115, "analysis_cache_builds": 23, "total_cache_builds": 138}


class ResourceProfile:
    def __init__(self, device="cpu", *, max_seconds=None):
        import torch
        self.device = torch.device(device)
        self.started = time.monotonic()
        self.deadline = None if max_seconds is None else self.started+max_seconds
        self.forwards = 0
        from .xai_ctp_cache import MAX_CACHE_BYTES
        self.cache = {"builds": 0, "decoded_frames": 0, "frame_hits": 0,
                      "peak_prepared_bytes": 0, "limit_bytes": MAX_CACHE_BYTES}
        self.stats = {name: {"calls": 0, "seconds": 0., "encoder_forwards": 0} for name in PHASES}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sync(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ValueError("synthetic resource-check time budget exceeded")

    @contextmanager
    def attach(self, model):
        # A function closure deliberately shares this counter with deepcopy
        # model hooks. It counts encoder calls in randomized copies as well.
        encoder = getattr(model, "encoder", model)
        def count(module, args):
            self._check_deadline()
            self.forwards += 1
        handle = encoder.stem[0].register_forward_pre_hook(count)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def phase(self, name):
        if name not in self.stats:
            raise ValueError("fixed resource phase required")
        self._check_deadline()
        self._sync()
        start, before = time.monotonic(), self.forwards
        try:
            yield
        finally:
            self._sync()
            stat = self.stats[name]
            stat["calls"] += 1
            stat["seconds"] += time.monotonic()-start
            stat["encoder_forwards"] += self.forwards-before

    def call(self, name, function, *args, **kwargs):
        with self.phase(name):
            return function(*args, **kwargs)

    def add_cache(self, record):
        from .xai_ctp_cache import validate_cache_record
        validate_cache_record(record)
        self.cache["builds"] += 1
        self.cache["decoded_frames"] += record["decoded_frames"]
        self.cache["frame_hits"] += record["frame_hits"]
        self.cache["peak_prepared_bytes"] = max(self.cache["peak_prepared_bytes"], record["prepared_bytes"])

    def record(self):
        import torch
        self._sync()
        rss = None
        if sys.platform.startswith("linux"):
            import resource
            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        return {"schema": RESOURCE_VERSION, "encoder_forwards": self.forwards,
                "elapsed_seconds": time.monotonic()-self.started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
                "process_peak_host_rss_bytes": rss,
                "phases": {name: dict(stat) for name, stat in self.stats.items()}, "cache": dict(self.cache)}


def validate_profile(record):
    if set(record) != {"schema", "encoder_forwards", "elapsed_seconds", "peak_cuda_allocated_bytes",
                       "peak_cuda_reserved_bytes", "process_peak_host_rss_bytes", "phases", "cache"} or record["schema"] != RESOURCE_VERSION:
        raise ValueError("resource metadata schema rejected")
    if set(record["phases"]) != set(PHASES):
        raise ValueError("resource phase schema rejected")
    from .xai_ctp_cache import MAX_CACHE_BYTES
    cache = record["cache"]
    if (set(cache) != {"builds", "decoded_frames", "frame_hits", "peak_prepared_bytes", "limit_bytes"}
            or any(type(n) is not int or n < 0 for n in cache.values())
            or cache["limit_bytes"] != MAX_CACHE_BYTES or cache["peak_prepared_bytes"] > MAX_CACHE_BYTES
            or cache["decoded_frames"] != 40*cache["builds"]):
        raise ValueError("aggregate cache metadata rejected")
    for key in ("encoder_forwards", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "process_peak_host_rss_bytes"):
        value = record[key]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("resource count rejected")
    if type(record["encoder_forwards"]) is not int:
        raise ValueError("resource counter required")
    for value in [record["elapsed_seconds"]]+[r["seconds"] for r in record["phases"].values()]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("resource timing rejected")
    for r in record["phases"].values():
        if set(r) != {"calls", "seconds", "encoder_forwards"} or any(type(r[k]) is not int or r[k] < 0 for k in ("calls", "encoder_forwards")):
            raise ValueError("resource phase counter rejected")
    return record


def planning_seconds(profile):
    """2x estimate: all138 decode/cache builds PLUS unchanged23345 forwards.

    The115 feature caches cannot span the historical gate into23 case analyses.
    Samples include cache-to-host preparation/normalization and GPU transfer.
    Fixture WRITING is synthetic setup and is not charged to patient runs.
    """
    validate_profile(profile)
    phases = profile["phases"]
    baseline = phases["resource_baseline"]["seconds"]/40
    cam = max(phases["resource_cam"]["seconds"], phases["resource_upper_cam"]["seconds"])
    spatial = phases["resource_spatial"]["seconds"]
    temporal = phases["resource_temporal"]["seconds"]/5
    return 2*(138*phases["resource_cache_creation"]["seconds"]+
              4600*baseline+23*(80*baseline+120*cam+520*spatial+95*temporal))
