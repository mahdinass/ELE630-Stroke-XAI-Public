"""Private numeric QA and display-only previews; no patient loader or model fitting."""
from __future__ import annotations
import io
import hashlib
import numpy as np
from .geometry import _affine
from .xai_geometry import project_native

METHODS = ("gradcam", "hirescam")
PUBLIC_KEYS = {"schema", "modality", "status", "gradcam_nonzero_3d", "gradcam_nonzero_slice",
    "gradcam_shared_scale_suppressed", "gradcam_positive_clipping_empty", "gradcam_signed_nonzero_3d", "gradcam_signed_nonzero_slice", "hirescam_nondegenerate",
    "image_cam_array_alignment_verified", "physical_display_orientation_verified",
    "slice_anatomically_informative", "suitable_for_report", "uncertainty"}
UNCERTAINTIES = {"raw_cam_unavailable", "affine_provenance_unverified", "human_anatomy_review_required",
                 "no_refit_artifact_missing", "coarse_grid_interpolation", "clinical_orientation_unverified"}

def public_summary(modality):
    if modality not in ("ncct", "ctp"):
        raise ValueError("fixed modality required")
    return dict(schema=1, modality=modality, status="INSUFFICIENT_ARTIFACTS_NO_REFIT",
        gradcam_nonzero_3d=None, gradcam_nonzero_slice=None, gradcam_shared_scale_suppressed=None,
        gradcam_positive_clipping_empty=None, gradcam_signed_nonzero_3d=None, gradcam_signed_nonzero_slice=None, hirescam_nondegenerate=None,
        image_cam_array_alignment_verified=None, physical_display_orientation_verified=False,
        slice_anatomically_informative=None, suitable_for_report=False,
        uncertainty=["raw_cam_unavailable", "affine_provenance_unverified",
                     "human_anatomy_review_required", "no_refit_artifact_missing"])

def validate_public(record):
    if set(record) != PUBLIC_KEYS or record["schema"] != 1 or record["modality"] not in ("ncct", "ctp"):
        raise ValueError("public QA schema rejected")
    if record["modality"] not in ("ncct", "ctp") or record["status"] not in ("INSUFFICIENT_ARTIFACTS_NO_REFIT", "MAP_QA_REQUIRES_PRIVATE_REVIEW"):
        raise ValueError("public QA enums rejected")
    for key in PUBLIC_KEYS - {"schema", "modality", "status", "uncertainty"}:
        if record[key] is not None and type(record[key]) is not bool:
            raise ValueError("public numeric or identifying content rejected")
    if (type(record["uncertainty"]) is not list or not all(type(x) is str and x in UNCERTAINTIES for x in record["uncertainty"])
            or record["suitable_for_report"] is not False or record["physical_display_orientation_verified"] is not False
            or record["slice_anatomically_informative"] is not None):
        raise ValueError("unverified clinical approval rejected")
    return record

def map_stats(values):
    a = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(a)
    valid = a[finite]
    return dict(finite_valid=bool(finite.all()), count=int(a.size), finite_count=int(finite.sum()),
        nonzero=int(np.count_nonzero(valid)), positive_count=int((valid > 0).sum()), negative_count=int((valid < 0).sum()),
        minimum=float(valid.min()) if valid.size else None, maximum=float(valid.max()) if valid.size else None,
        quantiles=np.quantile(valid, [0, .01, .5, .99, .995, 1]).tolist() if valid.size else None,
        signed_sum=float(valid.sum()), absolute_sum=float(np.abs(valid).sum()))

def midpoint(shape):
    if len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError("native 3D geometry required")
    return shape[0] // 2

def anatomy_slice(support, *, independently_verified=False):
    """Proposed rule only. Never accepts lesion masks as whole-brain provenance."""
    a = np.asarray(support)
    if not independently_verified or a.ndim != 3 or a.dtype != np.bool_ or not a.any():
        raise ValueError("independently verified anatomy support required")
    areas = a.sum((1, 2)); candidates = np.flatnonzero(areas == areas.max())
    return int(min(candidates, key=lambda z: (abs(z - (len(areas)-1)/2), z)))

def audit_maps(image, feature_maps, affine, *, modality, aggregate=False):
    """Trusted UiS in-memory input only. Private result includes numeric distributions.

    Does not establish clinical coordinate provenance. No flips/resampling of inputs.
    Signed and positive projections are measured independently (clipping and linear
    projection do not commute). CTP caller must identify frame20 versus aggregate.
    """
    image = np.asarray(image)
    matrix = np.asarray(_affine(affine))
    z = midpoint(tuple(image.shape))
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1 or set(feature_maps) != set(METHODS):
        raise ValueError("finite native image and matched methods required")
    qa = dict(positive_definition="mean of per-frame clipped maps" if aggregate else "clip signed feature map", schema=1, native_shape=list(image.shape), affine=matrix.tolist(), slice_index=z,
        image=map_stats(image), display_slice=map_stats(image[z]), feature_stride_zyx=[4,16,16],
        display_origin="lower", display_interpolation="nearest", physical_orientation_verified=False,
        pixel_aspect=float(np.linalg.norm(matrix[:3,1])/np.linalg.norm(matrix[:3,0])), methods={})
    cuts = {}
    for method in METHODS:
        data = feature_maps[method]
        signed = np.asarray(data["signed"]); positive = np.asarray(data["positive"])
        if signed.shape != positive.shape or not np.isfinite(signed).all() or not np.isfinite(positive).all():
            raise ValueError("nonfinite or unmatched feature maps")
        if (positive < 0).any() or (not aggregate and not np.array_equal(positive, np.maximum(signed, 0))):
            raise ValueError("frozen positive clipping mismatch")
        qa["methods"][method] = {"feature_shape": list(signed.shape)}
        for kind, feature in (("signed", signed), ("positive", positive)):
            native = project_native(feature, tuple(image.shape))
            qa["methods"][method][kind] = dict(feature=map_stats(feature), native=map_stats(native), slice=map_stats(native[z]))
            if kind == "positive":
                cuts[method] = native[z].copy()
            del native
    shared = float(np.percentile(np.concatenate([cuts[m].ravel() for m in METHODS]),99.5))
    g = qa["methods"]["gradcam"]; h = qa["methods"]["hirescam"]
    positive_max = g["positive"]["slice"]["maximum"]
    qa["shared_vmax"] = shared
    qa["gradcam_to_hirescam_slice_max_ratio"] = positive_max / h["positive"]["slice"]["maximum"] if h["positive"]["slice"]["maximum"] > 0 else None
    qa["gradcam_max_shared_alpha"] = float(min(positive_max/shared,1)*.65) if shared > 0 else 0.
    public = public_summary(modality)
    public.update(status="MAP_QA_REQUIRES_PRIVATE_REVIEW", gradcam_signed_nonzero_3d=g["signed"]["native"]["nonzero"] > 0,
        gradcam_signed_nonzero_slice=g["signed"]["slice"]["nonzero"] > 0, gradcam_nonzero_3d=g["positive"]["native"]["nonzero"] > 0,
        gradcam_nonzero_slice=g["positive"]["slice"]["nonzero"] > 0,
        gradcam_shared_scale_suppressed=positive_max > 0 and qa["gradcam_max_shared_alpha"] < 1/255,
        gradcam_positive_clipping_empty=g["signed"]["feature"]["nonzero"] > 0 and g["positive"]["feature"]["nonzero"] == 0,
        hirescam_nondegenerate=h["positive"]["native"]["maximum"] > 0 and h["positive"]["native"]["maximum"] != h["positive"]["native"]["minimum"],
        image_cam_array_alignment_verified=True,
        uncertainty=["affine_provenance_unverified", "human_anatomy_review_required", "coarse_grid_interpolation", "clinical_orientation_unverified"])
    return qa, validate_public(public), cuts

def render_preview(image, cuts, affine, *, modality, separate=False):
    """Display-only: fixed [0,1] model window, unchanged native slice and orientation.

    Never labels values HU; zero maps remain visible negative results.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    matrix = np.asarray(_affine(affine)); image = np.asarray(image)
    z = midpoint(tuple(image.shape))
    if not np.isfinite(image).all() or set(cuts) != set(METHODS):
        raise ValueError("invalid display inputs")
    if any(a.shape != image[z].shape or not np.isfinite(a).all() or (a < 0).any() for a in cuts.values()):
        raise ValueError("display alignment rejected")
    shared = float(np.percentile(np.concatenate([cuts[m].ravel() for m in METHODS]),99.5))
    aspect = float(np.linalg.norm(matrix[:3,1])/np.linalg.norm(matrix[:3,0]))
    fig, axes = plt.subplots(1,3,figsize=(11,4.2),constrained_layout=True)
    try:
        for ax in axes:
            ax.imshow(image[z],cmap="gray",vmin=0,vmax=1,origin="lower",interpolation="nearest",aspect=aspect)
            ax.axis("off")
        axes[0].set_title("Model intensity [0,1]; not HU")
        for ax, method in zip(axes[1:], METHODS):
            a = cuts[method]; vmax = float(np.percentile(a,99.5)) if separate else shared
            if vmax <= 0 and a.max() > 0 and separate:
                vmax = float(a.max())  # Explicit fallback only for sparse display.
            if vmax > 0:
                artist=ax.imshow(a,cmap="inferno",vmin=0,vmax=vmax,alpha=np.clip(a/vmax,0,1)*.65,origin="lower",interpolation="nearest",aspect=aspect)
                fig.colorbar(artist,ax=ax,fraction=.03,label="Positive attribution; unnormalized")
            ax.set_title(method + (" (zero positive slice)" if not a.any() else ""))
        fig.suptitle(("CTP frame20: per-frame contribution" if modality == "ctp" else "NCCT") + " | reused development23 illustration | native midpoint; clinical orientation unverified\n" +
            ("Separate display scales ONLY: magnitude comparison invalid" if separate else "Original shared absolute attribution scale"))
        out=io.BytesIO();fig.savefig(out,format="png",dpi=150,metadata={});return out.getvalue()
    finally:
        plt.close(fig)


def verify_original_manifest(prior, png, *, modality, protocol):
    expected = dict(schema=1, status="PRIVATE_REVIEW_ONLY", modality=modality,
        source_revision="79dd321f0deff7e6028da9081699c64c56e56d63", protocol_sha256=protocol,
        validation_count=23, selection="median absolute frozen fused logit; original ordinal tie-break",
        plane="native Z midpoint (unoriented array view)", ctp_original_frame_index=20 if modality=="ctp" else None,
        cam_target="original predicted-class fused support logit", cam_layer="stage4.conv2",
        exported_to_external_service=False, filename=f"{modality}_development_example.png")
    if set(prior) != set(expected) | {"selected_development_ordinal", "png_sha256"}:
        raise ValueError("original artifact schema requires compatibility review")
    if (any(prior[k] != v for k,v in expected.items())
            or type(prior["selected_development_ordinal"]) is not int
            or prior["selected_development_ordinal"] not in range(23)
            or hashlib.sha256(png).hexdigest() != prior["png_sha256"]):
        raise ValueError("original artifact provenance rejected")
    return prior
