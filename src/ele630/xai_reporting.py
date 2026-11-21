"""Aggregate-only tables and standard Matplotlib figures; no patient-artifact API."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from .ncct_private_output import open_private_binary
from .xai_runtime import validate_public_record
from .xai_schema import METHODS, COMPARABLE, SANITY, METRIC_NAMES


def _write(path, content):
    with open_private_binary(path) as stream:
        stream.write(content.encode("utf-8") if isinstance(content, str) else content)


def _csv(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def table_rows(record, keys):
    for key in keys:
        r = record["summary"][key]
        yield [record["mode"], key, r["n_total"], r["n_valid"], r["mean"], r["sd"], r["median"], r["ci_low"], r["ci_high"]]


def render_figures(record, *, fictional=False):
    """Bytes only, standard Matplotlib; no filesystem or patient input accepted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    validate_public_record(record)
    figures = {}
    prefix = "FICTIONAL SYNTHETIC EXAMPLE — " if fictional else "Development n=23 — "
    def save(name, figure):
        figure.tight_layout()
        for extension, fmt in (("svg", "svg"), ("png", "png")):
            data = io.BytesIO()
            figure.savefig(data, format=fmt, dpi=180, metadata={"Creator": "ELE630 aggregate XAI"} if fmt == "svg" else None)
            figures[f"{name}.{extension}"] = data.getvalue()
        plt.close(figure)
    def panel(name, metric_keys, labels, title, xlabel):
        fig, ax = plt.subplots(figsize=(8, max(3, len(metric_keys)*.55)))
        ax.axvline(0, color=".7", linewidth=.8)
        any_valid = False
        for y, key in enumerate(metric_keys):
            r = record["summary"][key]
            if r["mean"] is not None:
                any_valid = True
                # Bootstrap bounds need not enclose the observed mean for every
                # possible distribution; draw intervals directly, never abs().
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color="#315f85", linewidth=2)
                ax.plot(r["mean"], y, "o", color="#315f85")
            else:
                text = "Withheld fitted parameter" if key == "decomposition.intercept" else "NA / insufficient valid cases"
                ax.text(.02, y, text, transform=ax.get_yaxis_transform(), fontsize=8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel(xlabel)
        ax.set_title(prefix+title)
        ax.grid(axis="x", alpha=.2)
        save(name, fig)
    roi_keys = [f"{m}.roi.{r}.enrichment" for r in ("core", "penumbra_only", "lesion_union", "outside_lesion_brain") for m in METHODS]
    panel("roi_enrichment", roi_keys, [k.replace(".roi.", ": ").replace(".enrichment", "") for k in roi_keys],
          "ROI enrichment", "Attribution mass / brain-volume fraction")
    panel("perturbation_excess", [f"{m}.fidelity.excess_drop" for m in METHODS], list(METHODS),
          "Attribution intervention minus random median", "Predicted-class support-logit drop difference")
    sanity = [f"{m}.randomization.{control}.rank_correlation" for control in ("head", "upper") for m in METHODS]
    panel("randomization", sanity, [k.replace(".randomization.", ": ").replace(".rank_correlation", "") for k in sanity],
          "Parameter randomization", "Spatial rank correlation with original map")
    panel("logit_decomposition", [f"decomposition.{k}" for k in ("intercept", "image", "clinical")],
          ["Intercept", "Image PC term", "Clinical term"], "Exact additive logit terms", "Mean signed poor-outcome logit contribution")
    if record["mode"] == "ctp":
        rows = [record["summary"][f"temporal.frame_{i:02d}"] for i in range(40)]
        x = np.arange(40)
        y = np.array([np.nan if r["mean"] is None else r["mean"] for r in rows])
        lo = np.array([np.nan if r["ci_low"] is None else r["ci_low"] for r in rows])
        hi = np.array([np.nan if r["ci_high"] is None else r["ci_high"] for r in rows])
        fig, (ax, count_ax) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
        ax.axhline(0, color=".6", linewidth=.8)
        ax.plot(x, y, color="#315f85", marker=".")
        ax.fill_between(x, lo, hi, alpha=.2, color="#315f85", label="Pointwise descriptive 95% bootstrap interval")
        ax.set_ylabel("Support-logit drop\n(mean of covering windows)")
        ax.set_title(prefix+"CTP frame-content sensitivity")
        ax.legend(fontsize=8)
        count_ax.step(x, [r["n_valid"] for r in rows], where="mid", color="#666666")
        count_ax.set_ylabel("Valid n")
        count_ax.set_xlabel("Original CTP frame index (0–39); no clinical phase assignment")
        count_ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 39])
        save("ctp_temporal_sensitivity", fig)
    return figures, matplotlib.__version__


def write_report(record, output_dir, *, figures=False, fictional=False):
    """Create an exclusive directory of allowlisted aggregate artifacts only.

    Local use is permitted for a previously sanitized aggregate. Set fictional
    only for fictional examples; that label is placed on every figure/report.
    Matplotlib is optional and must already exist; never installs packages.
    """
    validate_public_record(record)
    rendered, plot_version = render_figures(record, fictional=fictional) if figures else ({}, None)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise ValueError("new report directory required; no overwrite")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    header = ["modality", "metric", "n_total", "n_valid", "mean", "population_sd", "median", "bootstrap95_low", "bootstrap95_high"]
    groups = {
        "gradcam_vs_hirescam": [f"{m}.{key}" for key in COMPARABLE for m in METHODS],
        "paired_method_differences": [f"paired.hirescam_minus_gradcam.{key}" for key in COMPARABLE],
        "parameter_randomization": [f"{m}.{key}" for key in SANITY for m in METHODS],
        "logit_contributions": [key for key in METRIC_NAMES if key.startswith(("decomposition.", "reference."))],
        "temporal_profile": [f"temporal.frame_{i:02d}" for i in range(40)],
        "temporal_windows": [key for key in METRIC_NAMES if key.startswith("temporal.window_")],
        "nuisance_and_degeneracy": [key for key in METRIC_NAMES if ".nuisance." in key or ".map_" in key or key.endswith(".degenerate") or key.startswith(("boundary.", "resolution."))],
    }
    for name, keys in groups.items():
        _write(output/f"{name}.csv", _csv(header, table_rows(record, keys)))
    missing_rows = [[record["mode"], key, r["n_total"], r["n_valid"], reason, count]
                    for key, r in record["summary"].items() for reason, count in r["missing"].items() if count]
    _write(output/"missingness.csv", _csv(["modality", "metric", "n_total", "n_valid", "reason", "count"], missing_rows))
    for name, content in rendered.items():
        _write(output/name, content)
    metadata = {k: record[k] for k in ("schema", "mode", "source_revision", "protocol_sha256", "dependencies",
                                      "validation_count", "outer_test_evaluated", "between_seed_stability_assessed")}
    metadata.update(fictional_synthetic_example=fictional, matplotlib_version=plot_version,
                    bootstrap={"seed": 630, "resamples": 2000, "unit": "patient", "interval": "percentile95_pointwise"},
                    suppression_minimum_valid=5, target="baseline_predicted_class_support_logit",
                    temporal_order_learned=False, cohort="original_fold0_reused_development_validation")
    _write(output/"reproducibility.json", json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
    label = "FICTIONAL SYNTHETIC EXAMPLE — NO PATIENT RESULTS" if fictional else "Reused development validation n=23"
    _write(output/"README.md", f"# {label}\n\n"
           "Tables contain patient-first descriptive aggregates. Blank values are NA.\n"
           "Only paired-method differences use jointly valid cases. Each row reports its denominator.\n"
           "Values with fewer than five valid cases are suppressed. Zero/invalid maps and missing masks remain in missingness counts.\n"
           "The fitted intercept coefficient is withheld; exact decomposition remains private.\n"
           "Intervals describe this reused cohort; they do not establish independent generalization.\n"
           "Temporal curves measure frame content, not learned order or clinical phases.\n"
           "Randomization tests dependence on trained parameters. Between-seed stability was not assessed.\n"
           "Attribution-to-lesion correspondence is descriptive; perturbation responses are artificial interventions.\n"
           "No clinical validity or causal claim follows from these outputs.\n")
    return output


def private_report(run_id, modality, report_id, record, *, figures=False):
    from .xai_runtime import run_path
    from .ncct_private_output import _validate_base
    if not re.fullmatch(r"report_[a-zA-Z0-9_-]{1,40}", report_id or ""):
        raise ValueError("new non-identifying report identity required")
    directory = run_path(run_id, modality, resume=True).fold_dir
    _validate_base(directory)
    return write_report(record, directory/report_id, figures=figures)
