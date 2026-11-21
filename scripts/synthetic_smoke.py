"""Exercise the NCCT, CTP, clinical fusion and CAM paths on generated inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from ele630.ctp_model import CompactCTPMean
from ele630.fusion import run_fusion
from ele630.fusion_features import freeze
from ele630.fusion_synthetic import FictionalCTP, fictional_plan, fictional_rows
from ele630.resnet3d import CompactResNet10
from ele630.xai_fused_score import torch_state
from ele630.xai_spatial import ctp_cam, ncct_cam
from ele630.xai_synthetic import fictional_state


def main():
    torch.manual_seed(630)
    torch.set_num_threads(1)

    clinical = torch.zeros(1, 11)
    attribution_state = torch_state(fictional_state(), "cpu")
    volume = torch.rand(1, 1, 8, 32, 32)
    series = FictionalCTP(630, (8, 32, 32))
    results = (
        ncct_cam(freeze(CompactResNet10()), volume, clinical, attribution_state),
        ctp_cam(freeze(CompactCTPMean()), series, clinical, attribution_state),
    )
    for result in results:
        if not np.isfinite(result["logit"]) or set(result["maps"]) != {"gradcam", "hirescam"}:
            raise RuntimeError("synthetic attribution check failed")
        if not all(np.isfinite(maps["positive"]).all() for maps in result["maps"].values()):
            raise RuntimeError("synthetic attribution contained nonfinite values")

    plan = fictional_plan()
    rng = np.random.default_rng(630)
    training = rng.normal(size=(len(plan.inner_train), 128))
    validation = rng.normal(size=(len(plan.inner_validation), 128))
    run_fusion(plan, fictional_rows(plan), training, validation, [.3] * len(validation))
    print("Synthetic CPU smoke passed for NCCT, CTP, clinical fusion, Grad-CAM and HiResCAM.")


if __name__ == "__main__":
    main()
