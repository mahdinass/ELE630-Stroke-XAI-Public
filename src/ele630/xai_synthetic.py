"""Fictional fixtures only; never reads a dataset or a checkpoint."""
from __future__ import annotations

import numpy as np


def fictional_state(seed=630):
    rng = np.random.default_rng(seed)
    return {"mean": rng.normal(size=128), "scale": rng.uniform(.5, 2, 128),
            "pca_mean": rng.normal(size=128), "components": rng.normal(size=(8, 128)) * .1,
            "coef": rng.normal(size=19) * .1, "intercept": np.array([.2])}
