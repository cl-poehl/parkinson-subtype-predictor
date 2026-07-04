"""Tests for the Venn-Abers per-patient probability interval."""
import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import vennabers  # noqa: E402
from src.constants import (  # noqa: E402
    get_vennabers_paths, get_fallback_vennabers_paths,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Trained models are not distributed publicly (DUA); artifact-existence skips.
_needs_models = pytest.mark.skipif(
    not glob.glob(os.path.join(_ROOT, "models", "*.joblib")),
    reason="trained models not distributed (DUA); see README")


def test_interval_is_valid_and_ordered():
    state = vennabers.fit([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.5, 0.4],
                          [0, 0, 0, 1, 1, 1, 0, 1])
    for lo, hi, point in vennabers.predict_intervals(state, [0.05, 0.5, 0.95]):
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= point <= hi or abs(point - hi) < 1e-9 or abs(point - lo) < 1e-9


def test_higher_score_gives_higher_interval():
    state = vennabers.fit([0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
                          [0, 0, 0, 1, 1, 1])
    (lo_lo, hi_lo, _), (lo_hi, hi_hi, _) = vennabers.predict_intervals(
        state, [0.1, 0.9])
    assert lo_hi >= lo_lo and hi_hi >= hi_lo


@_needs_models
def test_every_vennabers_artifact_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mode in ("luxpark", "full"):
        for n_visits in (1, 2):
            for path in get_vennabers_paths(mode, n_visits).values():
                assert os.path.exists(os.path.join(root, path)), path
            for pattern in ("coreI", "coreII", "coreIII", "coreNone"):
                for path in get_fallback_vennabers_paths(
                        mode, n_visits, pattern).values():
                    assert os.path.exists(os.path.join(root, path)), path
