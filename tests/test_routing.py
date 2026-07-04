"""Tests for missing-core fallback routing (paper 3.4.2)."""
import glob
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import (  # noqa: E402
    SCORES_LUXPARK, core_presence_route, fallback_scores,
    get_fallback_model_paths, get_fallback_conformal_paths,
)

SC = set(SCORES_LUXPARK)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Trained models are not distributed publicly (DUA); the artifact-existence test
# skips when they are absent (the routing logic itself is tested without them).
_needs_models = pytest.mark.skipif(
    not glob.glob(os.path.join(_ROOT, "models", "*.joblib")),
    reason="trained models not distributed (DUA); see README")


def test_no_route_when_full():
    assert core_presence_route(SC) is None


def test_no_route_when_single_core_missing():
    # at most one of the three core parts missing -> default model (imputation OK)
    assert core_presence_route(SC - {"UPDRS1"}) is None
    assert core_presence_route(SC - {"UPDRS2"}) is None
    assert core_presence_route(SC - {"UPDRS3_off", "UPDRS3_on"}) is None


def test_routes_when_two_or_more_core_missing():
    assert core_presence_route(SC - {"UPDRS1", "UPDRS2"}) == "coreIII"
    assert core_presence_route(SC - {"UPDRS2", "UPDRS3_off", "UPDRS3_on"}) == "coreI"
    assert core_presence_route(SC - {"UPDRS1", "UPDRS3_off", "UPDRS3_on"}) == "coreII"
    assert core_presence_route(
        SC - {"UPDRS1", "UPDRS2", "UPDRS3_off", "UPDRS3_on"}) == "coreNone"


def test_fallback_scores_drop_the_missing_core():
    kept = fallback_scores(list(SCORES_LUXPARK), "coreIII")
    assert "UPDRS1" not in kept and "UPDRS2" not in kept
    assert "UPDRS3_on" in kept  # the surviving core part is retained
    assert "MOCA" in kept       # periphery retained


@_needs_models
def test_every_fallback_artifact_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mode in ("luxpark", "full"):
        for n_visits in (1, 2):
            for pattern in ("coreI", "coreII", "coreIII", "coreNone"):
                m = get_fallback_model_paths(mode, n_visits, pattern)
                c = get_fallback_conformal_paths(mode, n_visits, pattern)
                for d in (m, c):
                    for path in d.values():
                        assert os.path.exists(os.path.join(root, path)), path
