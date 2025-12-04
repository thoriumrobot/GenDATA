import importlib.util
from pathlib import Path

_mod_path = Path(__file__).resolve().parents[2] / 'studies' / 'compute_case_study_metrics.py'
spec = importlib.util.spec_from_file_location('compute_case_study_metrics', str(_mod_path))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore
align_labels = mod.align_labels


def test_align_exact_line_match():
    gt_map = {"/f.java": [(10, "@Positive")]}
    pr_map = {"/f.java": [(10, "@Positive")]}  # exact match
    y_true, y_pred = align_labels(gt_map, pr_map, window=3)
    assert y_true == ["@Positive"]
    assert y_pred == ["@Positive"]


def test_align_within_window_prefers_nearest():
    gt_map = {"/f.java": [(20, "@NonNegative")]}  # GT at 20
    # Predictions at 18 and 23, both within window=3; 18 is nearest (2 vs 3)
    pr_map = {"/f.java": [(18, "@NonNegative"), (23, "@Positive")]}
    y_true, y_pred = align_labels(gt_map, pr_map, window=3)
    assert y_true == ["@NonNegative"]
    assert y_pred == ["@NonNegative"]


def test_align_outside_window_yields_none():
    gt_map = {"/f.java": [(50, "@GTENegativeOne")]}
    pr_map = {"/f.java": [(46, "@GTENegativeOne")]}  # distance 4, window 3
    y_true, y_pred = align_labels(gt_map, pr_map, window=3)
    assert y_true == ["@GTENegativeOne"]
    assert y_pred == ["NONE"]
