import os
import tempfile
from pathlib import Path
import importlib.util

import pytest

# Load module by file path to avoid sys.path/package issues
_mod_path = Path(__file__).resolve().parents[2] / 'studies' / 'extract_cs_ground_truth_from_checker.py'
spec = importlib.util.spec_from_file_location('extract_cs_ground_truth_from_checker', str(_mod_path))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore
infer_type = mod.infer_type
parse_warnings_to_gt = mod.parse_warnings_to_gt


def test_infer_type_rules():
    assert infer_type("must be > 0") == "@Positive"
    assert infer_type("value should be >= 0") == "@NonNegative"
    assert infer_type("index must be >= -1") == "@GTENegativeOne"
    assert infer_type("") == ""
    assert infer_type("some unrelated message") == ""


def test_parse_warnings_to_gt_basic():
    content = (
        "/tmp/Foo.java:10:5: compiler.err: [index] must be > 0\n"
        "/tmp/Bar.java:20:7: compiler.warn: [index] value should be >= 0\n"
        "/tmp/Baz.java:30:9: compiler.err: [index] index must be >= -1\n"
        "/tmp/Qux.java:40:9: compiler.err: [index] message not matching\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "warnings.out"
        p.write_text(content)
        records = parse_warnings_to_gt(p)
        # Expect 3 files captured with 1 annotation each
        m = {r["file_path"]: r for r in records}
        assert "/tmp/Foo.java" in m and m["/tmp/Foo.java"]["annotations"][0]["type"] == "@Positive"
        assert "/tmp/Bar.java" in m and m["/tmp/Bar.java"]["annotations"][0]["type"] == "@NonNegative"
        assert "/tmp/Baz.java" in m and m["/tmp/Baz.java"]["annotations"][0]["type"] == "@GTENegativeOne"
        # Non-matching message should not appear
        assert "/tmp/Qux.java" not in m
