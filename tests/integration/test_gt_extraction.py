import os
from pathlib import Path
import importlib.util
import types


def test_extract_ground_truth_from_checker(tmp_path, monkeypatch):
    # Arrange a temporary case_studies structure
    cs = tmp_path / 'case_studies'
    (cs / 'guava' / 'src').mkdir(parents=True)
    (cs / 'jfreechart' / 'src').mkdir(parents=True)
    (cs / 'plume-lib' / 'src').mkdir(parents=True)

    # Fake checker runner: write warnings file with simple messages
    mod_path = Path(__file__).resolve().parents[2] / 'studies' / 'extract_cs_ground_truth_from_checker.py'
    spec = importlib.util.spec_from_file_location('extract_cs_ground_truth_from_checker', str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore

    def fake_run_checker_on_project(project_root: str, out_file: str) -> bool:
        sample = f"{project_root}/Foo.java:10:5: compiler.err: [index] must be > 0\n"
        Path(out_file).write_text(sample)
        return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, 'run_checker_on_project', fake_run_checker_on_project)

    # Act
    mod.main()

    # Assert ground_truth.json exists with at least one label per project
    for proj in ['guava', 'jfreechart', 'plume-lib']:
        gt = cs / proj / 'ground_truth.json'
        assert gt.exists(), f"missing gt for {proj}"
        data = gt.read_text()
        assert '"@Positive"' in data or '"@NonNegative"' in data or '"@GTENegativeOne"' in data


