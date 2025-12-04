from pathlib import Path
import json
import importlib.util


def test_validate_summary(tmp_path, monkeypatch, capsys):
    # Minimal inputs: create evaluation_results with metrics so validator runs
    cs = tmp_path / 'case_studies'
    (cs / 'evaluation_results').mkdir(parents=True)
    # Also create project dirs and minimal gt/pred files for sampling
    for proj in ['guava', 'jfreechart', 'plume-lib']:
        pdir = cs / proj
        pdir.mkdir(parents=True)
        (pdir / 'ground_truth.json').write_text('[]')
        (pdir / 'predictions_gcn.json').write_text('[]')

    base = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('validate_case_study_eval', str(base / 'studies' / 'validate_case_study_eval.py'))
    val = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(val)  # type: ignore
    monkeypatch.chdir(tmp_path)
    val.main()
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert 'guava' in data and 'preds' in data['guava']


