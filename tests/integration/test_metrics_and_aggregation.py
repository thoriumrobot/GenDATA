from pathlib import Path
import json
import importlib.util


def test_metrics_and_aggregation(tmp_path, monkeypatch, capsys):
    # Arrange case_studies with GT and predictions
    cs = tmp_path / 'case_studies'
    proj = 'guava'
    pdir = cs / proj
    pdir.mkdir(parents=True)
    # GT: one positive label on line 10
    gt = [{
        'file_path': str(pdir / 'Foo.java'),
        'annotations': [{'line': 10, 'type': '@Positive', 'target': 'unknown'}]
    }]
    (pdir / 'ground_truth.json').write_text(json.dumps(gt))
    # Predictions: close-by within window
    preds = [{
        'file_path': str(pdir / 'Foo.java'),
        'predictions': [{'line': 9, 'type': '@Positive', 'confidence': 0.8}]
    }]
    (pdir / 'predictions_gcn.json').write_text(json.dumps(preds))

    # Import modules by file path
    base = Path(__file__).resolve().parents[2]
    spec_m = importlib.util.spec_from_file_location('compute_case_study_metrics', str(base / 'studies' / 'compute_case_study_metrics.py'))
    metrics = importlib.util.module_from_spec(spec_m)
    assert spec_m and spec_m.loader
    spec_m.loader.exec_module(metrics)  # type: ignore

    spec_c = importlib.util.spec_from_file_location('case_study_metrics_collector', str(base / 'studies' / 'case_study_metrics_collector.py'))
    collector = importlib.util.module_from_spec(spec_c)
    assert spec_c and spec_c.loader
    spec_c.loader.exec_module(collector)  # type: ignore

    spec_r = importlib.util.spec_from_file_location('generate_case_study_comparison', str(base / 'studies' / 'generate_case_study_comparison.py'))
    report = importlib.util.module_from_spec(spec_r)
    assert spec_r and spec_r.loader
    spec_r.loader.exec_module(report)  # type: ignore

    monkeypatch.chdir(tmp_path)
    metrics.main()

    # Assert per-project per-model metrics file exists and shows non-zero coverage
    out = tmp_path / 'case_studies' / 'evaluation_results' / f'{proj}_gcn_metrics.json'
    assert out.exists()
    data = json.loads(out.read_text())
    # Accuracy should be non-zero given a within-window match
    assert data.get('accuracy_partial', 0.0) >= 0.5

    # Aggregate and report
    collector.main()
    agg = tmp_path / 'case_studies' / 'evaluation_results' / 'aggregate_metrics.json'
    assert agg.exists()

    report.main()


