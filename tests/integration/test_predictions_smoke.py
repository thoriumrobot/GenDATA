import sys
from pathlib import Path
import importlib.util
import types


def test_run_annotation_type_predictions_smoke(tmp_path, monkeypatch):
    # Prepare case_studies minimal tree
    cs = tmp_path / 'case_studies'
    for proj in ['guava', 'jfreechart', 'plume-lib']:
        p = cs / proj / 'src'
        p.mkdir(parents=True)
        # create tiny java file
        (p / 'A.java').write_text('class A { int x = 1; }')

    # Build a fake model_based_predictor module returning constant predictions
    fake = types.ModuleType('model_based_predictor')

    class FakePredictor:
        def __init__(self, models_dir=None, auto_train=True, device='cpu'):
            pass
        def load_or_train_models(self, base_model_type='gcn', episodes=1, project_root=None):
            return True
        def predict_annotations_for_file_with_cfg(self, java_file, cfg_root, threshold=0.3):
            return [{
                'line': 1,
                'annotation_type': '@NonNegative',
                'confidence': 0.9
            }]

    fake.ModelBasedPredictor = FakePredictor
    sys.modules['model_based_predictor'] = fake

    # Import runner and predictor main by file path before chdir
    runner_path = Path(__file__).resolve().parents[2] / 'studies' / 'run_annotation_type_predictions.py'
    spec_r = importlib.util.spec_from_file_location('run_annotation_type_predictions', str(runner_path))
    runner = importlib.util.module_from_spec(spec_r)
    assert spec_r and spec_r.loader
    spec_r.loader.exec_module(runner)  # type: ignore

    pcsf_path = Path(__file__).resolve().parents[2] / 'predict_case_studies_fixed.py'
    spec_p = importlib.util.spec_from_file_location('predict_case_studies_fixed', str(pcsf_path))
    pcsf = importlib.util.module_from_spec(spec_p)
    assert spec_p and spec_p.loader
    spec_p.loader.exec_module(pcsf)  # type: ignore

    # Monkeypatch runner.run to avoid real subprocess calls
    def fake_run(cmd):
        # Simulate CFG generation regardless of command content
        out_dir = tmp_path / 'case_study_cfg_output'
        (out_dir / 'A').mkdir(parents=True, exist_ok=True)
        (out_dir / 'A' / 'cfg.json').write_text('{"nodes":[],"edges":[]}')
        # Simulate writing predictions for all models
        for proj in ['guava', 'jfreechart', 'plume-lib']:
            pdir = cs / proj
            for model in ['gcn','hgt','gbt','causal','gcsn','dg2n','dgcrf']:
                out = pdir / f'predictions_{model}.json'
                out.write_text('[{"file_path":"' + str((pdir / 'src' / 'A.java')) + '","predictions":[{"line":1,"type":"@NonNegative","confidence":0.9}]}]')
        return 0

    monkeypatch.setattr(runner, 'run', fake_run)
    # Run predictor script in the temp cwd
    monkeypatch.chdir(tmp_path)
    rc = runner.main()
    assert rc == 0

    # Verify standardized prediction files exist and contain entries
    for proj in ['guava', 'jfreechart', 'plume-lib']:
        out = cs / proj / 'predictions_gcn.json'  # one of the models
        assert out.exists()
        data = out.read_text()
        assert '"predictions"' in data


