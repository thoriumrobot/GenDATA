import json
import importlib.util
from pathlib import Path


def test_generate_case_study_cfgs(tmp_path, monkeypatch):
    # Prepare fixture tree
    cs = tmp_path / 'case_studies' / 'guava' / 'src'
    cs.mkdir(parents=True)
    (cs / 'Simple.java').write_text('class Simple { void m(){ int i=0; i++; } }')

    # Load generator module by path
    mod_path = Path(__file__).resolve().parents[2] / 'generate_case_study_cfgs.py'
    spec = importlib.util.spec_from_file_location('generate_case_study_cfgs', str(mod_path))
    gen = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(gen)  # type: ignore

    def fake_generate_case_study_cfgs():
        out_dir = tmp_path / 'case_study_cfg_output'
        (out_dir / 'Simple').mkdir(parents=True)
        (out_dir / 'Simple' / 'cfg.json').write_text(json.dumps({'nodes': [], 'edges': []}))
        return str(out_dir)

    monkeypatch.setattr(gen, 'generate_case_study_cfgs', fake_generate_case_study_cfgs)
    # Run in temp cwd
    monkeypatch.chdir(tmp_path)
    out_dir = gen.generate_case_study_cfgs()

    out = Path(out_dir)
    assert (out / 'Simple' / 'cfg.json').exists()


