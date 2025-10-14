import os
import tempfile
from pathlib import Path

from enhanced_semantic_augment_slices import EnhancedSemanticTransformer


def test_visible_change_demo_java():
    root = Path(__file__).parent
    demo_src = (root / 'src' / 'Demo.java')
    assert demo_src.exists(), 'Demo.java not found'

    transformer = EnhancedSemanticTransformer(seed=1337)
    out = transformer.transform_file(str(demo_src), variant_idx=0)

    with open(demo_src, 'r') as f:
        original = f.read()

    # Strip headers if present in output and compare
    out_lines = out.splitlines()
    if out_lines[:3] and out_lines[0].startswith('/* CFWR enhanced semantic augmentation'):
        out_body = '\n'.join(out_lines[4:])
    else:
        out_body = out

    assert out_body != original, 'Expected visible body difference after augmentation'


