#!/usr/bin/env python3
"""
Extract case study ground truth from Checker Framework warnings.

For each project under case_studies/{guava,jfreechart,plume-lib}:
- Run the Index Checker via CheckerFrameworkRunner
- Parse warnings to infer annotation type per location using simple rules:
  - messages with "> 0" → @Positive
  - messages with ">= 0" or "nonnegative" → @NonNegative
  - messages with ">= -1" → @GTENegativeOne
- Emit case_studies/{project}/ground_truth.json in the standard schema.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List


def run_checker_on_project(project_root: str, out_file: str) -> bool:
    try:
        from checker_framework_runner import run_checker_framework_on_project
        return run_checker_framework_on_project(project_root=project_root, output_file=out_file, max_files=None)
    except Exception as e:
        print(f"ERROR: failed to run checker on {project_root}: {e}")
        return False


GT_RULES = [
    (re.compile(r">\s*0|strictly positive|must be positive", re.I), "@Positive"),
    (re.compile(r">=\s*0|nonnegative|must be non[- ]?negative", re.I), "@NonNegative"),
    (re.compile(r">=\s*-?1|greater than or equal to -1", re.I), "@GTENegativeOne"),
]


def infer_type(msg: str) -> str:
    for pat, at in GT_RULES:
        if pat.search(msg or ""):
            return at
    return ""


def parse_warnings_to_gt(warnings_path: Path) -> List[Dict]:
    records: Dict[str, Dict] = {}
    try:
        text = warnings_path.read_text(errors='ignore')
    except Exception:
        return []

    for line in text.splitlines():
        # Expected javac-like lines: /abs/path/File.java:LINE:COL: ... MESSAGE
        if not line or line.startswith('#'):
            continue
        if '.java:' not in line:
            continue
        try:
            path_part, rest = line.split('.java:', 1)
            file_path = f"{path_part}.java"
            parts = rest.split(':', 2)
            if len(parts) < 2:
                continue
            line_no = int(parts[0])
            msg = parts[-1]
            atype = infer_type(msg)
            if not atype:
                continue
            rec = records.setdefault(file_path, {"file_path": file_path, "annotations": []})
            rec["annotations"].append({"line": line_no, "type": atype, "target": "unknown"})
        except Exception:
            continue

    return list(records.values())


def main():
    root = Path.cwd()
    cs_root = root / 'case_studies'
    projects = ['guava', 'jfreechart', 'plume-lib']
    for proj in projects:
        proj_dir = cs_root / proj
        if not proj_dir.exists():
            continue
        warnings_file = proj_dir / 'checker_warnings.out'
        ok = run_checker_on_project(str(proj_dir), str(warnings_file))
        if not ok:
            print(f"WARN: Checker run failed for {proj}")
        gt = parse_warnings_to_gt(warnings_file) if warnings_file.exists() else []
        out_path = proj_dir / 'ground_truth.json'
        out_path.write_text(json.dumps(gt, indent=2))
        print(f"WROTE: {out_path} with {sum(len(r['annotations']) for r in gt)} labels")


if __name__ == '__main__':
    main()


