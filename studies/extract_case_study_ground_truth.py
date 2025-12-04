#!/usr/bin/env python3
"""
Extract ground truth annotation occurrences from case study projects.

Approach:
- Scan Java sources for known lower-bound annotation types used in Index Checker:
  @Positive, @NonNegative, @GTENegativeOne.
- Record file path, line number, and type. Target kind is best-effort (method/parameter/variable)
  based on a simple heuristic; downstream evaluation can use line-level matching.

Outputs:
- case_studies/{project}/ground_truth.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List


ANNOTATION_PATTERNS = {
    '@Positive': re.compile(r'@\s*Positive\b'),
    '@NonNegative': re.compile(r'@\s*NonNegative\b'),
    '@GTENegativeOne': re.compile(r'@\s*GTENegativeOne\b'),
}


def classify_target(line: str) -> str:
    """Best-effort target classification: method | parameter | variable."""
    # Parameter: annotation before an identifier inside parentheses
    if '(' in line and ')' in line:
        return 'parameter'
    # Method: annotation on a line with method-like signature
    if '(' in line and '{' in line:
        return 'method'
    return 'variable'


def extract_from_java(java_path: Path) -> List[Dict]:
    items: List[Dict] = []
    try:
        text = java_path.read_text(errors='ignore')
    except Exception:
        return items
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        for atype, pat in ANNOTATION_PATTERNS.items():
            if pat.search(line):
                items.append({
                    'line': idx,
                    'type': atype,
                    'target': classify_target(line)
                })
    return items


def write_ground_truth_for_project(project_dir: Path) -> Path:
    records: List[Dict] = []
    for java_path in project_dir.rglob('*.java'):
        found = extract_from_java(java_path)
        if not found:
            continue
        records.append({
            'file_path': str(java_path),
            'annotations': found,
        })
    out_path = project_dir / 'ground_truth.json'
    out_path.write_text(json.dumps(records, indent=2))
    return out_path


def main():
    base = Path('case_studies')
    projects = [p for p in base.iterdir() if p.is_dir() and p.name in ['guava', 'jfreechart', 'plume-lib']]
    outputs: Dict[str, str] = {}
    for proj in projects:
        out = write_ground_truth_for_project(proj)
        outputs[proj.name] = str(out)
        print(f"WROTE: {out}")
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()


