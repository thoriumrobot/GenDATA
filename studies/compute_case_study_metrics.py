#!/usr/bin/env python3
"""
Compute evaluation metrics for case study predictions vs. ground truth.

Metrics:
- Accuracy (with optional partial credit for @Positive <-> @NonNegative swaps)
- Precision / Recall / F1 (macro and weighted)
- Confusion matrix
- Coverage (fraction of GT annotations matched by any prediction)

Inputs:
- case_studies/{project}/ground_truth.json
- case_studies/{project}/predictions_{model}.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


TARGET_ANNOTATIONS = ['@Positive', '@NonNegative', '@GTENegativeOne']


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _normalize_path(fp: str) -> str:
    """Normalize file path for matching: make relative to case_studies if absolute."""
    if not fp:
        return fp
    fp = str(fp)
    # Remove absolute path prefix if present
    if fp.startswith('/'):
        # Try to find case_studies in path
        parts = fp.split('/')
        if 'case_studies' in parts:
            idx = parts.index('case_studies')
            fp = '/'.join(parts[idx:])
    # Ensure it starts with case_studies
    if not fp.startswith('case_studies'):
        # Try to make relative
        if 'case_studies' in fp:
            idx = fp.find('case_studies')
            fp = fp[idx:]
    return fp


def flatten_annotations(records: List[Dict]) -> Dict[str, List[Tuple[int, str]]]:
    """Map file -> list of (line, type) for GT."""
    out: Dict[str, List[Tuple[int, str]]] = {}
    for r in records or []:
        fp = _normalize_path(r.get('file_path'))
        for ann in r.get('annotations', []):
            line = ann.get('line')
            atype = ann.get('type')
            if fp and isinstance(line, int) and isinstance(atype, str):
                out.setdefault(fp, []).append((line, atype))
    return out


def flatten_predictions(records: List[Dict]) -> Dict[str, List[Tuple[int, str]]]:
    out: Dict[str, List[Tuple[int, str]]] = {}
    for r in records or []:
        fp = _normalize_path(r.get('file_path'))
        for pred in r.get('predictions', []):
            line = pred.get('line')
            atype = pred.get('type')
            if fp and isinstance(line, int) and isinstance(atype, str):
                out.setdefault(fp, []).append((line, atype))
    return out


def align_labels(gt_map: Dict[str, List[Tuple[int, str]]], pr_map: Dict[str, List[Tuple[int, str]]], window: int = 3) -> Tuple[List[str], List[str]]:
    """Robust line matching with ±window tolerance. For each GT (file,line,type),
    choose the nearest prediction line within the window. If multiple candidates,
    pick the same-line first, else the minimum distance.
    LOCALIZATION FIX: Also check ±1 line for exact label matches to account for CFG offset.
    Unmatched → predicted label 'NONE'."""
    y_true: List[str] = []
    y_pred: List[str] = []
    for fp, gt_list in gt_map.items():
        preds = pr_map.get(fp, [])
        # Build map line→types (dedup types per line)
        by_line: Dict[int, List[str]] = {}
        for ln, at in preds:
            if ln is None:
                continue
            lst = by_line.setdefault(int(ln), [])
            if at not in lst:
                lst.append(at)
        lines = sorted(by_line.keys())
        for ln, at in gt_list:
            y_true.append(at)
            # Exact match first
            if ln in by_line and by_line[ln]:
                y_pred.append(by_line[ln][0])
                continue
            
            # LOCALIZATION FIX: Check ±1 line for same label (accounts for CFG offset)
            # This helps when CFG node is 1 line off but label is correct
            best_same_label = None
            best_same_label_dist = None
            for offset in [-1, 1]:
                cand_line = ln + offset
                if cand_line in by_line:
                    for pred_label in by_line[cand_line]:
                        if pred_label == at:
                            best_same_label = cand_line
                            best_same_label_dist = abs(offset)
                            break
                    if best_same_label:
                        break
            
            if best_same_label:
                y_pred.append(by_line[best_same_label][0])
                continue
            
            # Window search (original logic)
            best = None
            best_dist = None
            for cand in lines:
                d = abs(cand - ln)
                if d <= window:
                    if best is None or d < best_dist:
                        best = cand
                        best_dist = d
                        if d == 0:
                            break
            if best is not None and by_line[best]:
                y_pred.append(by_line[best][0])
            else:
                y_pred.append('NONE')
    return y_true, y_pred


def align_labels_with_diagnostics(gt_map: Dict[str, List[Tuple[int, str]]], pr_map: Dict[str, List[Tuple[int, str]]], window: int = 3) -> Tuple[List[str], List[str], Dict]:
    """Same as align_labels but also returns diagnostic information about match types."""
    y_true: List[str] = []
    y_pred: List[str] = []
    diagnostics = {
        'exact_line_match': 0,
        'near_match_same_label': 0,
        'near_match_pos_vs_nn': 0,
        'near_match_wrong_label': 0,
        'no_match': 0,
        'distances': []
    }
    
    for fp, gt_list in gt_map.items():
        preds = pr_map.get(fp, [])
        by_line: Dict[int, List[str]] = {}
        for ln, at in preds:
            if ln is None:
                continue
            lst = by_line.setdefault(int(ln), [])
            if at not in lst:
                lst.append(at)
        lines = sorted(by_line.keys())
        
        for ln, at in gt_list:
            y_true.append(at)
            matched = False
            best = None
            best_dist = None
            best_label = None
            
            # Exact match first
            if ln in by_line and by_line[ln]:
                best = ln
                best_dist = 0
                best_label = by_line[ln][0]
                matched = True
            else:
                # Window search
                for cand in lines:
                    d = abs(cand - ln)
                    if d <= window:
                        if best is None or d < best_dist:
                            best = cand
                            best_dist = d
                            best_label = by_line[cand][0]
                            matched = True
                            if d == 0:
                                break
            
            if matched and best is not None:
                y_pred.append(best_label)
                diagnostics['distances'].append(best_dist)
                if best_dist == 0:
                    if best_label == at:
                        diagnostics['exact_line_match'] += 1
                    else:
                        diagnostics['near_match_wrong_label'] += 1
                else:
                    if best_label == at:
                        diagnostics['near_match_same_label'] += 1
                    elif {best_label, at} == {'@Positive', '@NonNegative'}:
                        diagnostics['near_match_pos_vs_nn'] += 1
                    else:
                        diagnostics['near_match_wrong_label'] += 1
            else:
                y_pred.append('NONE')
                diagnostics['no_match'] += 1
    
    return y_true, y_pred, diagnostics


def partial_credit_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return 0.0
    score = 0.0
    for t, p in pairs:
        if t == p:
            score += 1.0
        elif (t == '@Positive' and p == '@NonNegative') or (t == '@NonNegative' and p == '@Positive'):
            score += 0.5
    return score / len(pairs)


def evaluate_project_model(project: str, model: str) -> Dict:
    base = Path('case_studies') / project
    gt_path = base / 'ground_truth.json'
    pr_path = base / f'predictions_{model}.json'
    gt = load_json(gt_path) or []
    pr = load_json(pr_path) or []
    
    # Load CFG index to filter to files that have both GT and CFGs
    cfg_index_path = Path('case_study_cfg_output/index.json')
    cfg_files = set()
    if cfg_index_path.exists():
        cfg_index = json.load(open(cfg_index_path))
        cfg_files = {Path(k).resolve() for k in cfg_index.keys()}
    
    # Filter GT to only include files that have CFGs
    if cfg_files:
        gt_filtered = []
        for r in gt:
            fp = r.get('file_path', '')
            if fp:
                fp_resolved = Path(fp).resolve()
                if fp_resolved in cfg_files:
                    gt_filtered.append(r)
        gt = gt_filtered
    
    gt_map = flatten_annotations(gt)
    pr_map = flatten_predictions(pr)
    # If no GT, mark and return empty metrics
    total_gt = sum(len(v) for v in gt_map.values())
    if total_gt == 0:
        return {
            'project': project,
            'model': model,
            'num_ground_truth': 0,
            'num_predictions': sum(len(v) for v in pr_map.values()),
            'note': 'no_gt',
            'accuracy_exact': 0.0,
            'accuracy_partial': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'coverage': 0.0,
            'confusion_matrix_labels': TARGET_ANNOTATIONS + ['NONE'],
            'confusion_matrix': [],
            'classification_report': {}
        }

    y_true, y_pred = align_labels(gt_map, pr_map)
    # Also compute diagnostics
    _, _, diagnostics = align_labels_with_diagnostics(gt_map, pr_map)
    
    labels = TARGET_ANNOTATIONS + ['NONE']
    # Standard metrics on exact labels
    acc_exact = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0) if y_true else 0.0
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_true else 0.0
    prec = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_true else 0.0
    rec = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist() if y_true else []
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0) if y_true else {}
    # Partial credit accuracy for near-miss swaps
    acc_partial = partial_credit_accuracy(y_true, y_pred)
    # Coverage: GT entries that had any prediction on their line
    covered = 0
    total = 0
    for fp, gt_list in gt_map.items():
        pred_lines = {ln for ln, _ in pr_map.get(fp, [])}
        for ln, _ in gt_list:
            total += 1
            if ln in pred_lines:
                covered += 1
    coverage = (covered / total) if total else 0.0
    # Compute diagnostic accuracy with ±1 line tolerance for exact matches
    # (to account for CFG node line offsets)
    # This counts matches where label is correct and distance is 0 or 1
    acc_exact_plus1 = 0.0
    if total > 0:
        # Count exact matches + same-label matches at distance 1
        exact_or_near1_same = diagnostics['exact_line_match'] + sum(
            1 for d in diagnostics['distances'] 
            if d == 1
        )
        # But we need to check if those distance-1 matches actually had correct labels
        # For now, approximate: exact_line_match + near_match_same_label where distance <= 1
        # (The diagnostics don't track distance per match type, so this is approximate)
        acc_exact_plus1 = (diagnostics['exact_line_match'] + diagnostics['near_match_same_label']) / total
    
    return {
        'project': project,
        'model': model,
        'num_ground_truth': total,
        'num_predictions': sum(len(v) for v in pr_map.values()),
        'accuracy_exact': acc_exact,
        'accuracy_partial': acc_partial,
        'precision_weighted': prec,
        'recall_weighted': rec,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'coverage': coverage,
        'confusion_matrix_labels': labels,
        'confusion_matrix': cm,
        'classification_report': report,
        'diagnostics': {
            'exact_line_match': diagnostics['exact_line_match'],
            'near_match_same_label': diagnostics['near_match_same_label'],
            'near_match_pos_vs_nn': diagnostics['near_match_pos_vs_nn'],
            'near_match_wrong_label': diagnostics['near_match_wrong_label'],
            'no_match': diagnostics['no_match'],
            'avg_distance': sum(diagnostics['distances']) / len(diagnostics['distances']) if diagnostics['distances'] else 0.0,
            'accuracy_exact_plus1_line': acc_exact_plus1,
        },
    }


def main():
    projects = ['guava', 'jfreechart', 'plume-lib']
    models = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
    results_dir = Path('case_studies') / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict] = []
    for proj in projects:
        for model in models:
            res = evaluate_project_model(proj, model)
            out_path = results_dir / f'{proj}_{model}_metrics.json'
            out_path.write_text(json.dumps(res, indent=2))
            print(f"WROTE: {out_path}")
            all_results.append(res)
    (results_dir / 'all_results.json').write_text(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()


