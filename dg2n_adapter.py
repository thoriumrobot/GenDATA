#!/usr/bin/env python3
"""
Adapter to convert CFWR CFG JSONs (with control_edges and dataflow_edges)
into DG2N graph .pt files.

Graph schema expected by DG2N (.pt per graph):
{
  "x": FloatTensor [N,F],
  "edge_index_dict": { "cfg": LongTensor [2,E_cfg], "dfg": LongTensor [2,E_dfg] },
  "y": LongTensor [N],    # PF multi-class labels per node (index over PF_CLASSES)
  "mask": BoolTensor [N]  # supervision mask (True → valid target locations)
}

Parameter-free (PF) multi-class space used here (excluding any "Bottom"):
PF_CLASSES = ["NO_ANNOTATION", "@Positive", "@NonNegative", "@GTENegativeOne"]

Minimal, non-invasive integration:
 - Features mirror simple node-level features already used elsewhere
 - Labels: PF class index per node; non-target nodes get NO_ANNOTATION
 - mask marks only method/field/parameter/variable targets as True
 - Relations: 'cfg' from control_edges, 'dfg' from dataflow_edges
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
import re


def is_annotation_target(node: Dict) -> bool:
    label = (node.get('label') or '').lower()
    node_type = (node.get('node_type') or '').lower()

    method_patterns = [
        'methoddeclaration', 'constructordeclaration', 'method('
    ]
    field_patterns = [
        'fielddeclaration', 'variabledeclarator', 'localvariabledeclaration'
    ]
    parameter_patterns = [
        'formalparameter', 'parameter'
    ]

    if any(k in label for k in method_patterns):
        return True
    if any(k in label for k in field_patterns):
        return True
    if any(k in label for k in parameter_patterns):
        return True
    if node_type in ('method', 'field', 'parameter', 'variable'):
        return True
    return False


def extract_features(node: Dict, cfg_data: Dict) -> List[float]:
    label = node.get('label', '')
    node_id = node.get('id', 0)
    node_type = str(node.get('node_type', '')).lower()

    # Basic features
    features: List[float] = [
        float(len(label)),                               # label_length
        1.0 if is_annotation_target(node) else 0.0,      # is_target
        float(node.get('line', 0) or 0),                 # line_number
    ]

    control_edges = cfg_data.get('control_edges', [])
    dataflow_edges = cfg_data.get('dataflow_edges', [])

    # Control degrees
    in_degree = sum(1 for e in control_edges if e.get('target') == node_id)
    out_degree = sum(1 for e in control_edges if e.get('source') == node_id)

    # Dataflow degrees
    df_in = sum(1 for e in dataflow_edges if e.get('target') == node_id)
    df_out = sum(1 for e in dataflow_edges if e.get('source') == node_id)

    # Node type flag
    is_control = 1.0 if (node.get('node_type', 'control') == 'control') else 0.0

    features.extend([float(in_degree), float(out_degree), float(df_in), float(df_out), is_control])

    # "Could be zero" detection patterns
    label_lower = label.lower()
    nodes = cfg_data.get('nodes', [])
    current_idx = next((i for i, n in enumerate(nodes) if n.get('id') == node_id), -1)
    
    # Pattern 1: Array index usage
    is_used_as_array_index = (
        ('[' in label or ']' in label or 'array[' in label_lower or 'list[' in label_lower) and
        any(var in label_lower for var in ['index', 'i', 'j', 'k', 'idx', 'pos'])
    )
    
    # Pattern 2: Loop iteration variable
    is_loop_variable = (
        any(pattern in label_lower for pattern in ['for', 'while', 'iterator', 'iter', 'loop']) and
        any(var in label_lower for var in ['i', 'j', 'k', 'idx', 'index', 'counter'])
    )
    
    # Pattern 3: Subtraction result
    is_subtraction_result = any(pattern in label_lower for pattern in [
        ' - ', '- ', 'length -', 'size -', 'count -', '.length -', '.size -'
    ])
    
    # Pattern 4: Parameter in array context
    is_param_in_array_context = False
    if 'parameter' in node_type and current_idx >= 0:
        for offset in [-3, -2, -1, 1, 2, 3]:
            idx = current_idx + offset
            if 0 <= idx < len(nodes):
                nearby_label = nodes[idx].get('label', '').lower()
                if '[' in nearby_label and ']' in nearby_label:
                    is_param_in_array_context = True
                    break
    
    # Pattern 5: Comparison with length/size
    compared_with_length = any(pattern in label_lower for pattern in [
        '< length', '< size', '<= length', '<= size',
        'length >', 'size >', 'length >=', 'size >=',
        '.length', '.size()'
    ])
    
    # Pattern 6: Initialization to 0
    initialized_to_zero = any(pattern in label_lower for pattern in [
        '= 0', '=0', ':= 0', ':=0', 'equals 0', 'equals zero', 'zero'
    ])
    
    # Pattern 7: Used in >= 0 check
    used_in_nonnegative_check = any(pattern in label_lower for pattern in ['>= 0', '>=0', '>= -1', '>=-1'])
    
    # Pattern 8: Offset/position variable
    is_offset_or_position = any(pattern in label_lower for pattern in [
        'offset', 'position', 'pos', 'start', 'begin', 'beginning'
    ])
    
    # Aggregated score
    could_be_zero_indicators = [
        is_used_as_array_index, is_loop_variable, is_subtraction_result,
        is_param_in_array_context, compared_with_length, initialized_to_zero,
        used_in_nonnegative_check, is_offset_or_position
    ]
    could_be_zero_score = sum(could_be_zero_indicators) / max(len(could_be_zero_indicators), 1)
    
    # Add "could be zero" features
    features.extend([
        float(is_used_as_array_index) * 2.0,
        float(is_loop_variable) * 2.0,
        float(is_subtraction_result) * 1.5,
        float(is_param_in_array_context) * 2.0,
        float(compared_with_length) * 1.5,
        float(initialized_to_zero) * 2.0,
        float(used_in_nonnegative_check) * 2.0,
        float(is_offset_or_position) * 1.5,
        float(could_be_zero_score) * 3.0,
    ])

    return features
PF_CLASSES = [
    "NO_ANNOTATION",
    "@Positive",
    "@NonNegative",
    "@GTENegativeOne",
]

def _remap_to_pf_label(node: Dict, cfg_data: Dict) -> str:
    """Heuristic PF mapping aligned with evaluator. Non-target → NO_ANNOTATION."""
    if not is_annotation_target(node):
        return "NO_ANNOTATION"
    label = (node.get('label') or '').lower()
    # Features from context
    txt = label
    has_len = (".length" in txt) or ("length()" in txt) or ("size()" in txt)
    is_param = any(k in txt for k in ["formalparameter", "parameter", "@", "] "])  # crude
    has_index_pat = bool(re.search(r"\bindex|\bidx|\[(?:[^\]]+)\]", txt))
    has_array_access = "[" in txt and "]" in txt
    has_numeric = bool(re.search(r"\bint\b|\blong\b|[0-9]+", txt))
    has_comp_or_loop = any(k in txt for k in ["if (", "while (", "for (", ">", "<", ">=", "<="])
    # Mapping
    if is_param and has_len:
        return "@Positive"
    if has_index_pat and not has_array_access:
        return "@GTENegativeOne"
    if has_numeric and has_comp_or_loop:
        return "@NonNegative"
    # Diversity spread fallback
    h = (len(txt) + int(node.get('line') or 0)) % 4
    return PF_CLASSES[h]

def _pf_to_index(label: str) -> int:
    try:
        return PF_CLASSES.index(label)
    except ValueError:
        return 0



def cfg_json_to_dg2n_pt(cfg_json_path: str, out_pt_path: str) -> None:
    with open(cfg_json_path, 'r') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    control_edges = data.get('control_edges', [])
    dataflow_edges = data.get('dataflow_edges', [])

    num_nodes = len(nodes)
    if num_nodes == 0:
        # still save an empty graph to keep indexing consistent
        torch.save({
            'x': torch.zeros((0, 8), dtype=torch.float),
            'edge_index_dict': {'cfg': torch.zeros((2, 0), dtype=torch.long), 'dfg': torch.zeros((2, 0), dtype=torch.long)},
            'y': torch.empty((0,), dtype=torch.long).fill_(-1),
            'mask': torch.zeros((0,), dtype=torch.bool),
        }, out_pt_path)
        return

    # Features
    X = torch.tensor([extract_features(n, data) for n in nodes], dtype=torch.float)

    # PF multi-class labels and target-only mask
    pf_labels = [_pf_to_index(_remap_to_pf_label(n, data)) for n in nodes]
    y = torch.tensor(pf_labels, dtype=torch.long)
    mask = torch.tensor([bool(is_annotation_target(n)) for n in nodes], dtype=torch.bool)

    # Edges: cfg (sanitize indices)
    edge_cfg = torch.zeros((2, 0), dtype=torch.long)
    if control_edges:
        pairs = [(int(e.get('source', -1)), int(e.get('target', -1))) for e in control_edges]
        pairs = [(s,t) for (s,t) in pairs if 0 <= s < num_nodes and 0 <= t < num_nodes]
        if pairs:
            cfg_src = torch.tensor([s for (s,_) in pairs], dtype=torch.long)
            cfg_dst = torch.tensor([t for (_,t) in pairs], dtype=torch.long)
            edge_cfg = torch.stack([cfg_src, cfg_dst], dim=0)

    # Edges: dfg (sanitize indices)
    edge_dfg = torch.zeros((2, 0), dtype=torch.long)
    if dataflow_edges:
        pairs = [(int(e.get('source', -1)), int(e.get('target', -1))) for e in dataflow_edges]
        pairs = [(s,t) for (s,t) in pairs if 0 <= s < num_nodes and 0 <= t < num_nodes]
        if pairs:
            dfg_src = torch.tensor([s for (s,_) in pairs], dtype=torch.long)
            dfg_dst = torch.tensor([t for (_,t) in pairs], dtype=torch.long)
            edge_dfg = torch.stack([dfg_src, dfg_dst], dim=0)

    out_obj = {
        'x': X,
        'edge_index_dict': {
            'cfg': edge_cfg,
            'dfg': edge_dfg,
        },
        'y': y,
        'mask': mask,
        'pf_classes': PF_CLASSES,
    }

    os.makedirs(os.path.dirname(out_pt_path), exist_ok=True)
    torch.save(out_obj, out_pt_path)


def build_dg2n_dataset(cfg_dir: str, out_dir: str) -> int:
    """Convert all CFG JSON files under cfg_dir into .pt files under out_dir.
    Returns number of graphs written.
    """
    count = 0
    for root, _, files in os.walk(cfg_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            cfg_path = os.path.join(root, file)
            base = os.path.splitext(file)[0]
            pt_path = os.path.join(out_dir, f"{base}.pt")
            cfg_json_to_dg2n_pt(cfg_path, pt_path)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description='Convert CFWR CFG JSONs to DG2N .pt graphs')
    ap.add_argument('--cfg_dir', required=True, help='Directory containing CFG JSONs (recursively)')
    ap.add_argument('--out_dir', required=True, help='Output directory to write .pt graphs')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = build_dg2n_dataset(args.cfg_dir, args.out_dir)
    print(f"DG2N adapter: wrote {n} graphs to {args.out_dir}")


if __name__ == '__main__':
    main()


