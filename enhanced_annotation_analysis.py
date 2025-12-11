#!/usr/bin/env python3
"""
Enhanced Annotation Impact Analysis

This script performs a deeper analysis of how annotations reduce warnings by:
1. Comparing original vs annotated files
2. Analyzing annotation placement patterns
3. Understanding constraint propagation
4. Documenting reduction mechanisms
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import difflib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

@dataclass
class AnnotationChange:
    """Represents an annotation that was added"""
    file_path: str
    line_number: int
    annotation_type: str
    original_code: str
    annotated_code: str
    context_before: str
    context_after: str
    placement_strategy: str  # 'before_statement', 'parameter', 'return_type', 'field'
    
    def to_dict(self):
        return asdict(self)

class AnnotationChangeAnalyzer:
    """Analyzes differences between original and annotated files"""
    
    def __init__(self):
        self.changes: List[AnnotationChange] = []
    
    def analyze_file_difference(self, original_file: Path, annotated_file: Path) -> List[AnnotationChange]:
        """Analyze differences between original and annotated file"""
        if not original_file.exists() or not annotated_file.exists():
            return []
        
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            with open(annotated_file, 'r', encoding='utf-8') as f:
                annotated_lines = f.readlines()
            
            changes = []
            
            # Use difflib to find differences
            diff = list(difflib.unified_diff(original_lines, annotated_lines, 
                                            fromfile=str(original_file),
                                            tofile=str(annotated_file),
                                            lineterm=''))
            
            # Parse diff to find annotation additions
            i = 0
            while i < len(diff):
                line = diff[i]
                
                # Look for lines that start with + (additions) containing annotations
                if line.startswith('+') and not line.startswith('+++'):
                    added_line = line[1:]  # Remove +
                    
                    for ann_type in ANNOTATION_TYPES:
                        if ann_type in added_line:
                            # Find context
                            context_start = max(0, i - 10)
                            context_end = min(len(diff), i + 10)
                            context_lines = diff[context_start:context_end]
                            
                            # Extract line number from diff
                            line_num = self._extract_line_number_from_diff(diff, i)
                            
                            # Determine placement strategy
                            placement = self._determine_placement_strategy(added_line, original_lines, annotated_lines, line_num)
                            
                            # Get surrounding context
                            context_before = ''.join(original_lines[max(0, line_num-3):line_num]) if line_num else ''
                            context_after = ''.join(annotated_lines[max(0, line_num-2):min(len(annotated_lines), line_num+3)]) if line_num else ''
                            
                            change = AnnotationChange(
                                file_path=str(annotated_file),
                                line_number=line_num or 0,
                                annotation_type=ann_type,
                                original_code=original_lines[line_num-1] if line_num and line_num <= len(original_lines) else '',
                                annotated_code=added_line,
                                context_before=context_before,
                                context_after=context_after,
                                placement_strategy=placement
                            )
                            changes.append(change)
                            break
                
                i += 1
            
            return changes
            
        except Exception as e:
            logger.warning(f"Error analyzing file difference: {e}")
            return []
    
    def _extract_line_number_from_diff(self, diff: List[str], index: int) -> Optional[int]:
        """Extract line number from unified diff"""
        # Look backwards for @@ line number marker
        for i in range(index, max(0, index-20), -1):
            if i < len(diff) and diff[i].startswith('@@'):
                # Format: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r'\+(\d+)', diff[i])
                if match:
                    base_line = int(match.group(1))
                    # Count lines added before current position
                    added_count = 0
                    for j in range(i+1, index):
                        if j < len(diff) and diff[j].startswith('+') and not diff[j].startswith('+++'):
                            added_count += 1
                    return base_line + added_count - 1
        return None
    
    def _determine_placement_strategy(self, annotated_line: str, original_lines: List[str], 
                                     annotated_lines: List[str], line_num: Optional[int]) -> str:
        """Determine placement strategy based on code context"""
        if not line_num or line_num <= 0:
            return 'unknown'
        
        # Get the line after annotation in annotated file
        if line_num < len(annotated_lines):
            next_line = annotated_lines[line_num].strip()
            
            # Check for method parameters
            if '(' in next_line and ')' in next_line and not next_line.startswith('//'):
                return 'parameter'
            
            # Check for return type
            if re.search(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', next_line):
                return 'return_type'
            
            # Check for field declaration
            if re.search(r'^\s*(private|public|protected)\s+\w+\s+\w+\s*[=;]', next_line):
                return 'field'
            
            # Check for variable declaration
            if re.search(r'\b(int|long|String|\w+)\s+\w+\s*[=;]', next_line):
                return 'variable'
            
            # Default: before statement
            return 'before_statement'
        
        return 'unknown'

class AnnotationImpactAnalyzer:
    """Main enhanced analyzer"""
    
    def __init__(self, base_dir: Path = Path('/home/ubuntu/GenDATA')):
        self.base_dir = Path(base_dir)
        self.annotation_eval_dir = self.base_dir / 'annotation_evaluation'
        self.change_analyzer = AnnotationChangeAnalyzer()
    
    def analyze_project_changes(self, project_name: str) -> Dict[str, Any]:
        """Analyze annotation changes for a project"""
        logger.info(f"Analyzing annotation changes for: {project_name}")
        
        backup_dir = self.annotation_eval_dir / 'backups' / project_name
        annotated_dir = self.annotation_eval_dir / 'temp_repos' / project_name
        
        if not backup_dir.exists() or not annotated_dir.exists():
            logger.error(f"Backup or annotated directory not found for {project_name}")
            return {}
        
        # Find all Java files
        backup_files = list(backup_dir.rglob('*.java'))
        annotated_files = list(annotated_dir.rglob('*.java'))
        
        # Match files and analyze differences
        changes_by_file = {}
        total_changes = 0
        
        # Create mapping from relative path to files
        backup_map = {f.relative_to(backup_dir): f for f in backup_files}
        annotated_map = {f.relative_to(annotated_dir): f for f in annotated_files}
        
        for rel_path in backup_map.keys():
            if rel_path in annotated_map:
                backup_file = backup_map[rel_path]
                annotated_file = annotated_map[rel_path]
                
                file_changes = self.change_analyzer.analyze_file_difference(backup_file, annotated_file)
                if file_changes:
                    changes_by_file[str(rel_path)] = [c.to_dict() for c in file_changes]
                    total_changes += len(file_changes)
        
        # Analyze patterns
        placement_patterns = defaultdict(int)
        annotation_type_counts = defaultdict(int)
        
        for file_changes in changes_by_file.values():
            for change in file_changes:
                placement_patterns[change['placement_strategy']] += 1
                annotation_type_counts[change['annotation_type']] += 1
        
        return {
            'project_name': project_name,
            'files_changed': len(changes_by_file),
            'total_annotations_added': total_changes,
            'changes_by_file': changes_by_file,
            'placement_patterns': dict(placement_patterns),
            'annotation_type_distribution': dict(annotation_type_counts)
        }
    
    def analyze_all_projects(self) -> Dict[str, Any]:
        """Analyze all three projects"""
        projects = ['sortpom', 'eclipse-external-annotations-m2e-plugin', 'pom-tuner']
        results = {}
        
        for project in projects:
            try:
                results[project] = self.analyze_project_changes(project)
            except Exception as e:
                logger.error(f"Error analyzing {project}: {e}")
                results[project] = {'error': str(e)}
        
        return results
    
    def generate_enhanced_report(self, analysis_results: Dict[str, Any], 
                                 change_results: Dict[str, Any],
                                 output_file: Path) -> None:
        """Generate enhanced analysis report"""
        logger.info(f"Generating enhanced report: {output_file}")
        
        report_lines = []
        report_lines.append("# Annotation Impact Analysis Report")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        # Summary statistics
        total_annotations = 0
        total_files_changed = 0
        
        for project_name in analysis_results.keys():
            if 'error' in analysis_results[project_name]:
                continue
            total_annotations += len(analysis_results[project_name].get('annotations', []))
            if project_name in change_results and 'files_changed' in change_results[project_name]:
                total_files_changed += change_results[project_name]['files_changed']
        
        report_lines.append(f"- **Total Projects Analyzed**: {len(analysis_results)}")
        report_lines.append(f"- **Total Annotations Placed**: {total_annotations}")
        report_lines.append(f"- **Total Files Modified**: {total_files_changed}")
        report_lines.append("")
        
        # Per-project detailed analysis
        for project_name in analysis_results.keys():
            project_data = analysis_results[project_name]
            change_data = change_results.get(project_name, {})
            
            if 'error' in project_data:
                report_lines.append(f"## {project_name}")
                report_lines.append(f"Error: {project_data['error']}")
                report_lines.append("")
                continue
            
            report_lines.append(f"## {project_name}")
            report_lines.append("")
            
            annotations = project_data.get('annotations', [])
            reduction = project_data.get('reduction_analysis', {})
            
            report_lines.append(f"### Summary Statistics")
            report_lines.append(f"- **Annotations Placed**: {len(annotations)}")
            report_lines.append(f"- **Baseline Warnings**: {reduction.get('baseline_warning_count', 'N/A')}")
            report_lines.append(f"- **Warnings After Annotation**: {reduction.get('annotated_warning_count', 'N/A')}")
            report_lines.append(f"- **Warning Reduction**: {reduction.get('actual_reduction', 'N/A')}")
            report_lines.append(f"- **Reduction Percentage**: {reduction.get('reduction_percentage', 0):.1f}%")
            report_lines.append("")
            
            # Placement patterns from change analysis
            if change_data and 'placement_patterns' in change_data:
                report_lines.append(f"### Annotation Placement Patterns")
                for pattern, count in sorted(change_data['placement_patterns'].items()):
                    report_lines.append(f"- **{pattern.replace('_', ' ').title()}**: {count}")
                report_lines.append("")
            
            # Annotation type distribution
            type_counts = defaultdict(int)
            for ann in annotations:
                type_counts[ann['annotation_type']] += 1
            
            report_lines.append(f"### Annotation Type Distribution")
            for ann_type, count in sorted(type_counts.items()):
                report_lines.append(f"- **{ann_type}**: {count}")
            report_lines.append("")
            
            # Sample annotations
            report_lines.append(f"### Sample Annotations")
            sample_annotations = annotations[:10]
            for i, ann in enumerate(sample_annotations, 1):
                file_name = Path(ann['file_path']).name
                report_lines.append(f"#### Annotation {i}")
                report_lines.append(f"- **Type**: {ann['annotation_type']}")
                report_lines.append(f"- **Location**: {file_name}:{ann['line_number']}")
                if ann.get('target_element'):
                    report_lines.append(f"- **Target**: {ann['target_element']}")
                report_lines.append(f"- **Context**: ```java")
                context_lines = ann.get('context', '').split('\n')[:3]
                for ctx_line in context_lines:
                    report_lines.append(ctx_line)
                report_lines.append("```")
                report_lines.append("")
        
        # How annotations reduce warnings section
        report_lines.append("## How Annotations Reduce Warnings")
        report_lines.append("")
        report_lines.append("### Mechanism Overview")
        report_lines.append("")
        report_lines.append("Lower Bound Checker annotations work by:")
        report_lines.append("")
        report_lines.append("1. **Constraint Propagation**: Annotations on method parameters and return types")
        report_lines.append("   propagate constraints through the dataflow graph.")
        report_lines.append("")
        report_lines.append("2. **Value Assertions**: `@Positive`, `@NonNegative`, and `@GTENegativeOne` annotations")
        report_lines.append("   assert constraints on values that are then verified by the checker.")
        report_lines.append("")
        report_lines.append("3. **Upstream Constraint Satisfaction**: Annotations placed on method parameters")
        report_lines.append("   satisfy constraints required by method bodies, eliminating warnings at call sites.")
        report_lines.append("")
        report_lines.append("4. **Return Type Constraints**: Annotations on return types ensure callers receive")
        report_lines.append("   values that satisfy constraints, preventing downstream warnings.")
        report_lines.append("")
        report_lines.append("### Example: @NonNegative Annotation")
        report_lines.append("")
        report_lines.append("When `@NonNegative` is placed on a parameter or variable:")
        report_lines.append("- The checker assumes the value is >= 0")
        report_lines.append("- Array index operations using this value are verified as safe")
        report_lines.append("- Comparisons with zero are type-checked correctly")
        report_lines.append("- Method calls passing this value satisfy non-negative requirements")
        report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Enhanced report written to: {output_file}")

def main():
    """Main function"""
    # Run both analyses
    from analyze_annotation_impact import AnnotationImpactAnalyzer as BaseAnalyzer
    
    base_analyzer = BaseAnalyzer()
    enhanced_analyzer = AnnotationImpactAnalyzer()
    
    # Run base analysis
    logger.info("Running base annotation analysis...")
    base_results = base_analyzer.analyze_all_projects()
    
    # Run change analysis
    logger.info("Running annotation change analysis...")
    change_results = enhanced_analyzer.analyze_all_projects()
    
    # Generate enhanced report
    report_file = Path('/home/ubuntu/GenDATA/ANNOTATION_IMPACT_ANALYSIS_REPORT.md')
    enhanced_analyzer.generate_enhanced_report(base_results, change_results, report_file)
    
    logger.info("Enhanced analysis complete!")
    logger.info(f"Report: {report_file}")

if __name__ == '__main__':
    main()
