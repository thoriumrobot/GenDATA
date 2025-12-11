#!/usr/bin/env python3
"""
Comprehensive Annotation Impact Analysis

This script analyzes placed annotations to understand how and why they reduce
Lower Bound Checker warnings. It extracts annotations, parses warnings, maps
annotations to warnings, and generates a comprehensive analysis report.
"""

import os
import json
import re
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lower Bound Checker annotation types
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

@dataclass
class AnnotationPlacement:
    """Represents a single annotation placement"""
    file_path: str
    line_number: int
    annotation_type: str
    context: str
    target_element: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class WarningInfo:
    """Represents a single warning"""
    file_path: str
    line_number: int
    message: str
    column: Optional[int] = None
    checker_message: Optional[str] = None
    inferred_required_type: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AnnotationWarningMapping:
    """Maps an annotation to warnings it fixes"""
    annotation: AnnotationPlacement
    fixed_warnings: List[WarningInfo]
    mapping_type: str  # 'direct', 'upstream', 'dependency', 'defensive'
    confidence: float
    
    def to_dict(self):
        return {
            'annotation': self.annotation.to_dict(),
            'fixed_warnings': [w.to_dict() for w in self.fixed_warnings],
            'mapping_type': self.mapping_type,
            'confidence': self.confidence
        }

class AnnotationExtractor:
    """Extracts annotations from Java source files"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.annotations: List[AnnotationPlacement] = []
    
    def extract_annotations_from_file(self, java_file: Path) -> List[AnnotationPlacement]:
        """Extract all Lower Bound annotations from a Java file"""
        annotations = []
        
        if not java_file.exists():
            return annotations
        
        try:
            with open(java_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, start=1):
                # Check for annotations on the current line
                for ann_type in ANNOTATION_TYPES:
                    if ann_type in line:
                        # Extract context (surrounding lines)
                        context_start = max(0, line_num - 3)
                        context_end = min(len(lines), line_num + 2)
                        context = ''.join(lines[context_start:context_end])
                        
                        # Try to identify target element
                        target_element = self._extract_target_element(line, lines, line_num)
                        
                        annotation = AnnotationPlacement(
                            file_path=str(java_file),
                            line_number=line_num,
                            annotation_type=ann_type,
                            context=context.strip(),
                            target_element=target_element
                        )
                        annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            logger.warning(f"Error extracting annotations from {java_file}: {e}")
            return []
    
    def _extract_target_element(self, line: str, all_lines: List[str], line_num: int) -> Optional[str]:
        """Extract the target element (variable, parameter, etc.) from annotation context"""
        # Try to find variable/parameter name after annotation
        line_after = line.replace('@Positive', '').replace('@NonNegative', '').replace('@GTENegativeOne', '').strip()
        
        # Pattern for variable declarations: type name = ...
        var_pattern = r'\b(int|long|short|byte|float|double|char|boolean|String|\w+)\s+(\w+)\s*[=;]'
        match = re.search(var_pattern, line_after)
        if match:
            return match.group(2)
        
        # Pattern for method parameters: (type param, ...)
        param_pattern = r'(\w+)\s*[,)]'
        match = re.search(param_pattern, line_after)
        if match:
            return match.group(1)
        
        # Pattern for method returns: public type methodName(...)
        method_pattern = r'\b(public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
        match = re.search(method_pattern, line_after)
        if match:
            return match.group(2) if match.lastindex >= 2 else None
        
        return None
    
    def extract_all_annotations(self, project_dir: Path) -> List[AnnotationPlacement]:
        """Extract all annotations from all Java files in a project"""
        all_annotations = []
        
        # Find all Java files
        java_files = list(project_dir.rglob('*.java'))
        logger.info(f"Scanning {len(java_files)} Java files for annotations")
        
        for java_file in java_files:
            file_annotations = self.extract_annotations_from_file(java_file)
            all_annotations.extend(file_annotations)
        
        logger.info(f"Extracted {len(all_annotations)} annotations from {len(java_files)} files")
        return all_annotations

class WarningParser:
    """Parses Lower Bound Checker warnings"""
    
    def __init__(self):
        # Pattern for Checker Framework warnings
        # Format: file:line:column: error/warning: [checker.message] message
        self.warning_pattern = re.compile(
            r'^(.+?\.java):(\d+)(?::(\d+))?:\s*(?:compiler\.(?:err|warn)\.proc\.messager|error|warning):\s*(?:\[(.+?)\])?\s*(.+)$',
            re.MULTILINE
        )
        
        # Simpler pattern for variations
        self.simple_pattern = re.compile(
            r'^(.+?\.java):(\d+):\s*(error|warning):\s*(.+)$',
            re.MULTILINE
        )
    
    def parse_warnings_from_file(self, warnings_file: Path) -> List[WarningInfo]:
        """Parse warnings from a warnings file, filtering out compilation errors"""
        warnings = []
        
        if not warnings_file.exists():
            return warnings
        
        try:
            with open(warnings_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip lines that are clearly not checker warnings
            # Filter out compilation errors like "package does not exist", "cannot find symbol"
            skip_patterns = [
                r'package\s+\S+\s+does not exist',
                r'cannot find symbol',
                r'compiler.err',
                r'error:\s*file not found',
                r'error:\s*cannot access',
                r'Note:',
                r'^===',
                r'^Command:',
                r'^Exit code:'
            ]
            
            # Try detailed pattern first
            matches = self.warning_pattern.findall(content)
            for match in matches:
                file_path, line_num, col_num, checker_msg, message = match
                
                # Skip compilation errors
                message_lower = message.lower()
                if any(re.search(pattern, message_lower, re.IGNORECASE) for pattern in skip_patterns):
                    continue
                
                # Only count actual checker warnings (index checker warnings)
                if checker_msg and ('index' in checker_msg.lower() or 'lowerbound' in message_lower):
                    inferred_type = self._infer_required_annotation_type(message, checker_msg)
                    
                    warning = WarningInfo(
                        file_path=file_path.strip(),
                        line_number=int(line_num),
                        column=int(col_num) if col_num else None,
                        message=message.strip(),
                        checker_message=checker_msg.strip() if checker_msg else None,
                        inferred_required_type=inferred_type
                    )
                    warnings.append(warning)
            
            # If no matches, try simple pattern but be more selective
            if not warnings:
                simple_matches = self.simple_pattern.findall(content)
                for match in simple_matches:
                    file_path, line_num, level, message = match
                    
                    # Skip compilation errors
                    message_lower = message.lower()
                    if any(re.search(pattern, message_lower, re.IGNORECASE) for pattern in skip_patterns):
                        continue
                    
                    # Look for index checker specific warnings
                    if 'index' in message_lower or 'array' in message_lower or 'lowerbound' in message_lower:
                        inferred_type = self._infer_required_annotation_type(message, None)
                        
                        warning = WarningInfo(
                            file_path=file_path.strip(),
                            line_number=int(line_num),
                            message=message.strip(),
                            inferred_required_type=inferred_type
                        )
                        warnings.append(warning)
            
            logger.info(f"Parsed {len(warnings)} checker warnings from {warnings_file}")
            return warnings
            
        except Exception as e:
            logger.warning(f"Error parsing warnings from {warnings_file}: {e}")
            return []
    
    def _infer_required_annotation_type(self, message: str, checker_msg: Optional[str]) -> Optional[str]:
        """Infer the required annotation type from warning message"""
        message_lower = message.lower()
        
        # Check for explicit mentions
        if '@positive' in message_lower or 'positive' in message_lower:
            return '@Positive'
        elif '@nonnegative' in message_lower or 'nonnegative' in message_lower or 'non-negative' in message_lower:
            return '@NonNegative'
        elif '@gtenegativeone' in message_lower or 'gtenegativeone' in message_lower or 'gte negative one' in message_lower:
            return '@GTENegativeOne'
        
        # Check for semantic indicators
        if any(word in message_lower for word in ['greater than zero', 'must be positive', 'positive value']):
            return '@Positive'
        elif any(word in message_lower for word in ['non-negative', '>= 0', 'greater than or equal to zero']):
            return '@NonNegative'
        elif any(word in message_lower for word in ['>= -1', 'greater than or equal to -1']):
            return '@GTENegativeOne'
        
        # Default based on common patterns
        if 'index' in message_lower or 'array' in message_lower:
            return '@NonNegative'  # Array indices are typically non-negative
        
        return None

class AnnotationWarningMapper:
    """Maps annotations to warnings they fix"""
    
    def __init__(self, line_tolerance: int = 3):
        self.line_tolerance = line_tolerance
    
    def map_annotations_to_warnings(self, 
                                   annotations: List[AnnotationPlacement],
                                   warnings: List[WarningInfo]) -> List[AnnotationWarningMapping]:
        """Map each annotation to warnings it fixes"""
        mappings = []
        
        # Group warnings by file
        warnings_by_file: Dict[str, List[WarningInfo]] = defaultdict(list)
        for warning in warnings:
            warnings_by_file[warning.file_path].append(warning)
        
        # For each annotation, find matching warnings
        for annotation in annotations:
            file_warnings = warnings_by_file.get(annotation.file_path, [])
            
            fixed_warnings = []
            mapping_type = 'defensive'  # Default assumption
            
            for warning in file_warnings:
                # Check if annotation is near warning location
                line_diff = abs(annotation.line_number - warning.line_number)
                
                if line_diff <= self.line_tolerance:
                    # Check if annotation type matches warning requirement
                    if warning.inferred_required_type == annotation.annotation_type:
                        if line_diff == 0:
                            mapping_type = 'direct'
                        elif line_diff <= 1:
                            mapping_type = 'direct'  # Within 1 line is effectively direct
                        else:
                            mapping_type = 'dependency'
                        
                        fixed_warnings.append(warning)
                    elif warning.inferred_required_type is None:
                        # Warning type unknown, assume it could be fixed
                        if line_diff <= 1:
                            mapping_type = 'direct'
                        fixed_warnings.append(warning)
            
            if fixed_warnings:
                # Calculate confidence based on type match and proximity
                confidence = self._calculate_confidence(annotation, fixed_warnings)
                
                mapping = AnnotationWarningMapping(
                    annotation=annotation,
                    fixed_warnings=fixed_warnings,
                    mapping_type=mapping_type,
                    confidence=confidence
                )
                mappings.append(mapping)
            else:
                # No direct warnings, might be defensive or upstream fix
                mapping = AnnotationWarningMapping(
                    annotation=annotation,
                    fixed_warnings=[],
                    mapping_type='defensive',
                    confidence=0.3  # Low confidence for defensive annotations
                )
                mappings.append(mapping)
        
        return mappings
    
    def _calculate_confidence(self, annotation: AnnotationPlacement, warnings: List[WarningInfo]) -> float:
        """Calculate confidence that annotation fixes warnings"""
        if not warnings:
            return 0.3
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for type matches
        type_matches = sum(1 for w in warnings if w.inferred_required_type == annotation.annotation_type)
        if type_matches > 0:
            confidence += 0.3 * (type_matches / len(warnings))
        
        # Increase confidence for proximity
        avg_distance = sum(abs(annotation.line_number - w.line_number) for w in warnings) / len(warnings)
        if avg_distance == 0:
            confidence += 0.2
        elif avg_distance <= 1:
            confidence += 0.1
        
        return min(1.0, confidence)

class AnnotationImpactAnalyzer:
    """Main analyzer class"""
    
    def __init__(self, base_dir: Path = Path('/home/ubuntu/GenDATA')):
        self.base_dir = Path(base_dir)
        self.annotation_eval_dir = self.base_dir / 'annotation_evaluation'
        self.extractor = AnnotationExtractor(self.base_dir)
        self.parser = WarningParser()
        self.mapper = AnnotationWarningMapper()
    
    def analyze_project(self, project_name: str, base_model: Optional[str] = None) -> Dict[str, Any]:
        """Analyze annotations for a single project"""
        logger.info(f"Analyzing project: {project_name}")
        
        project_dir = self.annotation_eval_dir / 'temp_repos' / project_name
        if not project_dir.exists():
            logger.error(f"Project directory not found: {project_dir}")
            return {}
        
        # Extract annotations
        annotations = self.extractor.extract_all_annotations(project_dir)
        
        # Get baseline warnings from backup files
        baseline_warnings = self._get_baseline_warnings(project_name)
        
        # Get warnings after annotation (should be 0 based on evaluation_report.json)
        annotated_warnings = self._run_checker_on_directory(project_dir, project_name, 'annotated')
        
        # Map annotations to warnings
        mappings = self.mapper.map_annotations_to_warnings(annotations, baseline_warnings)
        
        # Analyze reduction mechanisms
        reduction_analysis = self._analyze_reduction_mechanisms(mappings, baseline_warnings, annotated_warnings)
        
        return {
            'project_name': project_name,
            'annotations': [a.to_dict() for a in annotations],
            'baseline_warnings': [w.to_dict() for w in baseline_warnings],
            'annotated_warnings': [w.to_dict() for w in annotated_warnings],
            'warnings_eliminated': len(baseline_warnings) - len(annotated_warnings),
            'mappings': [m.to_dict() for m in mappings],
            'reduction_analysis': reduction_analysis
        }
    
    def _get_baseline_warnings(self, project_name: str) -> List[WarningInfo]:
        """Get baseline warnings for a project by running checker on backup files"""
        warnings = []
        
        # Try to find backup directory
        backup_dir = self.annotation_eval_dir / 'backups' / project_name
        if backup_dir.exists():
            logger.info(f"Running checker on backup files for {project_name}")
            warnings = self._run_checker_on_directory(backup_dir, project_name, 'baseline')
        
        # If no warnings from checker, try to find warnings file
        if not warnings:
            warnings_file = self.base_dir / 'case_studies' / project_name / f'{project_name}_warnings.out'
            if not warnings_file.exists():
                warnings_file = self.annotation_eval_dir / f'{project_name}_warnings.out'
            
            if warnings_file.exists():
                warnings = self.parser.parse_warnings_from_file(warnings_file)
            else:
                logger.warning(f"Warning file not found for {project_name}")
                # Still return empty list - we'll work with what we have
        
        return warnings
    
    def _run_checker_on_directory(self, project_dir: Path, project_name: str, suffix: str) -> List[WarningInfo]:
        """Run Lower Bound Checker on a directory and parse warnings"""
        try:
            from checker_framework_runner import CheckerFrameworkRunner
            
            runner = CheckerFrameworkRunner(checker_name='lower_bound')
            
            # Create temp warnings file
            temp_warnings = tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False)
            temp_warnings.close()
            temp_warnings_path = Path(temp_warnings.name)
            
            # Run checker
            success = runner.run_checker_on_project(
                str(project_dir),
                str(temp_warnings_path),
                max_files=50  # Limit for speed
            )
            
            warnings = []
            if success and temp_warnings_path.exists():
                warnings = self.parser.parse_warnings_from_file(temp_warnings_path)
                # Clean up temp file
                temp_warnings_path.unlink()
            
            return warnings
            
        except Exception as e:
            logger.warning(f"Error running checker on {project_dir}: {e}")
            return []
    
    def _get_warnings_from_evaluation_report(self, project_name: str) -> List[WarningInfo]:
        """Extract warning count from evaluation_report.json"""
        report_file = self.annotation_eval_dir / 'evaluation_report.json'
        if not report_file.exists():
            return []
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            for result in report.get('results', []):
                if result.get('project_name') == project_name:
                    baseline_count = result.get('baseline_warnings', 0)
                    logger.info(f"Found {baseline_count} baseline warnings for {project_name} in evaluation report")
                    # Return empty list as we don't have detailed warning info
                    return []
        except Exception as e:
            logger.warning(f"Error reading evaluation report: {e}")
        
        return []
    
    def _analyze_reduction_mechanisms(self, 
                                     mappings: List[AnnotationWarningMapping],
                                     baseline_warnings: List[WarningInfo],
                                     annotated_warnings: List[WarningInfo]) -> Dict[str, Any]:
        """Analyze how annotations reduce warnings"""
        reduction_types = defaultdict(int)
        type_distribution = defaultdict(int)
        
        for mapping in mappings:
            reduction_types[mapping.mapping_type] += len(mapping.fixed_warnings)
            type_distribution[mapping.annotation.annotation_type] += len(mapping.fixed_warnings)
        
        total_fixed = sum(len(m.fixed_warnings) for m in mappings)
        direct_fixes = sum(len(m.fixed_warnings) for m in mappings if m.mapping_type == 'direct')
        defensive_annotations = sum(1 for m in mappings if m.mapping_type == 'defensive' and not m.fixed_warnings)
        
        # Calculate actual reduction
        actual_reduction = len(baseline_warnings) - len(annotated_warnings)
        reduction_percentage = (actual_reduction / len(baseline_warnings) * 100) if baseline_warnings else 0.0
        
        return {
            'baseline_warning_count': len(baseline_warnings),
            'annotated_warning_count': len(annotated_warnings),
            'actual_reduction': actual_reduction,
            'reduction_percentage': reduction_percentage,
            'total_annotations': len(mappings),
            'warnings_fixed_mapped': total_fixed,
            'direct_fixes': direct_fixes,
            'defensive_annotations': defensive_annotations,
            'reduction_by_type': dict(reduction_types),
            'fixes_by_annotation_type': dict(type_distribution),
            'mapping_coverage': (total_fixed / len(baseline_warnings)) if baseline_warnings else 0.0
        }
    
    def analyze_all_projects(self) -> Dict[str, Any]:
        """Analyze all three case study projects"""
        projects = ['sortpom', 'eclipse-external-annotations-m2e-plugin', 'pom-tuner']
        results = {}
        
        for project in projects:
            try:
                results[project] = self.analyze_project(project)
            except Exception as e:
                logger.error(f"Error analyzing {project}: {e}")
                results[project] = {'error': str(e)}
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_file: Path) -> None:
        """Generate comprehensive analysis report"""
        logger.info(f"Generating report: {output_file}")
        
        report_lines = []
        report_lines.append("# Annotation Impact Analysis Report")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        # Summary statistics
        total_annotations = 0
        total_warnings = 0
        total_fixed = 0
        
        for project_name, project_data in results.items():
            if 'error' in project_data:
                continue
            total_annotations += len(project_data.get('annotations', []))
            total_warnings += len(project_data.get('warnings', []))
            reduction = project_data.get('reduction_analysis', {})
            total_fixed += reduction.get('warnings_fixed', 0)
        
        report_lines.append(f"- **Total Projects Analyzed**: {len(results)}")
        report_lines.append(f"- **Total Annotations Placed**: {total_annotations}")
        report_lines.append(f"- **Total Baseline Warnings**: {total_warnings}")
        report_lines.append(f"- **Warnings Fixed (Mapped)**: {total_fixed}")
        report_lines.append("")
        
        # Per-project analysis
        for project_name, project_data in results.items():
            if 'error' in project_data:
                report_lines.append(f"## {project_name}")
                report_lines.append(f"Error: {project_data['error']}")
                report_lines.append("")
                continue
            
            report_lines.append(f"## {project_name}")
            report_lines.append("")
            
            annotations = project_data.get('annotations', [])
            warnings = project_data.get('warnings', [])
            mappings = project_data.get('mappings', [])
            reduction = project_data.get('reduction_analysis', {})
            
            report_lines.append(f"### Summary")
            report_lines.append(f"- **Annotations Placed**: {len(annotations)}")
            report_lines.append(f"- **Baseline Warnings**: {len(warnings)}")
            report_lines.append(f"- **Warnings Fixed**: {reduction.get('warnings_fixed', 0)}")
            report_lines.append(f"- **Direct Fixes**: {reduction.get('direct_fixes', 0)}")
            report_lines.append(f"- **Defensive Annotations**: {reduction.get('defensive_annotations', 0)}")
            report_lines.append("")
            
            # Annotation type distribution
            type_counts = defaultdict(int)
            for ann in annotations:
                type_counts[ann['annotation_type']] += 1
            
            report_lines.append(f"### Annotation Type Distribution")
            for ann_type, count in sorted(type_counts.items()):
                report_lines.append(f"- **{ann_type}**: {count}")
            report_lines.append("")
            
            # Reduction mechanism breakdown
            report_lines.append(f"### Reduction Mechanism Breakdown")
            reduction_by_type = reduction.get('reduction_by_type', {})
            for red_type, count in sorted(reduction_by_type.items()):
                report_lines.append(f"- **{red_type}**: {count} warnings fixed")
            report_lines.append("")
            
            # Sample mappings
            report_lines.append(f"### Sample Annotation-to-Warning Mappings")
            sample_mappings = [m for m in mappings if m['fixed_warnings']][:5]
            for i, mapping in enumerate(sample_mappings, 1):
                ann = mapping['annotation']
                fixed = mapping['fixed_warnings']
                report_lines.append(f"#### Mapping {i}")
                report_lines.append(f"- **Annotation**: {ann['annotation_type']} at line {ann['line_number']} in {Path(ann['file_path']).name}")
                report_lines.append(f"- **Type**: {mapping['mapping_type']}")
                report_lines.append(f"- **Fixed Warnings**: {len(fixed)}")
                if fixed:
                    for warning in fixed[:2]:  # Show first 2
                        report_lines.append(f"  - Line {warning['line_number']}: {warning['message'][:100]}")
                report_lines.append("")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report written to: {output_file}")

def main():
    """Main function"""
    analyzer = AnnotationImpactAnalyzer()
    
    # Analyze all projects
    results = analyzer.analyze_all_projects()
    
    # Generate report
    report_file = Path('/home/ubuntu/GenDATA/ANNOTATION_IMPACT_ANALYSIS_REPORT.md')
    analyzer.generate_report(results, report_file)
    
    # Save detailed results to JSON
    results_file = Path('/home/ubuntu/GenDATA/annotation_impact_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Analysis complete!")
    logger.info(f"Report: {report_file}")
    logger.info(f"Detailed results: {results_file}")

if __name__ == '__main__':
    main()
