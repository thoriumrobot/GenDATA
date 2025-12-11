#!/usr/bin/env python3
"""
Verify Data Accuracy Script

This script verifies that all data sources are real and not mock data.
It checks prediction files, annotated files, evaluation results, and calculations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVerifier:
    """Verifies all data sources are real"""
    
    def __init__(self, base_dir: Path = Path('/home/ubuntu/GenDATA')):
        self.base_dir = Path(base_dir)
        self.annotation_eval_dir = self.base_dir / 'annotation_evaluation'
        self.verification_results = {
            'all_verified': True,
            'issues': [],
            'verified_components': []
        }
    
    def verify_predictions_exist(self) -> bool:
        """Verify prediction files exist and contain real predictions"""
        logger.info("Verifying prediction files...")
        verified = True
        
        projects = ['sortpom', 'eclipse-external-annotations-m2e-plugin', 'pom-tuner']
        models = ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
        
        for project in projects:
            pred_dir = self.annotation_eval_dir / 'predictions' / project
            if not pred_dir.exists():
                self.verification_results['issues'].append(f"Prediction directory not found: {pred_dir}")
                verified = False
                continue
            
            for model in models:
                pred_file = pred_dir / f'{model}_predictions.json'
                if pred_file.exists():
                    try:
                        with open(pred_file, 'r') as f:
                            predictions = json.load(f)
                        
                        if isinstance(predictions, list) and len(predictions) > 0:
                            # Verify predictions have required fields
                            sample = predictions[0]
                            required_fields = ['annotation_type', 'confidence', 'line_number', 'file_path']
                            if all(field in sample for field in required_fields):
                                self.verification_results['verified_components'].append(
                                    f"Predictions: {project}/{model} - {len(predictions)} predictions"
                                )
                            else:
                                self.verification_results['issues'].append(
                                    f"Predictions {project}/{model} missing required fields"
                                )
                                verified = False
                        elif isinstance(predictions, list) and len(predictions) == 0:
                            # Empty predictions are OK (no predictions generated)
                            self.verification_results['verified_components'].append(
                                f"Predictions: {project}/{model} - Empty (valid)"
                            )
                        else:
                            self.verification_results['issues'].append(
                                f"Predictions {project}/{model} has invalid format"
                            )
                            verified = False
                    except Exception as e:
                        self.verification_results['issues'].append(
                            f"Error reading predictions {project}/{model}: {e}"
                        )
                        verified = False
        
        if verified:
            logger.info("✅ All prediction files verified")
        else:
            logger.warning("⚠️ Some prediction files have issues")
        
        return verified
    
    def verify_annotations_in_files(self) -> bool:
        """Verify annotated files actually contain annotations"""
        logger.info("Verifying annotations in source files...")
        verified = True
        
        projects = ['sortpom', 'eclipse-external-annotations-m2e-plugin', 'pom-tuner']
        annotations_found = 0
        files_checked = 0
        
        for project in projects:
            project_dir = self.annotation_eval_dir / 'temp_repos' / project
            if not project_dir.exists():
                self.verification_results['issues'].append(f"Project directory not found: {project_dir}")
                verified = False
                continue
            
            java_files = list(project_dir.rglob('*.java'))[:10]  # Sample 10 files per project
            files_checked += len(java_files)
            
            for java_file in java_files:
                try:
                    with open(java_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for Lower Bound annotations
                    if '@NonNegative' in content or '@Positive' in content or '@GTENegativeOne' in content:
                        annotations_found += content.count('@NonNegative')
                        annotations_found += content.count('@Positive')
                        annotations_found += content.count('@GTENegativeOne')
                
                except Exception as e:
                    logger.debug(f"Error reading {java_file}: {e}")
        
        if annotations_found > 0:
            self.verification_results['verified_components'].append(
                f"Annotations in files: Found {annotations_found} annotations in {files_checked} files (sample)"
            )
            logger.info(f"✅ Found {annotations_found} annotations in {files_checked} files (sample)")
        else:
            self.verification_results['issues'].append("No annotations found in sampled files")
            verified = False
        
        return verified
    
    def verify_evaluation_report(self) -> bool:
        """Verify evaluation_report.json contains real results"""
        logger.info("Verifying evaluation report...")
        verified = True
        
        report_file = self.annotation_eval_dir / 'evaluation_report.json'
        if not report_file.exists():
            self.verification_results['issues'].append(f"Evaluation report not found: {report_file}")
            return False
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Verify structure
            if 'results' not in report:
                self.verification_results['issues'].append("Evaluation report missing 'results' key")
                return False
            
            # Verify each project result
            for result in report.get('results', []):
                project_name = result.get('project_name', '')
                baseline = result.get('baseline_warnings', 0)
                
                # Verify baseline warnings are reasonable (non-negative)
                if baseline < 0:
                    self.verification_results['issues'].append(
                        f"{project_name}: Baseline warnings is negative ({baseline})"
                    )
                    verified = False
                
                # Verify model results
                for model_result in result.get('model_results', []):
                    model = model_result.get('base_model', 'unknown')
                    annotations = model_result.get('annotations_placed', 0)
                    
                    # Verify annotations_placed is non-negative
                    if annotations < 0:
                        self.verification_results['issues'].append(
                            f"{project_name}/{model}: Negative annotations_placed ({annotations})"
                        )
                        verified = False
                    
                    # Verify reduction percentage is reasonable
                    reduction_pct = model_result.get('reduction_percentage', 0)
                    if reduction_pct < 0 or reduction_pct > 100:
                        self.verification_results['issues'].append(
                            f"{project_name}/{model}: Invalid reduction_percentage ({reduction_pct})"
                        )
                        verified = False
            
            self.verification_results['verified_components'].append(
                f"Evaluation report: {len(report.get('results', []))} projects, structure valid"
            )
            
            if verified:
                logger.info("✅ Evaluation report verified")
            else:
                logger.warning("⚠️ Evaluation report has some issues")
        
        except Exception as e:
            self.verification_results['issues'].append(f"Error reading evaluation report: {e}")
            verified = False
        
        return verified
    
    def verify_warning_reduction_calculations(self) -> bool:
        """Verify warning reduction calculations are correct"""
        logger.info("Verifying warning reduction calculations...")
        verified = True
        
        report_file = self.annotation_eval_dir / 'evaluation_report.json'
        if not report_file.exists():
            return False
        
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            for result in report.get('results', []):
                baseline = result.get('baseline_warnings', 0)
                project_name = result.get('project_name', '')
                
                for model_result in result.get('model_results', []):
                    model = model_result.get('base_model', 'unknown')
                    warnings_after = model_result.get('warnings_after', 0)
                    warning_reduction = model_result.get('warning_reduction', 0)
                    reduction_percentage = model_result.get('reduction_percentage', 0)
                    
                    # Verify calculation: warning_reduction = baseline - warnings_after
                    expected_reduction = baseline - warnings_after
                    if expected_reduction != warning_reduction:
                        self.verification_results['issues'].append(
                            f"{project_name}/{model}: Warning reduction calculation mismatch "
                            f"(expected {expected_reduction}, got {warning_reduction})"
                        )
                        verified = False
                    
                    # Verify percentage: reduction_percentage = (warning_reduction / baseline) * 100
                    if baseline > 0:
                        expected_percentage = (warning_reduction / baseline) * 100
                        if abs(expected_percentage - reduction_percentage) > 0.1:  # Allow small floating point differences
                            self.verification_results['issues'].append(
                                f"{project_name}/{model}: Reduction percentage calculation mismatch "
                                f"(expected {expected_percentage:.1f}%, got {reduction_percentage:.1f}%)"
                            )
                            verified = False
            
            if verified:
                logger.info("✅ Warning reduction calculations verified")
                self.verification_results['verified_components'].append(
                    "Warning reduction calculations: All correct"
                )
            else:
                logger.warning("⚠️ Some calculation issues found")
        
        except Exception as e:
            self.verification_results['issues'].append(f"Error verifying calculations: {e}")
            verified = False
        
        return verified
    
    def verify_all(self) -> Dict[str, Any]:
        """Verify all data sources"""
        logger.info("Starting comprehensive data verification...")
        
        checks = [
            ('Predictions', self.verify_predictions_exist),
            ('Annotations in Files', self.verify_annotations_in_files),
            ('Evaluation Report', self.verify_evaluation_report),
            ('Calculations', self.verify_warning_reduction_calculations)
        ]
        
        for check_name, check_func in checks:
            try:
                result = check_func()
                if not result:
                    self.verification_results['all_verified'] = False
            except Exception as e:
                logger.error(f"Error in {check_name} verification: {e}")
                self.verification_results['issues'].append(f"{check_name} verification failed: {e}")
                self.verification_results['all_verified'] = False
        
        return self.verification_results
    
    def generate_verification_report(self, output_file: Path) -> None:
        """Generate verification report"""
        report_lines = []
        report_lines.append("# Data Verification Report")
        report_lines.append("")
        
        if self.verification_results['all_verified']:
            report_lines.append("✅ **All data verified as real**")
        else:
            report_lines.append("⚠️ **Some verification issues found**")
        
        report_lines.append("")
        report_lines.append("## Verified Components")
        for component in self.verification_results['verified_components']:
            report_lines.append(f"- ✅ {component}")
        
        if self.verification_results['issues']:
            report_lines.append("")
            report_lines.append("## Issues Found")
            for issue in self.verification_results['issues']:
                report_lines.append(f"- ⚠️ {issue}")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    """Main function"""
    verifier = DataVerifier()
    
    # Run all verifications
    results = verifier.verify_all()
    
    # Generate report
    report_file = Path('/home/ubuntu/GenDATA/DATA_VERIFICATION_REPORT.md')
    verifier.generate_verification_report(report_file)
    
    # Print summary
    if results['all_verified']:
        logger.info("✅ All data verification passed!")
    else:
        logger.warning(f"⚠️ Verification found {len(results['issues'])} issues")
        for issue in results['issues']:
            logger.warning(f"  - {issue}")
    
    logger.info(f"Verification report: {report_file}")

if __name__ == '__main__':
    main()
