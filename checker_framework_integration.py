#!/usr/bin/env python3
"""
Checker Framework Integration Module

This module provides integration with the Checker Framework for evaluating
annotation quality and providing feedback for reinforcement learning.
"""

import os
import subprocess
import tempfile
import shutil
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

class CheckerType(Enum):
    """Types of Checker Framework checkers"""
    NULLNESS = "org.checkerframework.checker.nullness.NullnessChecker"
    INDEX = "org.checkerframework.checker.index.IndexChecker"
    INTERNING = "org.checkerframework.checker.interning.InterningChecker"
    LOCK = "org.checkerframework.checker.lock.LockChecker"
    REGEX = "org.checkerframework.checker.regex.RegexChecker"
    SIGNATURE = "org.checkerframework.checker.signature.SignatureChecker"

@dataclass
class WarningInfo:
    """Information about a Checker Framework warning"""
    file_path: str
    line_number: int
    column_number: int
    warning_type: str
    message: str
    checker_type: str
    severity: str = "warning"

@dataclass
class EvaluationResult:
    """Result of Checker Framework evaluation"""
    original_warnings: List[WarningInfo]
    new_warnings: List[WarningInfo]
    warning_count_change: int
    success: bool
    error_message: Optional[str] = None
    compilation_success: bool = True

class CheckerFrameworkEvaluator:
    """Evaluates Java code using Checker Framework"""
    
    def __init__(self, checker_framework_home: str = "/home/ubuntu/checker-framework"):
        self.checker_framework_home = checker_framework_home
        self.checker_cp = self._build_classpath()
        self.temp_dir = None
        
    def _build_classpath(self) -> str:
        """Build the Checker Framework classpath"""
        cp_parts = []
        
        # Add Checker Framework JARs
        checker_dist = os.path.join(self.checker_framework_home, "checker", "dist")
        if os.path.exists(checker_dist):
            for jar_file in os.listdir(checker_dist):
                if jar_file.endswith('.jar'):
                    cp_parts.append(os.path.join(checker_dist, jar_file))
        
        # Add dataflow JARs
        dataflow_dist = os.path.join(self.checker_framework_home, "dataflow", "build", "libs")
        if os.path.exists(dataflow_dist):
            for jar_file in os.listdir(dataflow_dist):
                if jar_file.endswith('.jar'):
                    cp_parts.append(os.path.join(dataflow_dist, jar_file))
        
        return ":".join(cp_parts)
    
    def evaluate_file(self, java_file: str, checker_type: CheckerType = CheckerType.NULLNESS) -> EvaluationResult:
        """Evaluate a single Java file with Checker Framework"""
        try:
            # Create temporary directory for compilation
            self.temp_dir = tempfile.mkdtemp()
            
            # Get original warnings
            original_warnings = self._run_checker(java_file, checker_type)
            
            # Return evaluation result
            return EvaluationResult(
                original_warnings=original_warnings,
                new_warnings=[],  # Will be filled by comparison
                warning_count_change=0,
                success=True,
                compilation_success=True
            )
            
        except Exception as e:
            return EvaluationResult(
                original_warnings=[],
                new_warnings=[],
                warning_count_change=0,
                success=False,
                error_message=str(e),
                compilation_success=False
            )
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def compare_evaluations(self, original_file: str, annotated_file: str, 
                           checker_type: CheckerType = CheckerType.NULLNESS) -> EvaluationResult:
        """Compare Checker Framework evaluations of original and annotated files"""
        try:
            # Evaluate original file
            original_result = self.evaluate_file(original_file, checker_type)
            if not original_result.success:
                return original_result
            
            # Evaluate annotated file
            annotated_result = self.evaluate_file(annotated_file, checker_type)
            if not annotated_result.success:
                return annotated_result
            
            # Calculate warning count change
            warning_count_change = len(annotated_result.original_warnings) - len(original_result.original_warnings)
            
            return EvaluationResult(
                original_warnings=original_result.original_warnings,
                new_warnings=annotated_result.original_warnings,
                warning_count_change=warning_count_change,
                success=True,
                compilation_success=True
            )
            
        except Exception as e:
            return EvaluationResult(
                original_warnings=[],
                new_warnings=[],
                warning_count_change=0,
                success=False,
                error_message=str(e),
                compilation_success=False
            )
    
    def _run_checker(self, java_file: str, checker_type: CheckerType) -> List[WarningInfo]:
        """Run Checker Framework on a Java file and parse warnings"""
        try:
            # Build javac command
            cmd = [
                'javac',
                '-cp', self.checker_cp,
                '-processor', checker_type.value,
                '-Xmaxwarns', '1000',
                '-d', self.temp_dir,
                java_file
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Parse warnings from stderr
            warnings = self._parse_warnings(result.stderr, java_file, checker_type)
            
            return warnings
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running Checker Framework on {java_file}")
            return []
        except Exception as e:
            print(f"Error running Checker Framework: {e}")
            return []
    
    def _parse_warnings(self, stderr: str, file_path: str, checker_type: CheckerType) -> List[WarningInfo]:
        """Parse warnings from Checker Framework stderr output"""
        warnings = []
        
        if not stderr:
            return warnings
        
        # Pattern to match Checker Framework warnings
        warning_pattern = r'(\S+\.java):(\d+):\s*warning:\s*\[([^\]]+)\]\s*(.+)'
        
        for line in stderr.split('\n'):
            match = re.match(warning_pattern, line)
            if match:
                file_name, line_num, warning_type, message = match.groups()
                
                # Extract column number if present
                column_match = re.search(r':(\d+):', line)
                column_num = int(column_match.group(1)) if column_match else 0
                
                warnings.append(WarningInfo(
                    file_path=file_name,
                    line_number=int(line_num),
                    column_number=column_num,
                    warning_type=warning_type,
                    message=message.strip(),
                    checker_type=checker_type.value
                ))
        
        return warnings
    
    def get_reward(self, evaluation_result: EvaluationResult) -> float:
        """Calculate reward based on evaluation result"""
        if not evaluation_result.success:
            return -1.0  # Penalty for failed evaluation
        
        if not evaluation_result.compilation_success:
            return -0.5  # Penalty for compilation failure
        
        # Reward based on warning count change
        warning_change = evaluation_result.warning_count_change
        
        if warning_change < 0:
            # Reduced warnings - positive reward
            return abs(warning_change) / max(len(evaluation_result.original_warnings), 1)
        elif warning_change == 0:
            # No change - neutral reward
            return 0.0
        else:
            # Increased warnings - negative reward
            return -warning_change / max(len(evaluation_result.original_warnings), 1)
    
    def get_detailed_feedback(self, evaluation_result: EvaluationResult) -> Dict:
        """Get detailed feedback about the evaluation"""
        feedback = {
            'success': evaluation_result.success,
            'compilation_success': evaluation_result.compilation_success,
            'warning_count_change': evaluation_result.warning_count_change,
            'original_warning_count': len(evaluation_result.original_warnings),
            'new_warning_count': len(evaluation_result.new_warnings),
            'reward': self.get_reward(evaluation_result),
            'warnings_by_type': {},
            'error_message': evaluation_result.error_message
        }
        
        # Group warnings by type
        for warning in evaluation_result.new_warnings:
            warning_type = warning.warning_type
            if warning_type not in feedback['warnings_by_type']:
                feedback['warnings_by_type'][warning_type] = 0
            feedback['warnings_by_type'][warning_type] += 1
        
        return feedback

class BatchEvaluator:
    """Evaluates multiple files in batch"""
    
    def __init__(self, checker_framework_home: str = "/home/ubuntu/checker-framework"):
        self.evaluator = CheckerFrameworkEvaluator(checker_framework_home)
    
    def evaluate_batch(self, file_pairs: List[Tuple[str, str]], 
                      checker_type: CheckerType = CheckerType.NULLNESS) -> List[EvaluationResult]:
        """Evaluate multiple file pairs"""
        results = []
        
        for original_file, annotated_file in file_pairs:
            print(f"Evaluating {original_file} -> {annotated_file}")
            result = self.evaluator.compare_evaluations(original_file, annotated_file, checker_type)
            results.append(result)
        
        return results
    
    def get_batch_statistics(self, results: List[EvaluationResult]) -> Dict:
        """Get statistics for a batch of evaluations"""
        total_files = len(results)
        successful_evaluations = sum(1 for r in results if r.success)
        successful_compilations = sum(1 for r in results if r.compilation_success)
        
        total_warning_reduction = sum(r.warning_count_change for r in results if r.warning_count_change < 0)
        total_warning_increase = sum(r.warning_count_change for r in results if r.warning_count_change > 0)
        
        avg_reward = sum(self.evaluator.get_reward(r) for r in results) / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'successful_evaluations': successful_evaluations,
            'successful_compilations': successful_compilations,
            'success_rate': successful_evaluations / total_files if total_files > 0 else 0,
            'compilation_success_rate': successful_compilations / total_files if total_files > 0 else 0,
            'total_warning_reduction': total_warning_reduction,
            'total_warning_increase': total_warning_increase,
            'net_warning_change': total_warning_reduction + total_warning_increase,
            'average_reward': avg_reward
        }

class CheckerFrameworkConfig:
    """Configuration for Checker Framework evaluation"""
    
    def __init__(self):
        self.checker_types = {
            'nullness': CheckerType.NULLNESS,
            'index': CheckerType.INDEX,
            'interning': CheckerType.INTERNING,
            'lock': CheckerType.LOCK,
            'regex': CheckerType.REGEX,
            'signature': CheckerType.SIGNATURE
        }
        
        self.default_checker = CheckerType.NULLNESS
        self.timeout_seconds = 30
        self.max_warnings = 1000
    
    def get_checker_type(self, checker_name: str) -> CheckerType:
        """Get checker type by name"""
        return self.checker_types.get(checker_name.lower(), self.default_checker)

def main():
    """Test the Checker Framework integration"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python checker_framework_integration.py <java_file> [checker_type]")
        sys.exit(1)
    
    java_file = sys.argv[1]
    checker_name = sys.argv[2] if len(sys.argv) > 2 else 'nullness'
    
    if not os.path.exists(java_file):
        print(f"File not found: {java_file}")
        sys.exit(1)
    
    # Initialize evaluator
    config = CheckerFrameworkConfig()
    checker_type = config.get_checker_type(checker_name)
    evaluator = CheckerFrameworkEvaluator()
    
    print(f"Evaluating {java_file} with {checker_type.value}")
    
    # Evaluate the file
    result = evaluator.evaluate_file(java_file, checker_type)
    
    if result.success:
        print(f"Evaluation successful!")
        print(f"Found {len(result.original_warnings)} warnings")
        
        for warning in result.original_warnings:
            print(f"  Line {warning.line_number}: [{warning.warning_type}] {warning.message}")
        
        # Get detailed feedback
        feedback = evaluator.get_detailed_feedback(result)
        print(f"\nDetailed Feedback:")
        print(f"  Reward: {feedback['reward']:.3f}")
        print(f"  Warning count: {feedback['original_warning_count']}")
        
    else:
        print(f"Evaluation failed: {result.error_message}")

if __name__ == '__main__':
    main()
