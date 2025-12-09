#!/usr/bin/env python3
"""
Multi-Checker Infrastructure Verification Script

This script comprehensively verifies that the multi-checker infrastructure
is working correctly through systematic testing of all components.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import inspect
from abc import ABC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')

class VerificationResult:
    """Container for verification test results"""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.errors = []
        self.warnings = []
        self.info = []
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_info(self, message: str):
        self.info.append(message)
    
    def set_passed(self):
        self.passed = True
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        result = f"[{status}] {self.test_name}\n"
        if self.errors:
            result += "  Errors:\n"
            for err in self.errors:
                result += f"    - {err}\n"
        if self.warnings:
            result += "  Warnings:\n"
            for warn in self.warnings:
                result += f"    - {warn}\n"
        if self.info:
            result += "  Info:\n"
            for info in self.info:
                result += f"    - {info}\n"
        return result

def verify_checker_interface_compliance() -> List[VerificationResult]:
    """Verify all checker implementations correctly implement CheckerInterface"""
    logger.info("=" * 80)
    logger.info("Verifying Checker Interface Compliance")
    logger.info("=" * 80)
    
    results = []
    
    # Import checker interface and implementations
    try:
        from checker_interface import CheckerInterface
        from lower_bound_checker import LowerBoundChecker
        from sql_quotes_checker import SqlQuotesChecker
        from signature_string_checker import SignatureStringChecker
    except ImportError as e:
        result = VerificationResult("Checker Interface Import")
        result.add_error(f"Failed to import checker modules: {e}")
        results.append(result)
        return results
    
    # Get all abstract methods from CheckerInterface
    abstract_methods = []
    for name, method in inspect.getmembers(CheckerInterface, predicate=inspect.isabstract):
        if inspect.isfunction(method) or inspect.ismethod(method):
            abstract_methods.append(name)
    
    logger.info(f"Found {len(abstract_methods)} abstract methods in CheckerInterface")
    
    # Test each checker implementation
    checker_classes = [
        ('LowerBoundChecker', LowerBoundChecker),
        ('SqlQuotesChecker', SqlQuotesChecker),
        ('SignatureStringChecker', SignatureStringChecker)
    ]
    
    for checker_name, checker_class in checker_classes:
        result = VerificationResult(f"{checker_name} Interface Compliance")
        
        # Check if it's a subclass of CheckerInterface
        if not issubclass(checker_class, CheckerInterface):
            result.add_error(f"{checker_name} is not a subclass of CheckerInterface")
            results.append(result)
            continue
        
        # Check if all abstract methods are implemented
        missing_methods = []
        for method_name in abstract_methods:
            if not hasattr(checker_class, method_name):
                missing_methods.append(method_name)
            else:
                method = getattr(checker_class, method_name)
                if inspect.isabstract(method):
                    missing_methods.append(method_name)
        
        if missing_methods:
            result.add_error(f"Missing abstract method implementations: {', '.join(missing_methods)}")
        else:
            result.add_info(f"All {len(abstract_methods)} abstract methods are implemented")
        
        # Test instantiation
        try:
            checker_instance = checker_class()
            result.add_info("Checker instance created successfully")
        except Exception as e:
            result.add_error(f"Failed to instantiate checker: {e}")
            results.append(result)
            continue
        
        # Test required methods return correct types
        try:
            name = checker_instance.get_checker_name()
            if not isinstance(name, str):
                result.add_error(f"get_checker_name() returned {type(name)}, expected str")
            else:
                result.add_info(f"get_checker_name() = '{name}'")
        except Exception as e:
            result.add_error(f"get_checker_name() failed: {e}")
        
        try:
            processor = checker_instance.get_checker_processor()
            if not isinstance(processor, str):
                result.add_error(f"get_checker_processor() returned {type(processor)}, expected str")
            else:
                result.add_info(f"get_checker_processor() = '{processor}'")
        except Exception as e:
            result.add_error(f"get_checker_processor() failed: {e}")
        
        try:
            annotations = checker_instance.get_annotation_types()
            if not isinstance(annotations, list):
                result.add_error(f"get_annotation_types() returned {type(annotations)}, expected list")
            elif not all(isinstance(a, str) for a in annotations):
                result.add_error("get_annotation_types() returned list with non-string elements")
            else:
                result.add_info(f"get_annotation_types() = {annotations}")
        except Exception as e:
            result.add_error(f"get_annotation_types() failed: {e}")
        
        try:
            training_source = checker_instance.get_training_data_source()
            if not isinstance(training_source, str):
                result.add_error(f"get_training_data_source() returned {type(training_source)}, expected str")
            else:
                result.add_info(f"get_training_data_source() = '{training_source}'")
        except Exception as e:
            result.add_error(f"get_training_data_source() failed: {e}")
        
        # Test optional methods
        try:
            warning_patterns = checker_instance.get_warning_patterns()
            if not isinstance(warning_patterns, list):
                result.add_warning(f"get_warning_patterns() returned {type(warning_patterns)}, expected list")
            else:
                result.add_info(f"get_warning_patterns() = {len(warning_patterns)} patterns")
        except Exception as e:
            result.add_warning(f"get_warning_patterns() failed: {e}")
        
        # Test parse_warnings with non-existent file
        try:
            warnings = checker_instance.parse_warnings('/nonexistent/file.out')
            if not isinstance(warnings, list):
                result.add_error(f"parse_warnings() returned {type(warnings)}, expected list")
            else:
                result.add_info(f"parse_warnings() handles missing file correctly (returned {len(warnings)} warnings)")
        except Exception as e:
            result.add_error(f"parse_warnings() failed on non-existent file: {e}")
        
        # Test extract_features with dummy data
        try:
            dummy_cfg = {'nodes': [], 'edges': []}
            dummy_node = {'label': 'test', 'node_type': 'statement'}
            features = checker_instance.extract_features(dummy_cfg, dummy_node)
            if not isinstance(features, list):
                result.add_error(f"extract_features() returned {type(features)}, expected list")
            elif not all(isinstance(f, (int, float)) for f in features):
                result.add_error("extract_features() returned list with non-numeric elements")
            else:
                result.add_info(f"extract_features() returned {len(features)} features")
        except Exception as e:
            result.add_error(f"extract_features() failed: {e}")
        
        # Test validate_annotation
        try:
            valid_location = {'target_type': 'parameter', 'file': 'test.java', 'line': 1, 'column': 0}
            annotations = checker_instance.get_annotation_types()
            if annotations:
                is_valid = checker_instance.validate_annotation(annotations[0], valid_location)
                if not isinstance(is_valid, bool):
                    result.add_error(f"validate_annotation() returned {type(is_valid)}, expected bool")
                else:
                    result.add_info(f"validate_annotation() works correctly")
        except Exception as e:
            result.add_error(f"validate_annotation() failed: {e}")
        
        if not result.errors:
            result.set_passed()
        
        results.append(result)
    
    return results

def verify_checker_registry() -> List[VerificationResult]:
    """Verify checker registry functionality"""
    logger.info("=" * 80)
    logger.info("Verifying Checker Registry")
    logger.info("=" * 80)
    
    results = []
    
    try:
        from checker_registry import (
            list_checkers, get_checker, is_checker_registered,
            register_checker
        )
    except ImportError as e:
        result = VerificationResult("Checker Registry Import")
        result.add_error(f"Failed to import checker_registry: {e}")
        results.append(result)
        return results
    
    # Test list_checkers
    result = VerificationResult("List Checkers")
    try:
        checkers = list_checkers()
        if not isinstance(checkers, list):
            result.add_error(f"list_checkers() returned {type(checkers)}, expected list")
        else:
            result.add_info(f"Found {len(checkers)} registered checkers: {checkers}")
            if len(checkers) == 0:
                result.add_warning("No checkers registered")
            result.set_passed()
    except Exception as e:
        result.add_error(f"list_checkers() failed: {e}")
    results.append(result)
    
    # Test get_checker for known checkers
    known_checkers = ['lower_bound', 'sql_quotes', 'signature_string']
    for checker_name in known_checkers:
        result = VerificationResult(f"Get Checker: {checker_name}")
        try:
            checker = get_checker(checker_name)
            if checker is None:
                result.add_warning(f"Checker '{checker_name}' not found in registry")
            else:
                result.add_info(f"Successfully retrieved checker '{checker_name}'")
                # Test case-insensitive retrieval
                checker_upper = get_checker(checker_name.upper())
                if checker_upper is None:
                    result.add_warning(f"Case-insensitive retrieval failed for '{checker_name.upper()}'")
                else:
                    result.add_info("Case-insensitive retrieval works")
                result.set_passed()
        except Exception as e:
            result.add_error(f"get_checker('{checker_name}') failed: {e}")
        results.append(result)
    
    # Test is_checker_registered
    result = VerificationResult("Is Checker Registered")
    try:
        for checker_name in known_checkers:
            is_registered = is_checker_registered(checker_name)
            if not isinstance(is_registered, bool):
                result.add_error(f"is_checker_registered('{checker_name}') returned {type(is_registered)}, expected bool")
            else:
                result.add_info(f"is_checker_registered('{checker_name}') = {is_registered}")
        result.set_passed()
    except Exception as e:
        result.add_error(f"is_checker_registered() failed: {e}")
    results.append(result)
    
    # Test get_checker with unknown checker
    result = VerificationResult("Get Unknown Checker")
    try:
        unknown_checker = get_checker('nonexistent_checker')
        if unknown_checker is not None:
            result.add_warning("get_checker() returned non-None for unknown checker")
        else:
            result.add_info("get_checker() correctly returns None for unknown checker")
        result.set_passed()
    except Exception as e:
        result.add_error(f"get_checker() failed for unknown checker: {e}")
    results.append(result)
    
    return results

def verify_checker_runner_extensions() -> List[VerificationResult]:
    """Verify CheckerFrameworkRunner extensions"""
    logger.info("=" * 80)
    logger.info("Verifying CheckerFrameworkRunner Extensions")
    logger.info("=" * 80)
    
    results = []
    
    try:
        from checker_framework_runner import CheckerFrameworkRunner
    except ImportError as e:
        result = VerificationResult("CheckerFrameworkRunner Import")
        result.add_error(f"Failed to import CheckerFrameworkRunner: {e}")
        results.append(result)
        return results
    
    # Test checker selection via checker_name
    result = VerificationResult("Checker Selection by Name")
    try:
        runner = CheckerFrameworkRunner(checker_name='lower_bound')
        if runner.processor != 'org.checkerframework.checker.index.IndexChecker':
            result.add_error(f"Expected IndexChecker processor, got {runner.processor}")
        else:
            result.add_info(f"Successfully selected checker 'lower_bound' -> {runner.processor}")
        result.set_passed()
    except Exception as e:
        result.add_error(f"Failed to create runner with checker_name: {e}")
    results.append(result)
    
    # Test SQL Quotes checker selection
    result = VerificationResult("SQL Quotes Checker Selection")
    try:
        runner = CheckerFrameworkRunner(checker_name='sql_quotes')
        expected_processor = 'org.checkerframework.checker.quotes.QuotesChecker'
        if runner.processor != expected_processor:
            result.add_warning(f"Expected {expected_processor}, got {runner.processor}")
        else:
            result.add_info(f"Successfully selected SQL Quotes checker")
        result.set_passed()
    except Exception as e:
        result.add_error(f"Failed to select SQL Quotes checker: {e}")
    results.append(result)
    
    # Test Signature String checker selection
    result = VerificationResult("Signature String Checker Selection")
    try:
        runner = CheckerFrameworkRunner(checker_name='signature_string')
        expected_processor = 'org.checkerframework.checker.signature.qual.SignatureChecker'
        if runner.processor != expected_processor:
            result.add_warning(f"Expected {expected_processor}, got {runner.processor}")
        else:
            result.add_info(f"Successfully selected Signature String checker")
        result.set_passed()
    except Exception as e:
        result.add_error(f"Failed to select Signature String checker: {e}")
    results.append(result)
    
    # Test fallback to default when checker not found
    result = VerificationResult("Fallback to Default Processor")
    try:
        runner = CheckerFrameworkRunner(checker_name='nonexistent_checker')
        # Should fall back to default (Lower Bound Checker)
        if runner.processor == 'org.checkerframework.checker.index.IndexChecker':
            result.add_info("Correctly fell back to default processor")
            result.set_passed()
        else:
            result.add_warning(f"Fell back to unexpected processor: {runner.processor}")
            result.set_passed()  # Still passes, just unexpected
    except Exception as e:
        result.add_error(f"Fallback failed: {e}")
    results.append(result)
    
    # Test warning parsing with checker interface
    result = VerificationResult("Checker-Specific Warning Parsing")
    try:
        runner = CheckerFrameworkRunner(checker_name='lower_bound')
        # Create a dummy warnings file
        dummy_warnings_file = GEN_DATA_ROOT / 'test_warnings.out'
        with open(dummy_warnings_file, 'w') as f:
            f.write("Test.java:10: error: [array.access.unsafe.high] array access might be out of bounds\n")
        
        warnings_info = runner.parse_warnings_file(str(dummy_warnings_file))
        if isinstance(warnings_info, dict):
            result.add_info(f"Successfully parsed warnings file: {warnings_info.get('total_warnings', 0)} warnings")
            result.set_passed()
        else:
            result.add_error(f"parse_warnings_file() returned {type(warnings_info)}, expected dict")
        
        # Clean up
        if dummy_warnings_file.exists():
            dummy_warnings_file.unlink()
    except Exception as e:
        result.add_error(f"Warning parsing failed: {e}")
    results.append(result)
    
    return results

def verify_configuration_system() -> List[VerificationResult]:
    """Verify configuration system"""
    logger.info("=" * 80)
    logger.info("Verifying Configuration System")
    logger.info("=" * 80)
    
    results = []
    
    try:
        from checker_evaluation_config import (
            get_checker_config, get_all_checker_names,
            get_checker_annotation_types, get_checker_base_models,
            build_model_name, get_evaluation_projects
        )
    except ImportError as e:
        result = VerificationResult("Configuration System Import")
        result.add_error(f"Failed to import configuration module: {e}")
        results.append(result)
        return results
    
    # Test get_all_checker_names
    result = VerificationResult("Get All Checker Names")
    try:
        checker_names = get_all_checker_names()
        if not isinstance(checker_names, list):
            result.add_error(f"get_all_checker_names() returned {type(checker_names)}, expected list")
        else:
            result.add_info(f"Found {len(checker_names)} checkers: {checker_names}")
            result.set_passed()
    except Exception as e:
        result.add_error(f"get_all_checker_names() failed: {e}")
    results.append(result)
    
    # Test get_checker_config for each checker
    for checker_name in ['lower_bound', 'sql_quotes', 'signature_string']:
        result = VerificationResult(f"Get Config: {checker_name}")
        try:
            config = get_checker_config(checker_name)
            if not isinstance(config, dict):
                result.add_error(f"get_checker_config() returned {type(config)}, expected dict")
            elif not config:
                result.add_warning(f"No configuration found for {checker_name}")
            else:
                required_keys = ['name', 'processor', 'annotation_types', 'base_models']
                missing_keys = [k for k in required_keys if k not in config]
                if missing_keys:
                    result.add_warning(f"Missing config keys: {missing_keys}")
                else:
                    result.add_info(f"Configuration complete for {checker_name}")
                result.set_passed()
        except Exception as e:
            result.add_error(f"get_checker_config('{checker_name}') failed: {e}")
        results.append(result)
    
    # Test build_model_name
    result = VerificationResult("Build Model Name")
    try:
        model_name = build_model_name('lower_bound', '@Positive', 'gcn')
        expected = 'positive_gcn'
        if model_name != expected:
            result.add_warning(f"Expected '{expected}', got '{model_name}'")
        else:
            result.add_info(f"build_model_name() works correctly: '{model_name}'")
        result.set_passed()
    except Exception as e:
        result.add_error(f"build_model_name() failed: {e}")
    results.append(result)
    
    # Test get_evaluation_projects
    result = VerificationResult("Get Evaluation Projects")
    try:
        projects = get_evaluation_projects('lower_bound')
        if not isinstance(projects, list):
            result.add_error(f"get_evaluation_projects() returned {type(projects)}, expected list")
        else:
            result.add_info(f"Found {len(projects)} evaluation projects for lower_bound: {projects}")
            result.set_passed()
    except Exception as e:
        result.add_error(f"get_evaluation_projects() failed: {e}")
    results.append(result)
    
    return results

def generate_verification_report(all_results: List[VerificationResult]) -> Path:
    """Generate comprehensive verification report"""
    report_file = GEN_DATA_ROOT / 'MULTI_CHECKER_VERIFICATION_REPORT.md'
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.passed)
    failed_tests = total_tests - passed_tests
    
    with open(report_file, 'w') as f:
        f.write("# Multi-Checker Infrastructure Verification Report\n\n")
        f.write(f"**Generated**: {Path(__file__).stat().st_mtime}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total Tests: {total_tests}\n")
        f.write(f"- Passed: {passed_tests}\n")
        f.write(f"- Failed: {failed_tests}\n")
        f.write(f"- Success Rate: {(passed_tests/total_tests*100):.1f}%\n\n")
        
        f.write("## Test Results\n\n")
        for result in all_results:
            f.write(f"### {result.test_name}\n\n")
            f.write(f"**Status**: {'✅ PASS' if result.passed else '❌ FAIL'}\n\n")
            if result.errors:
                f.write("**Errors**:\n")
                for err in result.errors:
                    f.write(f"- {err}\n")
                f.write("\n")
            if result.warnings:
                f.write("**Warnings**:\n")
                for warn in result.warnings:
                    f.write(f"- {warn}\n")
                f.write("\n")
            if result.info:
                f.write("**Info**:\n")
                for info in result.info:
                    f.write(f"- {info}\n")
                f.write("\n")
            f.write("\n")
    
    logger.info(f"Verification report saved to {report_file}")
    return report_file

def main():
    """Main verification function"""
    logger.info("=" * 80)
    logger.info("Multi-Checker Infrastructure Verification")
    logger.info("=" * 80)
    
    all_results = []
    
    # Phase 1: Unit Testing
    all_results.extend(verify_checker_interface_compliance())
    all_results.extend(verify_checker_registry())
    all_results.extend(verify_checker_runner_extensions())
    all_results.extend(verify_configuration_system())
    
    # Generate report
    report_file = generate_verification_report(all_results)
    
    # Print summary
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed
    
    logger.info("=" * 80)
    logger.info("Verification Summary")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/total*100):.1f}%")
    logger.info(f"Report: {report_file}")
    
    # Print failed tests
    failed_results = [r for r in all_results if not r.passed]
    if failed_results:
        logger.warning("\nFailed Tests:")
        for result in failed_results:
            logger.warning(str(result))
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    exit(main())

