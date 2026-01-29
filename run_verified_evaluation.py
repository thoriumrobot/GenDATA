#!/usr/bin/env python3
"""
Run Verified Evaluation

Main entry point for running the verified annotation placement evaluation.
This script provides options for either:
1. Re-verifying existing evaluation results
2. Running a new verified evaluation from scratch

The verified evaluation ensures that warning reduction claims are accurate
by detecting checker crashes and verifying that the checker actually ran.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_existing_results(args):
    """Re-verify existing evaluation results"""
    from generate_verified_report import generate_verified_report
    
    logger.info("="*80)
    logger.info("VERIFYING EXISTING EVALUATION RESULTS")
    logger.info("="*80)
    
    report = generate_verified_report(
        input_report_path=args.input,
        output_report_path=args.output,
        temp_repos_dir=args.temp_repos,
        re_run_checker=args.re_run
    )
    
    return report


def run_new_evaluation(args):
    """Run a new verified evaluation"""
    from verified_evaluation_wrapper import VerifiedEvaluationWrapper
    
    logger.info("="*80)
    logger.info("RUNNING NEW VERIFIED EVALUATION")
    logger.info("="*80)
    
    wrapper = VerifiedEvaluationWrapper(
        work_dir=args.work_dir,
        checker_cp=args.checker_cp,
        cfg_dir=args.cfg_dir,
        timeout=args.timeout,
        save_checker_outputs=not args.no_save_outputs
    )
    
    report = wrapper.run_verified_evaluation(
        candidates_file=args.candidates,
        output_file=args.output
    )
    
    return report


def test_crash_detector(args):
    """Test the crash detector with sample outputs"""
    from checker_crash_detector import detect_checker_crash, verify_checker_processed_files
    
    logger.info("="*80)
    logger.info("TESTING CRASH DETECTOR")
    logger.info("="*80)
    
    test_cases = [
        {
            'name': 'Normal output with warnings',
            'output': '''
/home/ubuntu/project/src/Main.java:10: warning: [index] possible array index out of bounds
        arr[i] = 5;
             ^
/home/ubuntu/project/src/Main.java:15: warning: [lowerbound] index might be negative
        arr[x] = 10;
             ^
2 warnings
            ''',
            'expected_crash': False
        },
        {
            'name': 'OutOfMemory crash',
            'output': '''
Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
	at java.base/java.util.Arrays.copyOf(Arrays.java:3720)
	at org.checkerframework.checker.index.IndexChecker.process(IndexChecker.java:100)
	at com.sun.tools.javac.main.Main.compile(Main.java:561)
            ''',
            'expected_crash': True
        },
        {
            'name': 'Compilation error (not crash)',
            'output': '''
/home/ubuntu/project/src/Main.java:5: error: cannot find symbol
    UnknownClass obj = new UnknownClass();
    ^
  symbol:   class UnknownClass
  location: class Main
1 error
            ''',
            'expected_crash': False
        },
        {
            'name': 'Stack overflow crash',
            'output': '''
Exception in thread "main" java.lang.StackOverflowError
	at org.checkerframework.dataflow.cfg.CFGBuilder$PhaseOneResult.visit(CFGBuilder.java:1234)
	at org.checkerframework.dataflow.cfg.CFGBuilder$PhaseOneResult.visit(CFGBuilder.java:1234)
	at org.checkerframework.dataflow.cfg.CFGBuilder$PhaseOneResult.visit(CFGBuilder.java:1234)
            ''',
            'expected_crash': True
        },
        {
            'name': 'Empty output',
            'output': '',
            'expected_crash': True
        },
        {
            'name': 'Internal compiler error',
            'output': '''
An exception has occurred in the compiler (17.0.2). Please file a bug.
Internal compiler error
	at com.sun.tools.javac.comp.Check.checkNonCyclic(Check.java:1834)
            ''',
            'expected_crash': True
        },
        {
            'name': 'Normal output with no warnings',
            'output': '''
Note: Processing class Main
0 warnings
            ''',
            'expected_crash': False
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = detect_checker_crash(test['output'])
        processing = verify_checker_processed_files(test['output'])
        
        success = result.crashed == test['expected_crash']
        status = "PASS" if success else "FAIL"
        
        if success:
            passed += 1
        else:
            failed += 1
        
        logger.info(f"\n[{status}] {test['name']}")
        logger.info(f"  Expected crash: {test['expected_crash']}")
        logger.info(f"  Detected crash: {result.crashed}")
        logger.info(f"  Confidence: {result.confidence:.2f}")
        logger.info(f"  Files processed: {processing.files_processed}")
        
        if result.crashed:
            logger.info(f"  Crash reason: {result.crash_reason}")
        if result.crash_indicators_found:
            logger.info(f"  Indicators: {result.crash_indicators_found[:3]}")
    
    logger.info("\n" + "="*40)
    logger.info(f"Tests passed: {passed}/{passed + failed}")
    logger.info(f"Tests failed: {failed}")
    
    return failed == 0


def quick_verify(args):
    """Quickly verify a single checker output file or string"""
    from checker_crash_detector import detect_checker_crash, verify_checker_processed_files
    
    if args.file:
        with open(args.file) as f:
            output = f.read()
    else:
        output = args.string
    
    crash_result = detect_checker_crash(output)
    processing = verify_checker_processed_files(output)
    
    print("\n" + "="*60)
    print("VERIFICATION RESULT")
    print("="*60)
    print(f"Crashed: {crash_result.crashed}")
    if crash_result.crashed:
        print(f"Reason: {crash_result.crash_reason}")
    print(f"Has stack trace: {crash_result.has_stack_trace}")
    print(f"Has compilation errors: {crash_result.has_compilation_errors}")
    print(f"Has success indicators: {crash_result.has_success_indicators}")
    print(f"Crash indicators found: {crash_result.crash_indicators_found[:5] if crash_result.crash_indicators_found else 'None'}")
    print(f"Confidence: {crash_result.confidence:.2f}")
    print("-"*40)
    print(f"Files processed: {processing.files_processed}")
    print(f"Files mentioned: {processing.files_mentioned}")
    print(f"Warnings found: {processing.warning_count}")
    print(f"Errors found: {processing.error_count}")
    print("="*60)
    
    return not crash_result.crashed


def print_summary(args):
    """Print summary of existing evaluation report with verification status"""
    input_path = args.input or 'annotation_evaluation/evaluation_report.json'
    
    logger.info(f"Loading report from: {input_path}")
    
    try:
        with open(input_path) as f:
            report = json.load(f)
    except FileNotFoundError:
        logger.error(f"Report not found: {input_path}")
        return False
    
    results = report.get('results', [])
    
    print("\n" + "="*80)
    print("EVALUATION REPORT SUMMARY")
    print("="*80)
    
    suspicious_count = 0
    
    for project in results:
        project_name = project.get('project_name', 'unknown')
        baseline = project.get('baseline_warnings', 0)
        
        print(f"\n{project_name}:")
        print(f"  Baseline warnings: {baseline}")
        
        model_results = project.get('model_results', [])
        
        for model in model_results:
            model_name = model.get('base_model', 'unknown')
            warnings_after = model.get('warnings_after', 0)
            reduction = model.get('reduction_percentage', 0)
            compilation = model.get('compilation_success', True)
            
            # Flag suspicious results
            suspicious = reduction == 100.0 and baseline > 0 and warnings_after == 0
            flag = " ⚠️ SUSPICIOUS" if suspicious else ""
            
            if suspicious:
                suspicious_count += 1
            
            print(f"    {model_name}: {reduction:.1f}% reduction "
                  f"({baseline} → {warnings_after}){flag}")
    
    print("\n" + "-"*40)
    print(f"Suspicious 100% reductions: {suspicious_count}")
    if suspicious_count > 0:
        print("\n⚠️  Use --mode verify to verify these results")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Verified Annotation Placement Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Verify existing results
  python run_verified_evaluation.py --mode verify --input annotation_evaluation/evaluation_report.json
  
  # Run new verified evaluation
  python run_verified_evaluation.py --mode new --candidates project_candidates.json
  
  # Test crash detector
  python run_verified_evaluation.py --mode test
  
  # Quick verify a checker output file
  python run_verified_evaluation.py --mode quick --file checker_output.txt
  
  # Print summary of existing report
  python run_verified_evaluation.py --mode summary
'''
    )
    
    parser.add_argument(
        '--mode',
        choices=['verify', 'new', 'test', 'quick', 'summary'],
        default='verify',
        help='Mode: verify (existing results), new (run evaluation), test (test crash detector), quick (verify single output), summary (print report summary)'
    )
    
    # Common options
    parser.add_argument(
        '--output', '-o',
        default='annotation_evaluation/verified_evaluation_report.json',
        help='Output file path'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # Verify mode options
    parser.add_argument(
        '--input', '-i',
        default='annotation_evaluation/evaluation_report.json',
        help='Input evaluation report (for verify mode)'
    )
    parser.add_argument(
        '--temp-repos',
        default='annotation_evaluation/temp_repos',
        help='Temp repos directory (for verify mode)'
    )
    parser.add_argument(
        '--re-run',
        action='store_true',
        help='Re-run checker on projects (for verify mode)'
    )
    
    # New evaluation options
    parser.add_argument(
        '--candidates',
        default='project_discovery_manual/lower_bound_project_candidates.json',
        help='Project candidates file (for new mode)'
    )
    parser.add_argument(
        '--work-dir',
        default='./annotation_evaluation_verified',
        help='Working directory (for new mode)'
    )
    parser.add_argument(
        '--checker-cp',
        help='Checker Framework classpath'
    )
    parser.add_argument(
        '--cfg-dir',
        help='CFG directory'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout for checker runs'
    )
    parser.add_argument(
        '--no-save-outputs',
        action='store_true',
        help='Do not save checker outputs'
    )
    
    # Quick mode options
    parser.add_argument(
        '--file', '-f',
        help='Checker output file to verify (for quick mode)'
    )
    parser.add_argument(
        '--string', '-s',
        help='Checker output string to verify (for quick mode)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute based on mode
    if args.mode == 'verify':
        result = verify_existing_results(args)
        return 0 if result else 1
        
    elif args.mode == 'new':
        result = run_new_evaluation(args)
        return 0 if result.results else 1
        
    elif args.mode == 'test':
        success = test_crash_detector(args)
        return 0 if success else 1
        
    elif args.mode == 'quick':
        if not args.file and not args.string:
            parser.error("--file or --string required for quick mode")
        success = quick_verify(args)
        return 0 if success else 1
        
    elif args.mode == 'summary':
        success = print_summary(args)
        return 0 if success else 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
