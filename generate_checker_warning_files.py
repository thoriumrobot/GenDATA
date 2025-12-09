#!/usr/bin/env python3
"""
Generate Warning Files for GenDATA Checkers

This script generates warning files from Checker Framework test suites for the checkers
that GenDATA trains models for:
- Lower Bound Checker
- SQL Quotes Checker  
- Signature String Checker

The warning files are used as training data for the GenDATA models.
"""

import os
import sys
import logging
import tempfile
import re
from pathlib import Path
from typing import Dict, Optional, List, Set

# Add GenDATA root to path
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
sys.path.insert(0, str(GEN_DATA_ROOT))

from checker_framework_runner import CheckerFrameworkRunner
from checker_evaluation_config import CHECKER_CONFIGS, get_checker_config
from extract_test_suite_warnings import extract_warnings_from_test_suite
from parse_expected_errors import parse_expected_errors_from_test_suite

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GenDATA checkers (only these need warning files)
GENDATA_CHECKERS = ['lower_bound', 'sql_quotes', 'signature_string']

# Warning file naming convention
WARNING_FILE_NAMES = {
    'lower_bound': 'lower_bound_warnings.out',
    'sql_quotes': 'sql_quotes_warnings.out',
    'signature_string': 'signature_string_warnings.out'
}


def deduplicate_warnings(warning_lines: List[str]) -> List[str]:
    """
    Remove duplicate warnings from a list of warning lines.
    
    Warnings are considered duplicates if they have the same file, line, and column.
    """
    seen = set()
    unique_warnings = []
    
    for line in warning_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            unique_warnings.append(line)
            continue
        
        # Extract file:line:column to identify duplicates
        match = re.match(r'^(.+?):(\d+):(\d+):', line)
        if match:
            key = (match.group(1), match.group(2), match.group(3))
            if key not in seen:
                seen.add(key)
                unique_warnings.append(line)
        else:
            # Keep non-standard format lines
            unique_warnings.append(line)
    
    return unique_warnings


def merge_warning_files(files: List[Path], output_file: Path, checker_name: str) -> int:
    """
    Merge multiple warning files into one, deduplicating warnings.
    
    Returns:
        Number of unique warnings merged
    """
    all_warnings = []
    
    for warning_file in files:
        if not warning_file.exists():
            continue
        
        with open(warning_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    all_warnings.append(line)
    
    # Deduplicate
    unique_warnings = deduplicate_warnings(all_warnings)
    
    # Write merged warnings
    with open(output_file, 'w') as f:
        f.write(f"# Checker Framework {checker_name} Test Suite Warnings\n")
        f.write(f"# Merged from multiple extraction methods\n")
        f.write(f"# Total Unique Warnings: {len(unique_warnings)}\n")
        f.write(f"\n")
        for warning in unique_warnings:
            f.write(warning + '\n')
    
    return len(unique_warnings)


def generate_warning_file(checker_name: str, output_dir: Path = GEN_DATA_ROOT, 
                          use_extraction_methods: bool = True) -> Optional[Path]:
    """
    Generate warning file for a specific checker from its test suite.
    
    Args:
        checker_name: Name of the checker (lower_bound, sql_quotes, signature_string)
        output_dir: Directory to save warning file (default: /home/ubuntu/GenDATA)
        
    Returns:
        Path to generated warning file if successful, None otherwise
    """
    logger.info(f"=" * 80)
    logger.info(f"Generating warning file for {checker_name} checker")
    logger.info(f"=" * 80)
    
    # Get checker configuration
    config = get_checker_config(checker_name)
    if not config:
        logger.error(f"❌ No configuration found for checker '{checker_name}'")
        return None
    
    checker_display_name = config.get('name', checker_name)
    test_suite_path = config.get('test_suite', '')
    
    # Check if test suite exists
    if not test_suite_path:
        logger.warning(f"⚠️ No test suite path configured for {checker_display_name}")
        return None
    
    test_suite = Path(test_suite_path)
    if not test_suite.exists():
        logger.warning(f"⚠️ Test suite not found for {checker_display_name}")
        logger.warning(f"   Expected location: {test_suite_path}")
        logger.warning(f"   Training for {checker_display_name} will be blocked until test suite is available.")
        return None
    
    logger.info(f"📁 Test suite: {test_suite_path}")
    
    # Determine output file path
    output_file_name = WARNING_FILE_NAMES.get(checker_name, f'{checker_name}_warnings.out')
    output_file = output_dir / output_file_name
    
    logger.info(f"📄 Output file: {output_file}")
    
    # Initialize CheckerFrameworkRunner
    try:
        runner = CheckerFrameworkRunner(checker_name=checker_name)
        logger.info(f"✅ Initialized {checker_display_name} runner")
    except Exception as e:
        logger.error(f"❌ Failed to initialize {checker_display_name} runner: {e}")
        return None
    
    # Run checker on test suite
    logger.info(f"🚀 Running {checker_display_name} on test suite...")
    try:
        success = runner.run_checker_on_project(
            project_root=str(test_suite),
            output_file=str(output_file),
            max_files=None  # Process all files in test suite
        )
        
        if not success:
            logger.error(f"❌ Failed to run {checker_display_name} on test suite")
            return None
        
        # Verify warnings were generated (not just compilation errors)
        try:
            warning_count = runner.count_checker_warnings(str(output_file))
        except AttributeError:
            # Fallback: count lines that look like warnings
            warning_count = 0
            if output_file.exists():
                with open(output_file, 'r') as f:
                    for line in f:
                        if ':' in line and ('error:' in line or 'warning:' in line or 'compiler.' in line):
                            warning_count += 1
        
        # If no warnings found and extraction methods enabled, try alternative methods
        if warning_count == 0 and use_extraction_methods:
            logger.info(f"⚠️ No warnings from direct checker run. Trying extraction methods...")
            
            temp_files = []
            
            # Method 1: Annotation removal
            logger.info(f"📝 Method 1: Removing annotations and running checker...")
            temp_annotation_output = Path(tempfile.mktemp(suffix='_annotation.out'))
            try:
                annotation_success = extract_warnings_from_test_suite(
                    checker_name=checker_name,
                    test_suite_path=test_suite,
                    output_file=temp_annotation_output,
                    use_temp_dir=True
                )
                if annotation_success and temp_annotation_output.exists():
                    annotation_count = runner.count_checker_warnings(str(temp_annotation_output))
                    if annotation_count > 0:
                        logger.info(f"✅ Annotation removal method generated {annotation_count} warnings")
                        temp_files.append(temp_annotation_output)
                    else:
                        logger.info(f"⚠️ Annotation removal method generated 0 warnings")
                        temp_annotation_output.unlink(missing_ok=True)
                else:
                    logger.warning(f"⚠️ Annotation removal method failed")
            except Exception as e:
                logger.warning(f"⚠️ Annotation removal method error: {e}")
                temp_annotation_output.unlink(missing_ok=True)
            
            # Method 2: Parse expected errors
            logger.info(f"📝 Method 2: Parsing expected error comments...")
            temp_expected_output = Path(tempfile.mktemp(suffix='_expected.out'))
            try:
                expected_success = parse_expected_errors_from_test_suite(
                    test_suite_path=test_suite,
                    output_file=temp_expected_output,
                    checker_name=checker_name
                )
                if expected_success and temp_expected_output.exists():
                    # Count expected errors (lines that aren't comments)
                    with open(temp_expected_output, 'r') as f:
                        expected_count = sum(1 for line in f 
                                           if line.strip() and not line.strip().startswith('#'))
                    if expected_count > 0:
                        logger.info(f"✅ Expected error parsing found {expected_count} expected errors")
                        temp_files.append(temp_expected_output)
                    else:
                        logger.info(f"⚠️ Expected error parsing found 0 errors")
                        temp_expected_output.unlink(missing_ok=True)
                else:
                    logger.warning(f"⚠️ Expected error parsing failed")
            except Exception as e:
                logger.warning(f"⚠️ Expected error parsing error: {e}")
                temp_expected_output.unlink(missing_ok=True)
            
            # Merge results if we have any
            if temp_files:
                logger.info(f"🔄 Merging warnings from {len(temp_files)} extraction method(s)...")
                merged_count = merge_warning_files(temp_files, output_file, checker_name)
                warning_count = merged_count
                logger.info(f"✅ Merged {merged_count} unique warnings")
                
                # Clean up temp files
                for temp_file in temp_files:
                    temp_file.unlink(missing_ok=True)
            else:
                logger.warning(f"⚠️ All extraction methods failed or produced 0 warnings")
                logger.warning(f"   File generated but may not be useful for training.")
        
        elif warning_count == 0:
            logger.warning(f"⚠️ No warnings found in {output_file_name}")
            logger.warning(f"   This may indicate the test suite is fully annotated.")
            logger.warning(f"   File generated but may not be useful for training.")
        else:
            logger.info(f"✅ Generated {output_file_name} with {warning_count} warnings")
        
        # Report file statistics
        if output_file.exists():
            file_size = output_file.stat().st_size
            with open(output_file, 'r') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"📊 File statistics: {file_size:,} bytes, {line_count:,} lines, {warning_count} warnings")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error running {checker_display_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def generate_all_warning_files(output_dir: Path = GEN_DATA_ROOT, 
                               checkers: Optional[list] = None) -> Dict[str, Optional[Path]]:
    """
    Generate warning files for all GenDATA checkers.
    
    Args:
        output_dir: Directory to save warning files
        checkers: List of checker names to generate (default: all GenDATA checkers)
        
    Returns:
        Dictionary mapping checker names to output file paths (None if failed)
    """
    if checkers is None:
        checkers = GENDATA_CHECKERS
    
    results = {}
    
    logger.info("=" * 80)
    logger.info("Generating Warning Files for GenDATA Checkers")
    logger.info("=" * 80)
    logger.info(f"Checkers: {', '.join(checkers)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    for checker_name in checkers:
        output_file = generate_warning_file(checker_name, output_dir, use_extraction_methods=True)
        results[checker_name] = output_file
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("Generation Summary")
    logger.info("=" * 80)
    
    success_count = sum(1 for f in results.values() if f is not None)
    total_count = len(results)
    
    for checker_name, output_file in results.items():
        config = get_checker_config(checker_name)
        checker_display_name = config.get('name', checker_name) if config else checker_name
        
        if output_file:
            status = "✅ Generated"
            file_name = output_file.name
        else:
            status = "❌ Failed/Missing"
            file_name = WARNING_FILE_NAMES.get(checker_name, 'N/A')
        
        logger.info(f"{checker_display_name:30s} {status:20s} {file_name}")
    
    logger.info("")
    logger.info(f"Successfully generated: {success_count}/{total_count} warning files")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate warning files for GenDATA checkers from Checker Framework test suites'
    )
    parser.add_argument(
        '--checker',
        choices=GENDATA_CHECKERS,
        help='Generate warning file for specific checker only'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=GEN_DATA_ROOT,
        help='Directory to save warning files (default: /home/ubuntu/GenDATA)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip generation if warning file already exists'
    )
    parser.add_argument(
        '--no-extraction-methods',
        action='store_true',
        help='Disable annotation removal and error comment parsing (use direct checker run only)'
    )
    
    args = parser.parse_args()
    
    # Determine which checkers to process
    checkers_to_process = [args.checker] if args.checker else GENDATA_CHECKERS
    
    # Check for existing files if skip flag is set
    if args.skip_existing:
        filtered_checkers = []
        for checker_name in checkers_to_process:
            output_file_name = WARNING_FILE_NAMES.get(checker_name)
            if output_file_name:
                output_file = args.output_dir / output_file_name
                if output_file.exists():
                    logger.info(f"⏭️  Skipping {checker_name}: {output_file_name} already exists")
                    continue
            filtered_checkers.append(checker_name)
        checkers_to_process = filtered_checkers
    
    if not checkers_to_process:
        logger.info("All warning files already exist. Use without --skip-existing to regenerate.")
        return
    
    # Generate warning files
    use_extraction = not args.no_extraction_methods
    results = {}
    for checker_name in checkers_to_process:
        output_file = generate_warning_file(
            checker_name=checker_name,
            output_dir=args.output_dir,
            use_extraction_methods=use_extraction
        )
        results[checker_name] = output_file
    
    # Exit with appropriate code
    if all(f is not None for f in results.values()):
        logger.info("✅ All requested warning files generated successfully")
        sys.exit(0)
    elif any(f is not None for f in results.values()):
        logger.warning("⚠️ Some warning files failed to generate")
        sys.exit(1)
    else:
        logger.error("❌ All warning file generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

