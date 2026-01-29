#!/usr/bin/env python3
"""
Checker Crash Detector Module

Detects crashes, fatal errors, and verifies successful checker execution.
Distinguishes between compilation errors (acceptable) and crashes (not acceptable).
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Fatal crash indicators - these indicate the JVM or checker crashed
CRASH_INDICATORS = [
    'Exception in thread',
    'java.lang.Error',
    'java.lang.VirtualMachineError',
    'java.lang.OutOfMemoryError',
    'java.lang.StackOverflowError',
    'java.lang.InternalError',
    'java.lang.UnknownError',
    'java.lang.NoClassDefFoundError',
    'java.lang.ClassNotFoundException',
    'java.lang.LinkageError',
    'java.lang.ExceptionInInitializerError',
    'java.lang.NoSuchMethodError',
    'java.lang.NoSuchFieldError',
    'java.lang.IllegalAccessError',
    'java.lang.InstantiationError',
    'java.lang.AbstractMethodError',
    'java.lang.UnsupportedClassVersionError',
    'java.lang.VerifyError',
    'FATAL ERROR',
    'Internal compiler error',
    'An exception has occurred in the compiler',
    'Compilation failed: internal java compiler error',
    'at org.checkerframework',  # Stack trace in CF code
    'at com.sun.tools',  # Stack trace in compiler
    'at jdk.compiler',  # Stack trace in compiler (JDK 9+)
    # FIXED: Add file not found indicators - javac couldn't process files
    'error: file not found:',  # javac can't find source files
    'file not found:',  # Alternative format
]

# Indicators that javac failed to process any files (no real analysis happened)
NO_FILES_PROCESSED_INDICATORS = [
    'Usage: javac',  # javac printed usage = no files processed
    'use --help for a list of possible options',  # javac help text
    'no source files',  # No source files provided
]

# These indicate the checker crashed specifically (not compilation issues)
CHECKER_CRASH_INDICATORS = [
    'BugInCF',
    'at org.checkerframework.*Exception',
    'The Checker Framework crashed',
    'checker internal error',
]

# These are compilation errors, not crashes - we allow these
COMPILATION_ERROR_INDICATORS = [
    'error: cannot find symbol',
    'error: package .* does not exist',
    'error: cannot access',
    'error: incompatible types',
    'error: method .* in class .* cannot be applied',
    'error: unreported exception',
    'error: variable .* might not have been initialized',
    'error: \';\' expected',
    'error: illegal start of',
    'error: class .* is public, should be declared',
    'error: reached end of file while parsing',
    'error: unclosed',
    'error: not a statement',
    'error: missing return statement',
]

# Success indicators - at least one should be present for valid output
SUCCESS_INDICATORS = [
    r'\d+ error',           # "X errors" summary message
    r'\d+ warning',         # "X warnings" summary message
    r'Note:',               # CF notes
    r'warning:',            # Individual warnings
    r'error:',              # Individual errors (compilation, not crash)
    r'\[checking\]',        # CF processing message
    r'round \d+',           # Annotation processing rounds
]

# Stack trace pattern
STACK_TRACE_PATTERN = re.compile(r'^\s+at\s+[\w.$]+\([\w.]+:\d+\)', re.MULTILINE)


@dataclass
class CrashDetectionResult:
    """Result of crash detection analysis"""
    crashed: bool
    crash_reason: Optional[str]
    crash_indicators_found: List[str]
    has_stack_trace: bool
    has_compilation_errors: bool
    has_success_indicators: bool
    confidence: float  # 0.0 to 1.0, how confident we are in the assessment
    no_files_processed: bool = False  # True if javac didn't process any files
    compilation_error_count: int = 0  # Number of compilation errors
    
    def __str__(self) -> str:
        if self.crashed:
            return f"CRASHED: {self.crash_reason} (confidence: {self.confidence:.2f})"
        elif self.has_success_indicators:
            return f"SUCCESS: Checker ran normally (confidence: {self.confidence:.2f})"
        else:
            return f"UNCERTAIN: No clear indicators (confidence: {self.confidence:.2f})"
    
    def is_valid_result(self) -> bool:
        """Check if this result represents a valid checker run (not crashed, files processed)"""
        return not self.crashed and not self.no_files_processed
    
    def checker_analysis_succeeded(self) -> bool:
        """
        Check if the checker actually performed analysis.
        
        Returns False if:
        - Checker crashed
        - No files were processed
        - Compilation failed with errors but no checker analysis was done
        """
        if self.crashed or self.no_files_processed:
            return False
        
        # If there are compilation errors but no success indicators (like checker warnings),
        # then the checker likely never ran its analysis
        if self.has_compilation_errors and not self.has_success_indicators:
            return False
        
        return True


@dataclass 
class ProcessingVerificationResult:
    """Result of verifying checker actually processed files"""
    files_processed: bool
    warnings_parsed: bool
    warning_count: int
    error_count: int
    files_mentioned: int
    confidence: float


def detect_checker_crash(output: str, returncode: Optional[int] = None) -> CrashDetectionResult:
    """
    Detect if the Checker Framework crashed during execution.
    
    Args:
        output: Combined stdout/stderr from checker execution
        returncode: Optional return code from the process (0 = success)
        
    Returns:
        CrashDetectionResult with crash detection details
    """
    if not output:
        # Empty output handling depends on return code
        if returncode == 0:
            # Return code 0 with empty output is valid success (no warnings/errors)
            return CrashDetectionResult(
                crashed=False,
                crash_reason=None,
                crash_indicators_found=[],
                has_stack_trace=False,
                has_compilation_errors=False,
                has_success_indicators=True,  # Empty output with success code is valid
                confidence=0.9
            )
        else:
            # Non-zero return with empty output is suspicious
            return CrashDetectionResult(
                crashed=True,
                crash_reason="Empty output with non-zero exit code - checker may have crashed",
                crash_indicators_found=[],
                has_stack_trace=False,
                has_compilation_errors=False,
                has_success_indicators=False,
                confidence=0.7  # Not 100% confident
            )
    
    output_lower = output.lower()
    crash_indicators_found = []
    
    # Check for fatal crash indicators
    for indicator in CRASH_INDICATORS:
        if indicator.lower() in output_lower:
            crash_indicators_found.append(indicator)
    
    # Check for CF-specific crash indicators with regex
    for pattern in CHECKER_CRASH_INDICATORS:
        if re.search(pattern, output, re.IGNORECASE):
            crash_indicators_found.append(pattern)
    
    # Check for stack traces (strong crash indicator)
    has_stack_trace = bool(STACK_TRACE_PATTERN.search(output))
    
    # Check for compilation errors (acceptable)
    has_compilation_errors = False
    for pattern in COMPILATION_ERROR_INDICATORS:
        if re.search(pattern, output, re.IGNORECASE):
            has_compilation_errors = True
            break
    
    # Check for success indicators
    has_success_indicators = False
    for pattern in SUCCESS_INDICATORS:
        if re.search(pattern, output, re.IGNORECASE):
            has_success_indicators = True
            break
    
    # Determine if crashed
    crashed = False
    crash_reason = None
    confidence = 0.5
    
    if crash_indicators_found:
        # Check if stack trace is in the context of an actual crash vs just being printed
        if has_stack_trace:
            # Stack trace with crash indicators = definite crash
            crashed = True
            crash_reason = f"Fatal error detected: {crash_indicators_found[0]}"
            confidence = 0.95
        elif any('Exception in thread' in ind for ind in crash_indicators_found):
            # Exception in thread without stack trace is still a crash
            crashed = True
            crash_reason = f"Thread exception detected: {crash_indicators_found[0]}"
            confidence = 0.9
        elif any('OutOfMemory' in ind or 'StackOverflow' in ind for ind in crash_indicators_found):
            # OOM or SOE is definitely a crash
            crashed = True
            crash_reason = f"JVM resource exhaustion: {crash_indicators_found[0]}"
            confidence = 0.95
        elif any('Internal' in ind or 'FATAL' in ind for ind in crash_indicators_found):
            # Internal errors are crashes
            crashed = True
            crash_reason = f"Internal error: {crash_indicators_found[0]}"
            confidence = 0.9
        else:
            # Other indicators with no stack trace - might be false positive
            if has_success_indicators:
                # Has success indicators, probably not a crash
                crashed = False
                confidence = 0.7
            else:
                # No success indicators, probably crashed
                crashed = True
                crash_reason = f"Potential crash indicator: {crash_indicators_found[0]}"
                confidence = 0.6
    elif has_stack_trace and not has_success_indicators:
        # Stack trace without clear crash indicators but no success either
        crashed = True
        crash_reason = "Stack trace detected without normal checker output"
        confidence = 0.7
    elif has_success_indicators:
        # Clear success indicators, no crash
        crashed = False
        confidence = 0.9
    elif has_compilation_errors:
        # Compilation errors but no crash indicators
        crashed = False
        confidence = 0.8
    else:
        # No clear indicators either way
        # Check output length - very short output might indicate early crash
        if len(output) < 50:
            crashed = True
            crash_reason = "Very short output - checker may have failed early"
            confidence = 0.5
        else:
            crashed = False
            confidence = 0.5
    
    # FIXED: Check for "no files processed" indicators
    for indicator in NO_FILES_PROCESSED_INDICATORS:
        if indicator.lower() in output_lower:
            crashed = True
            crash_reason = f"No files were processed: {indicator}"
            confidence = 0.95
            crash_indicators_found.append(indicator)
            break
    
    # FIXED: Check for file not found errors (javac couldn't find source files)
    if 'file not found:' in output_lower:
        crashed = True
        crash_reason = "Source files not found - checker could not analyze any files"
        confidence = 0.95
    
    # FIXED: Check for high error count with 0 checker warnings
    # This indicates compilation failed and checker never ran
    error_summary_match = re.search(r'(\d+)\s+error', output)
    warning_summary_match = re.search(r'(\d+)\s+warning', output)
    
    # Track no files processed and error count
    no_files_processed = False
    compilation_error_count = 0
    
    if error_summary_match:
        compilation_error_count = int(error_summary_match.group(1))
        # Track that there were compilation errors (but don't treat as crash)
        # Compilation errors are a separate issue from crashes
        # The evaluation code will check compilation_error_count and handle appropriately
        if compilation_error_count > 0:
            has_compilation_errors = True
    
    # Check for no files processed indicators
    for indicator in NO_FILES_PROCESSED_INDICATORS:
        if indicator.lower() in output_lower:
            no_files_processed = True
            break
    
    if 'file not found:' in output_lower:
        no_files_processed = True
    
    return CrashDetectionResult(
        crashed=crashed,
        crash_reason=crash_reason,
        crash_indicators_found=crash_indicators_found,
        has_stack_trace=has_stack_trace,
        has_compilation_errors=has_compilation_errors,
        has_success_indicators=has_success_indicators,
        confidence=confidence,
        no_files_processed=no_files_processed,
        compilation_error_count=compilation_error_count
    )


def verify_checker_processed_files(output: str, expected_files: int = 0) -> ProcessingVerificationResult:
    """
    Verify that the checker actually processed files (vs crashing early).
    
    Args:
        output: Checker output string
        expected_files: Expected number of files (0 if unknown)
        
    Returns:
        ProcessingVerificationResult with processing details
    """
    if not output:
        return ProcessingVerificationResult(
            files_processed=False,
            warnings_parsed=False,
            warning_count=0,
            error_count=0,
            files_mentioned=0,
            confidence=0.3
        )
    
    # Count files mentioned in output (as .java files)
    file_pattern = re.compile(r'[\w/\\]+\.java', re.IGNORECASE)
    files_mentioned = len(set(file_pattern.findall(output)))
    
    # Count warnings
    warning_pattern = re.compile(r'\.java:\d+.*?warning:', re.IGNORECASE)
    warnings = warning_pattern.findall(output)
    warning_count = len(warnings)
    
    # Count errors (compilation errors, not crashes)
    error_pattern = re.compile(r'\.java:\d+.*?error:', re.IGNORECASE)
    errors = error_pattern.findall(output)
    error_count = len(errors)
    
    # Check for summary line
    summary_pattern = re.compile(r'(\d+)\s+(?:error|warning)', re.IGNORECASE)
    summary_matches = summary_pattern.findall(output)
    
    # Determine if files were processed
    files_processed = files_mentioned > 0 or warning_count > 0 or error_count > 0
    warnings_parsed = warning_count > 0 or bool(summary_matches)
    
    # Calculate confidence
    if files_processed and (warning_count > 0 or error_count > 0):
        confidence = 0.9
    elif files_mentioned > 0:
        confidence = 0.7
    elif bool(summary_matches):
        confidence = 0.6
    else:
        confidence = 0.3
    
    return ProcessingVerificationResult(
        files_processed=files_processed,
        warnings_parsed=warnings_parsed,
        warning_count=warning_count,
        error_count=error_count,
        files_mentioned=files_mentioned,
        confidence=confidence
    )


def has_valid_checker_output(output: str) -> bool:
    """
    Quick check if output looks like valid checker output.
    
    Args:
        output: Checker output string
        
    Returns:
        True if output appears to be valid checker output
    """
    if not output:
        return False
    
    # Check for any success indicators
    for pattern in SUCCESS_INDICATORS:
        if re.search(pattern, output, re.IGNORECASE):
            return True
    
    # Check for file references (indicates processing happened)
    if re.search(r'\.java:\d+', output):
        return True
    
    return False


def analyze_checker_output(output: str, expected_files: int = 0, returncode: Optional[int] = None) -> Tuple[CrashDetectionResult, ProcessingVerificationResult]:
    """
    Comprehensive analysis of checker output.
    
    Args:
        output: Checker output string
        expected_files: Expected number of files processed
        returncode: Optional return code from the process
        
    Returns:
        Tuple of (CrashDetectionResult, ProcessingVerificationResult)
    """
    crash_result = detect_checker_crash(output, returncode=returncode)
    processing_result = verify_checker_processed_files(output, expected_files)
    
    return crash_result, processing_result


# Custom exception for checker timeout
class CheckerTimeoutError(Exception):
    """Raised when checker execution times out"""
    pass


# Custom exception for checker crash
class CheckerCrashError(Exception):
    """Raised when checker crashes"""
    def __init__(self, message: str, crash_result: CrashDetectionResult):
        super().__init__(message)
        self.crash_result = crash_result


if __name__ == '__main__':
    # Test with sample outputs
    print("Testing crash detector...")
    
    # Test 1: Normal output with warnings
    normal_output = """
    /path/to/File.java:10: warning: [index] possible array index out of bounds
        arr[i] = 5;
             ^
    1 warning
    """
    result = detect_checker_crash(normal_output)
    print(f"Normal output: {result}")
    assert not result.crashed, "Should not detect crash in normal output"
    
    # Test 2: Crash with stack trace
    crash_output = """
    Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
        at java.util.Arrays.copyOf(Arrays.java:3236)
        at org.checkerframework.checker.index.IndexChecker.process(IndexChecker.java:100)
    """
    result = detect_checker_crash(crash_output)
    print(f"Crash output: {result}")
    assert result.crashed, "Should detect crash in OOM output"
    
    # Test 3: Compilation error (not crash)
    compilation_error = """
    /path/to/File.java:5: error: cannot find symbol
        UnknownClass obj = new UnknownClass();
        ^
    1 error
    """
    result = detect_checker_crash(compilation_error)
    print(f"Compilation error: {result}")
    assert not result.crashed, "Should not detect crash in compilation error"
    
    # Test 4: Empty output
    result = detect_checker_crash("")
    print(f"Empty output: {result}")
    assert result.crashed, "Should detect potential crash in empty output"
    
    print("\nAll tests passed!")
