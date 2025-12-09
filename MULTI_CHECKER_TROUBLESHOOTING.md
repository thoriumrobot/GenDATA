# Multi-Checker Infrastructure Troubleshooting Guide

## Overview

This guide helps diagnose and resolve common issues when working with GenDATA's multi-checker evaluation infrastructure.

## Common Issues and Solutions

### Issue 1: Checker Not Found in Registry

**Symptoms**:
```
Error: Checker 'X' not found in registry
```

**Causes**:
- Checker implementation not imported
- Checker not registered with `@register_checker` decorator
- Import error preventing checker module from loading

**Solutions**:
1. Verify checker implementation file exists (e.g., `lower_bound_checker.py`)
2. Check that `@register_checker` decorator is applied to checker class
3. Ensure checker module is imported in `checker_registry.py` or auto-imported
4. Run verification script: `python3 verify_multi_checker_infrastructure.py`

**Example Fix**:
```python
# In checker_registry.py, ensure import exists:
from lower_bound_checker import LowerBoundChecker
register_checker('lower_bound', LowerBoundChecker)
```

### Issue 2: Test Suite Not Found

**Symptoms**:
```
Warning: Test suite not found for checker X
```

**Causes**:
- Checker Framework installation missing test suite
- Incorrect path in configuration
- Test suite not included in Checker Framework distribution

**Solutions**:
1. Verify Checker Framework installation location
2. Check `checker_evaluation_config.py` for correct test suite path
3. For SQL Quotes Checker: Test suite may not be available in current CF version
4. Use `verify_checker_training.py` to check test suite availability

**Known Limitations**:
- SQL Quotes Checker test suite (`/home/ubuntu/checker-framework/checker/tests/quotes/`) not found in current installation
- This prevents model training but does not block infrastructure verification

### Issue 3: No Warnings Generated

**Symptoms**:
```
Status: no_warnings
Project has no checker warnings
```

**Causes**:
- Project is well-annotated (no warnings to fix)
- Project doesn't use features that trigger checker warnings
- Compilation errors preventing checker from running
- Checker not finding relevant code patterns

**Solutions**:
1. This is **normal behavior** - not all projects trigger warnings for all checkers
2. Verify project compiles successfully (check for compilation errors)
3. Check warnings file contains actual checker warnings vs compilation errors
4. Try different projects - some are more suitable for specific checkers
5. Use `identify_checker_projects.py` to find projects with relevant patterns

**Verification**:
```bash
# Check warnings file
python3 -c "from checker_framework_runner import count_checker_warnings; print(count_checker_warnings('path/to/warnings.out'))"

# Check compilation errors vs checker warnings
python3 -c "from checker_framework_runner import CheckerFrameworkRunner; runner = CheckerFrameworkRunner(); info = runner.parse_warnings_file('path/to/warnings.out'); print(f'Warnings: {info[\"total_warnings\"]}, Errors: {info[\"total_compilation_errors\"]}')"
```

### Issue 4: Models Not Available

**Symptoms**:
```
Status: no_models_available
Models not available for checker X
```

**Causes**:
- Models not yet trained for the checker
- Models in wrong location or wrong naming convention
- Model files corrupted or incomplete

**Solutions**:
1. Verify model training status: `python3 verify_checker_training.py`
2. Check models directory: `ls -la models_annotation_types/`
3. Train models if missing: Use checker-specific training scripts
4. Verify model naming matches expected pattern: `{annotation}_{model}`

**Model Training**:
```bash
# For Lower Bound Checker (already trained)
# Models should exist in models_annotation_types/

# For SQL Quotes Checker (needs training)
python3 train_sql_quotes_models.py

# For Signature String Checker (needs training)
python3 train_signature_string_models.py
```

### Issue 5: Warning Parsing Fails

**Symptoms**:
```
Error parsing warnings file
Parsed 0 warnings but file exists
```

**Causes**:
- Warning format doesn't match expected pattern
- Checker-specific parser not implemented correctly
- Generic fallback parser not matching warning format

**Solutions**:
1. Check warning file format manually: `head warnings.out`
2. Verify checker-specific parser implementation in checker class
3. Check `get_warning_patterns()` returns correct patterns
4. Test parser with sample warnings file

**Debugging**:
```python
from lower_bound_checker import LowerBoundChecker
checker = LowerBoundChecker()
warnings = checker.parse_warnings('path/to/warnings.out')
print(f"Parsed {len(warnings)} warnings")
```

### Issue 6: Configuration Errors

**Symptoms**:
```
KeyError: 'annotation_types'
Configuration not found for checker X
```

**Causes**:
- Checker not configured in `checker_evaluation_config.py`
- Missing required configuration keys
- Typo in checker name

**Solutions**:
1. Verify checker configuration exists in `CHECKER_CONFIGS`
2. Check all required keys are present: `name`, `processor`, `annotation_types`, `base_models`, `expected_models`
3. Verify checker name matches registry name (case-insensitive)

**Example Configuration**:
```python
CHECKER_CONFIGS = {
    'my_checker': {
        'name': 'My Checker',
        'processor': 'org.checkerframework.checker.mine.MyChecker',
        'test_suite': '/path/to/test/suite',
        'annotation_types': ['@Annotation1', '@Annotation2'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'],
        'expected_models': 14,
        'evaluation_projects': ['project1', 'project2'],
        'model_naming_pattern': '{annotation}_{model}',
    }
}
```

### Issue 7: Project Preparation Fails

**Symptoms**:
```
Failed to generate warnings file
Project preparation failed
```

**Causes**:
- Project doesn't compile
- Checker Framework not installed correctly
- Classpath issues
- Permission errors

**Solutions**:
1. Verify project compiles without checker: `javac -cp ... *.java`
2. Check Checker Framework installation: `ls /home/ubuntu/checker-framework-3.42.0/checker/dist/`
3. Verify classpath includes checker JARs
4. Check file permissions on project directory

**Verification**:
```bash
# Test checker execution manually
python3 -c "
from checker_framework_runner import CheckerFrameworkRunner
runner = CheckerFrameworkRunner(checker_name='lower_bound')
success = runner.run_checker_on_project('project_path', 'output.out')
print(f'Success: {success}')
"
```

### Issue 8: Evaluation Script Errors

**Symptoms**:
```
AttributeError: 'NoneType' object has no attribute 'X'
Evaluation script crashes
```

**Causes**:
- Checker interface not properly implemented
- Missing required methods
- Type mismatches in return values

**Solutions**:
1. Run verification script: `python3 verify_multi_checker_infrastructure.py`
2. Check checker implementation against `CheckerInterface`
3. Verify all abstract methods are implemented
4. Test checker instantiation: `checker = MyChecker()`

### Issue 9: Report Generation Issues

**Symptoms**:
```
Report not generated
Empty or incomplete report
```

**Causes**:
- No evaluation results to report
- File permission issues
- JSON serialization errors

**Solutions**:
1. Verify evaluation completed successfully
2. Check results directory exists and is writable
3. Verify JSON results file is valid: `python3 -m json.tool results.json`
4. Check for serialization errors in evaluation results

## Verification Commands

### Quick Verification

```bash
# Verify infrastructure
python3 verify_multi_checker_infrastructure.py

# Verify checker training status
python3 verify_checker_training.py

# Run integration tests
python3 test_checker_integration.py
```

### Checker-Specific Verification

```bash
# Test Lower Bound Checker
python3 -c "
from checker_registry import get_checker
checker = get_checker('lower_bound')
print(f'Checker: {checker.get_checker_name()}')
print(f'Processor: {checker.get_checker_processor()}')
print(f'Annotations: {checker.get_annotation_types()}')
"

# Test SQL Quotes Checker
python3 -c "
from checker_registry import get_checker
checker = get_checker('sql_quotes')
print(f'Checker: {checker.get_checker_name()}')
"

# Test Signature String Checker
python3 -c "
from checker_registry import get_checker
checker = get_checker('signature_string')
print(f'Checker: {checker.get_checker_name()}')
"
```

## Known Limitations

### 1. SQL Quotes Checker Test Suite Missing

**Status**: Test suite not found in current Checker Framework installation

**Impact**: Cannot train SQL Quotes Checker models until test suite is available

**Workaround**: Infrastructure is ready; models can be trained once test suite is available

### 2. Models Not Trained

**Status**: SQL Quotes (0/14) and Signature String (0/21) models not yet trained

**Impact**: Full evaluation cannot proceed for these checkers

**Workaround**: Evaluation can still run checker and generate warnings; prediction and metrics require trained models

### 3. Projects with No Warnings

**Status**: Some projects don't trigger warnings for certain checkers

**Impact**: Normal behavior - projects may be well-annotated or don't use relevant patterns

**Workaround**: System handles this gracefully; reports status as `no_warnings`

## Getting Help

### Check Logs

```bash
# Check evaluation logs
tail -f evaluation_output.log

# Check verification report
cat MULTI_CHECKER_VERIFICATION_REPORT.md

# Check evaluation results
cat multi_checker_results/MULTI_CHECKER_EVALUATION_REPORT.md
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verification Report

The verification report (`MULTI_CHECKER_VERIFICATION_REPORT.md`) contains detailed test results for all infrastructure components. Check this first when troubleshooting.

## Best Practices

1. **Always verify infrastructure** before running evaluations: `python3 verify_multi_checker_infrastructure.py`

2. **Check model availability** before expecting predictions: `python3 verify_checker_training.py`

3. **Validate projects** before evaluation: Ensure projects compile and have relevant code patterns

4. **Review warnings files** to distinguish compilation errors from checker warnings

5. **Use appropriate projects** for each checker: Use `identify_checker_projects.py` to find suitable projects

6. **Check evaluation status** in reports: Review `MULTI_CHECKER_EVALUATION_REPORT.md` for detailed status

## Related Documentation

- **Multi-Checker Guide**: `MULTI_CHECKER_EVALUATION_GUIDE.md` - Complete usage guide
- **Verification Report**: `MULTI_CHECKER_VERIFICATION_REPORT.md` - Infrastructure test results
- **Evaluation Results**: `multi_checker_results/MULTI_CHECKER_EVALUATION_REPORT.md` - Evaluation outcomes

