# Annotation Removal for Evaluation Guide

## Overview

When evaluating GenDATA on projects that have no checker warnings, it's useful to remove existing annotations from the project to generate warnings for evaluation purposes. This guide explains how to use the annotation removal utility to create evaluation scenarios.

## Purpose

The annotation removal utility (`studies/remove_annotations_for_evaluation.py`) removes checker-specific annotations from Java source files. This enables:

1. **Generating Warnings**: Projects with existing annotations may have zero warnings. Removing annotations creates warnings that can be used for evaluation.

2. **Testing Evaluation Pipeline**: Verify that the evaluation pipeline works correctly by testing on projects with known annotation locations.

3. **Creating Evaluation Scenarios**: Generate controlled evaluation scenarios where ground truth annotations are known.

## Usage

### Basic Usage

Remove annotations from a project:

```bash
# Remove Lower Bound Checker annotations
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/plume-lib \
    --checker lower_bound \
    --output_dir case_studies/plume-lib_no_annotations

# Remove SQL Quotes Checker annotations
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/guava \
    --checker sql_quotes \
    --output_dir case_studies/guava_no_annotations

# Remove Signature String Checker annotations
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/hipparchus \
    --checker signature_string \
    --output_dir case_studies/hipparchus_no_annotations
```

### Testing Annotation Removal

Test annotation removal on a single file:

```bash
python3 studies/remove_annotations_for_evaluation.py \
    --test \
    --project_path case_studies/plume-lib \
    --checker lower_bound \
    --test_file case_studies/plume-lib/java/src/plume/TestPlume.java
```

### In-Place Modification (with Backup)

Modify files in place, creating backups:

```bash
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/plume-lib \
    --checker lower_bound
# Original files are modified, backups created with .backup extension
```

## Supported Checkers

### Lower Bound Checker
- **Annotations Removed**: `@Positive`, `@NonNegative`, `@GTENegativeOne`
- **Usage**: `--checker lower_bound`

### SQL Quotes Checker
- **Annotations Removed**: `@SqlEvenQuotes`, `@SqlOddQuotes`
- **Usage**: `--checker sql_quotes`

### Signature String Checker
- **Annotations Removed**: `@FullyQualifiedName`, `@BinaryName`, `@FieldDescriptor`
- **Usage**: `--checker signature_string`

## How It Works

1. **Annotation Detection**: The utility scans Java files for checker-specific annotations using regex patterns.

2. **Annotation Removal**: Annotations are removed using pattern matching:
   - Standalone annotation lines: `@Positive\n` → removed entirely
   - Inline annotations: `@Positive int value` → `int value`
   - Comment-style annotations: `/*@Positive*/ int value` → `int value`

3. **File Processing**: All `.java` files in the project are processed recursively.

4. **Output**: Modified files are written to the output directory (or original location with backup).

## Integration with Evaluation Pipeline

### Step 1: Remove Annotations

```bash
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/guava \
    --checker lower_bound \
    --output_dir case_studies/guava_no_annotations
```

### Step 2: Run Checker on Modified Project

```bash
from checker_framework_runner import CheckerFrameworkRunner

runner = CheckerFrameworkRunner(checker_name='lower_bound')
warnings_file = 'case_studies/guava_no_annotations/warnings.out'
runner.run_checker_on_project('case_studies/guava_no_annotations', warnings_file)
```

### Step 3: Run Evaluation

```bash
python3 evaluate_multi_checker.py \
    --checker lower_bound \
    --projects guava_no_annotations
```

## Verification

After removing annotations, verify that:

1. **Annotations Removed**: Check that target annotations are no longer present:
   ```bash
   grep -r "@Positive\|@NonNegative\|@GTENegativeOne" case_studies/guava_no_annotations/
   # Should return no results
   ```

2. **Warnings Generated**: Run the checker and verify warnings are generated:
   ```bash
   python3 checker_framework_runner.py case_studies/guava_no_annotations warnings.out
   # Check warnings.out for checker warnings
   ```

3. **Code Still Compiles**: Ensure the modified code still compiles (annotations don't affect compilation):
   ```bash
   cd case_studies/guava_no_annotations && javac -cp ... *.java
   ```

## Limitations

1. **Comment-Style Annotations**: The utility handles both standard (`@Annotation`) and comment-style (`/*@Annotation*/`) annotations.

2. **Package Prefixes**: Annotations with package prefixes (e.g., `@org.checkerframework.common.value.qual.Positive`) are detected and removed.

3. **Multiple Annotations**: Files with multiple annotations per line are handled correctly.

4. **Formatting**: Some formatting may be affected (extra spaces), but code structure is preserved.

## Example Workflow

Complete workflow for evaluation with annotation removal:

```bash
# 1. Remove annotations
python3 studies/remove_annotations_for_evaluation.py \
    --project_path case_studies/plume-lib \
    --checker lower_bound \
    --output_dir case_studies/plume-lib_no_annotations

# 2. Prepare project (run checker, generate warnings)
python3 prepare_checker_projects.py \
    --checker lower_bound \
    --projects plume-lib_no_annotations

# 3. Run evaluation
python3 evaluate_multi_checker.py \
    --checker lower_bound \
    --projects plume-lib_no_annotations

# 4. Compare results
# Original project: 0 warnings (all annotated)
# Modified project: N warnings (annotations removed)
# Evaluation: Models predict annotations → warning reduction
```

## Notes

- **Backup Files**: When modifying in place, backup files are created with `.backup` extension. These can be used to restore original files.

- **Selective Removal**: The utility removes only annotations for the specified checker. Other annotations (e.g., `@Override`, `@Deprecated`) are preserved.

- **Project Structure**: Directory structure is preserved when using `--output_dir`.

- **Performance**: Processing is efficient for projects with hundreds of files.

## Troubleshooting

**Issue**: No annotations found
- **Solution**: Verify the project contains annotations for the specified checker. Use `grep` to search for annotations.

**Issue**: Warnings not generated after removal
- **Solution**: Ensure the project compiles correctly. Some projects may require annotations to compile (type system dependencies).

**Issue**: Code formatting issues
- **Solution**: The utility preserves code structure but may introduce minor formatting changes. Use a code formatter if needed.

## Related Files

- `studies/remove_annotations_for_evaluation.py`: Main annotation removal utility
- `studies/apply_predictions_to_files.py`: Utility for applying annotations (reverse operation)
- `checker_framework_runner.py`: Checker execution utility
- `evaluate_multi_checker.py`: Evaluation pipeline

