# Annotation Injection Training Guide

This document describes the annotation injection approach for training models to place Checker Framework type annotations.

## Overview

The key constraint is that **models can only ADD annotations, not remove them**. This means warnings must be created in a way that can be fixed by adding more annotations.

## Pattern: Entry Point Annotation

The pattern works by:
1. Adding annotations to "entry point" method parameters (API boundaries)
2. Leaving internal variables, fields, and return types unannotated
3. When unannotated code flows into annotated parameters, warnings are generated
4. The model learns to add annotations to the unannotated locations to fix warnings

```
┌─────────────────────────────────┐
│ Entry Point (Annotated)         │
│ void execute(@Annotated sql)    │
└─────────────────────────────────┘
         ▲
         │ WARNING: incompatible argument
         │
┌─────────────────────────────────┐
│ Internal Code (Unannotated)     │
│ String query = "SELECT...";     │◄── Model adds annotation here
│ execute(query);                 │
└─────────────────────────────────┘
```

## Training Projects

### SQL Quotes Checker

**Project: commons-dbutils**
- Location: `case_studies/commons-dbutils/`
- Annotations added: 102 method parameters with `@SqlEvenQuotes`
- Files modified: `QueryRunner.java`, `AsyncQueryRunner.java`, `AbstractQueryRunner.java`
- Warnings generated: 3 checker-specific errors

**Training Examples: training_sql_quotes/**
- `SqlQuotesTrainingExample.java` - Full training example with warnings
- `SqlQuotesBeforeAnnotation.java` - State BEFORE annotation (2 warnings)
- `SqlQuotesVerification.java` - State AFTER annotation (0 warnings)

**CF Test Suite: cf_sqlquotes_tests/**
- 34 checker warnings from official Checker Framework tests

### Signature String Checker

**Project: kryo**
- Location: `case_studies/kryo/`
- Annotations added: 4 method parameters with `@BinaryName`
- Files modified: `Kryo.java`, `DefaultClassResolver.java`, `Util.java`

**Project: guice**
- Location: `case_studies/guice/`
- Annotations added: 3 method parameters with `@BinaryName`
- Files modified: `CheckedProviders.java`, `SourceProvider.java`, `ClassBuilding.java`

**Training Examples: training_signature/**
- `SignatureTrainingExample.java` - Full training example with warnings
- `SignatureBeforeAnnotation.java` - State BEFORE annotation (2 warnings)
- `SignatureVerification.java` - State AFTER annotation (0 warnings)

**CF Test Suite: cf_signature_tests/**
- 358+ checker warnings from official Checker Framework tests

## Verification Results

### Manual Annotation (Theoretical Maximum)

| Checker | File | Before | After | Reduction |
|---------|------|--------|-------|-----------|
| SQL Quotes | SqlQuotesBeforeAnnotation.java | 2 | 0 | 100% |
| Signature String | SignatureBeforeAnnotation.java | 2 | 0 | 100% |

### Automated Pipeline (Warning-Targeted Placement)

| Checker | Project | Baseline | Annotations | After | Reduction |
|---------|---------|----------|-------------|-------|-----------|
| SQL Quotes | training_sql_quotes | 6 | 2 | 4 | **33.3%** |
| Signature String | training_signature | 8 | 3 | 5 | **37.5%** |

The difference between manual and automated results reflects:
1. **Declaration line finding**: Not all warning lines have directly annotatable declarations
2. **Method return types**: Some warnings require annotating return types (more complex)
3. **Compound expressions**: String concatenation results can't be directly annotated

## How the Model Learns

1. **Input**: Code with warnings (e.g., `SqlQuotesBeforeAnnotation.java`)
2. **Target**: Locations where annotations should be added
3. **Output**: Annotation placement decisions (field, return type, local variable)
4. **Reward**: Reduction in checker warnings

The model learns to:
- Identify unannotated variables that flow into annotated parameters
- Determine the correct annotation type based on the target parameter
- Place annotations on fields, local variables, and method return types

## Running the Checker

```bash
# SQL Quotes Checker
/home/ubuntu/checker-framework/checker/bin/javac \
  -processor org.checkerframework.checker.sqlquotes.SqlQuotesChecker \
  YourFile.java

# Signature String Checker  
/home/ubuntu/checker-framework/checker/bin/javac \
  -processor org.checkerframework.checker.signature.SignatureChecker \
  YourFile.java
```

## Placement Pipeline

### Pipeline Runner

The `run_placement_pipeline.py` script orchestrates the full annotation placement pipeline:

```bash
# Run full pipeline (training + evaluation)
python run_placement_pipeline.py --all

# Run in background
nohup python run_placement_pipeline.py --all > pipeline.log 2>&1 &

# Run specific phases
python run_placement_pipeline.py --train      # Training phase only
python run_placement_pipeline.py --evaluate   # Evaluation phase only

# Run for specific checker
python run_placement_pipeline.py --checker sql_quotes
python run_placement_pipeline.py --checker signature_string
```

### Warning-Targeted Placement Flow

The evaluation pipeline uses **warning-targeted placement**:

```
┌──────────────────────────────────────────────────────────────┐
│  1. Restore from Backup (get fresh unannotated state)        │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Count Baseline Warnings (run checker, count errors)      │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Parse Warning Locations (extract file:line from output)  │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Place Annotations at Warnings (target specific lines)    │
└─────────────────────────────┬────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Count After Warnings (measure reduction)                 │
└──────────────────────────────────────────────────────────────┘
```

### Pipeline Results (January 2026)

Results are saved to `placement_pipeline_results.json`:

#### Training Phase
| Checker | Source | Java Files | Warnings |
|---------|--------|------------|----------|
| SQL Quotes | cf_sqlquotes_tests | 2 | 34 |
| Signature String | cf_signature_tests | 14 | 100 |

#### Evaluation Phase (Warning-Targeted Placement)
| Checker | Project | Baseline | Annotations | After | Reduction |
|---------|---------|----------|-------------|-------|-----------|
| SQL Quotes | training_sql_quotes | 6 | 2 | 4 | **33.3%** |
| Signature String | training_signature | 8 | 3 | 5 | **37.5%** |

### Key Functions in Pipeline

| Function | Purpose |
|----------|---------|
| `clone_fresh_project()` | Clone from GitHub (for future use) |
| `restore_from_backup()` | Restore local project from backup |
| `parse_warnings()` | Extract file:line locations from checker output |
| `find_declaration_line()` | Find the variable declaration for a warning |
| `place_annotations_at_warnings()` | Place annotations at parsed warning locations |
| `run_checker()` | Run Checker Framework and count warnings |

## Backup Systems

### Annotated Projects Backup

```bash
# Backup annotated projects only
python backup_annotated_projects.py

# List projects without backing up
python backup_annotated_projects.py --list
```

Creates `annotated_projects_backup/` with structure:
```
annotated_projects_backup/
├── sql_quotes/
│   ├── commons-dbutils/
│   ├── training_sql_quotes/
│   └── cf_sqlquotes_tests/
└── signature_string/
    ├── kryo/
    ├── guice/
    ├── training_signature/
    └── cf_signature_tests/
```

### Full Case Studies Backup

```bash
# Backup all case study projects
python backup_case_studies.py

# List projects without backing up
python backup_case_studies.py --list
```

Creates `case_studies_backup/` with all 16 projects.

## Test Suite

### Placement Tests

```bash
# Run all placement tests
pytest tests/test_sql_quotes_placement.py tests/test_signature_placement.py -v

# Run SQL Quotes tests only
pytest tests/test_sql_quotes_placement.py -v

# Run Signature String tests only
pytest tests/test_signature_placement.py -v
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_sql_quotes_placement.py` | 14 | Enum, imports, placer methods, integration |
| `test_signature_placement.py` | 21 | Enum, imports, format detection, context inference |

## Placement Systems

### SQL Quotes Placement

```python
from place_sql_quotes_annotations import SqlQuotesAnnotationPlacer, SqlQuotesAnnotationType

placer = SqlQuotesAnnotationPlacer('YourFile.java')
placements = placer.analyze_and_place()
placer.add_imports()
placer.save_file()
```

Annotations:
- `@SqlEvenQuotes` - String with even number of single quotes
- `@SqlOddQuotes` - String with odd number of single quotes

### Signature String Placement

```python
from place_signature_annotations import SignatureAnnotationPlacer, SignatureAnnotationType

placer = SignatureAnnotationPlacer('YourFile.java')
placements = placer.analyze_and_place()
placer.add_imports(annotations_used)
placer.save_file()
```

Annotations:
- `@BinaryName` - Binary class name (e.g., `java.util.Map$Entry`)
- `@FullyQualifiedName` - Fully qualified name (e.g., `java.lang.String`)
- `@FieldDescriptor` - Field descriptor (e.g., `Ljava/lang/String;`)
- `@ClassGetName` - Result of `Class.getName()`
- `@InternalForm` - Internal form (e.g., `java/lang/String`)

## Files Created

### Core Scripts
- `annotate_project_for_training.py` - Script to inject annotations into projects
- `annotation_patterns.json` - Configuration for annotation injection patterns
- `run_placement_pipeline.py` - Full pipeline runner

### Placement Systems
- `place_sql_quotes_annotations.py` - SQL Quotes annotation placement
- `place_signature_annotations.py` - Signature String annotation placement

### Backup Scripts
- `backup_annotated_projects.py` - Backup annotated projects
- `backup_case_studies.py` - Backup all case studies

### Test Files
- `tests/test_sql_quotes_placement.py` - SQL Quotes placement tests
- `tests/test_signature_placement.py` - Signature String placement tests
- `tests/fixtures/sql_quotes_sample.java` - Test fixture
- `tests/fixtures/signature_sample.java` - Test fixture

### Training Examples
- `case_studies/training_sql_quotes/` - SQL Quotes training examples
- `case_studies/training_signature/` - Signature String training examples

## Related Documentation

- [PROGRAM_ANALYSIS_WARNING_REDUCTION.md](PROGRAM_ANALYSIS_WARNING_REDUCTION.md) - Detailed analysis of how annotations reduce warnings
- [BALANCED_TRAINING_ALL_CHECKERS_DOCUMENTATION.md](BALANCED_TRAINING_ALL_CHECKERS_DOCUMENTATION.md) - Multi-checker training details
