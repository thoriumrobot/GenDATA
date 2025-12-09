# Multi-Checker Evaluation Guide

## Overview

GenDATA's multi-checker evaluation infrastructure enables evaluation of machine learning models across multiple Checker Framework checkers using a unified, extensible architecture. This guide explains the architecture, usage, and how to extend the system for new checkers.

## Architecture

### Core Components

#### 1. Checker Interface Abstraction (`checker_interface.py`)

The `CheckerInterface` abstract base class defines the contract that all checker implementations must follow:

```python
class CheckerInterface(ABC):
    @abstractmethod
    def get_checker_name(self) -> str:
        """Return the name of the checker"""
    
    @abstractmethod
    def get_checker_processor(self) -> str:
        """Return the Checker Framework processor class name"""
    
    @abstractmethod
    def get_annotation_types(self) -> List[str]:
        """Return list of annotation types this checker supports"""
    
    @abstractmethod
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        """Parse warnings from Checker Framework output file"""
    
    @abstractmethod
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        """Extract checker-specific features from CFG node"""
    
    @abstractmethod
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        """Validate if an annotation can be placed at a given location"""
    
    @abstractmethod
    def get_training_data_source(self) -> str:
        """Return path to training data source (Checker Framework test suite directory)"""
```

#### 2. Checker Registry (`checker_registry.py`)

The registry manages checker implementations and provides factory methods:

- `register_checker(name, checker_class)`: Register a checker implementation
- `get_checker(name)`: Retrieve a checker instance by name
- `list_checkers()`: List all registered checkers
- `is_checker_registered(name)`: Check if a checker is registered

Built-in checkers are auto-registered on import:
- `lower_bound` / `lowerbound` / `index` → `LowerBoundChecker`
- `sql_quotes` / `sqlquotes` → `SqlQuotesChecker`
- `signature_string` / `signaturestring` → `SignatureStringChecker`

#### 3. Extended CheckerFrameworkRunner (`checker_framework_runner.py`)

The `CheckerFrameworkRunner` class has been extended to support dynamic checker selection:

```python
runner = CheckerFrameworkRunner(
    checker_name='lower_bound',  # Select checker by name
    checker_cp=classpath
)

# Or use processor directly
runner = CheckerFrameworkRunner(
    processor='org.checkerframework.checker.index.IndexChecker'
)
```

Features:
- Automatic checker loading from registry
- Checker-specific warning parsing
- Fallback to generic parsing if checker-specific parser unavailable
- Distinguishes compilation errors from checker warnings

#### 4. Configuration System (`checker_evaluation_config.py`)

Centralized configuration for all checkers:

```python
from checker_evaluation_config import (
    get_checker_config,
    get_evaluation_projects,
    build_model_name
)

config = get_checker_config('lower_bound')
projects = get_evaluation_projects('lower_bound')
model_name = build_model_name('lower_bound', '@Positive', 'gcn')
```

#### 5. Evaluation Scripts

- **`evaluate_multi_checker.py`**: Core evaluation script supporting multiple checkers
- **`run_multi_checker_evaluations.py`**: Main orchestrator for multi-checker evaluation
- **`prepare_checker_projects.py`**: Prepare projects for evaluation with specific checkers
- **`identify_checker_projects.py`**: Identify suitable projects for each checker

## Supported Checkers

### Lower Bound Checker

**Status**: Fully supported with 21 trained models

**Annotations**: `@Positive`, `@NonNegative`, `@GTENegativeOne`

**Processor**: `org.checkerframework.checker.index.IndexChecker`

**Test Suite**: `/home/ubuntu/checker-framework/checker/tests/index/`

**Evaluation Projects**: guava, jfreechart, plume-lib, agrona, hipparchus, eclipse-collections

### SQL Quotes Checker

**Status**: Infrastructure ready, models need training (0/14 models)

**Annotations**: `@SqlEvenQuotes`, `@SqlOddQuotes`

**Processor**: `org.checkerframework.checker.quotes.QuotesChecker`

**Test Suite**: `/home/ubuntu/checker-framework/checker/tests/quotes/` (not found)

**Evaluation Projects**: guava, hipparchus, jfreechart, agrona, eclipse-collections

**Note**: Test suite not found in current Checker Framework installation. Models need to be trained before evaluation.

### Signature String Checker

**Status**: Infrastructure ready with **internal string feature extraction**, models need training (0/21 models)

**Annotations**: `@FullyQualifiedName`, `@BinaryName`, `@FieldDescriptor`

**Processor**: `org.checkerframework.checker.signature.qual.SignatureChecker`

**Test Suite**: `/home/ubuntu/checker-framework/checker/tests/signature/`

**Evaluation Projects**: guava, hipparchus, jfreechart, agrona, eclipse-collections, plume-lib

**Note**: Test suite exists. Models need to be trained before evaluation.

## Generating Warning Files for Training

Before training models for any checker, you need to generate warning files from Checker Framework test suites. GenDATA provides a unified script to generate warning files for all supported checkers.

### Using the Warning File Generation Script

```bash
# Generate warning files for all GenDATA checkers
python3 generate_checker_warning_files.py

# Generate warning file for specific checker
python3 generate_checker_warning_files.py --checker lower_bound
python3 generate_checker_warning_files.py --checker sql_quotes
python3 generate_checker_warning_files.py --checker signature_string

# Skip generation if files already exist
python3 generate_checker_warning_files.py --skip-existing

# Specify custom output directory
python3 generate_checker_warning_files.py --output-dir /path/to/warnings
```

### Warning Files Generated

The script generates checker-specific warning files in `/home/ubuntu/GenDATA/`:

- **Lower Bound Checker**: `lower_bound_warnings.out` (or use existing `index1.out` for backward compatibility)
- **SQL Quotes Checker**: `sql_quotes_warnings.out` (only if test suite exists)
- **Signature String Checker**: `signature_string_warnings.out`

### Important Notes

1. **Only GenDATA Checkers**: The script only generates warning files for checkers that GenDATA trains models for (Lower Bound, SQL Quotes, Signature String). Other Checker Framework checkers are not included.

2. **Test Suite Requirements**: 
   - Lower Bound: Test suite exists at `/home/ubuntu/checker-framework/checker/tests/index/`
   - SQL Quotes: Test suite missing at `/home/ubuntu/checker-framework/checker/tests/quotes/` (generation will be skipped)
   - Signature String: Test suite exists at `/home/ubuntu/checker-framework/checker/tests/signature/`

3. **Validation**: The script validates that actual checker warnings (not just compilation errors) are present in the generated files.

4. **Integration with Training**: Training scripts automatically check for warning files and provide clear error messages if files are missing. You can also use `train_all_checkers.py --generate-warnings` to generate warning files before training.

#### **Internal String Feature Extraction**

The Signature String Checker uses a comprehensive 30-feature extraction system that analyzes Java source code to distinguish between the three annotation types:

**Feature Categories**:
1. **Format Detection (6 features)**: Detects FullyQualifiedName (dotted), BinaryName (slashed), and FieldDescriptor (L...;) formats with confidence scores
2. **Structural Features (8 features)**: Package depth, class name length, array/method indicators, primitive/object type detection
3. **Pattern Features (6 features)**: Character-level analysis (dots, slashes, semicolons, capitalization patterns)
4. **Context Features (6 features)**: Usage patterns (Class.forName, Class.getName, reflection APIs, type conversion)
5. **CFG Context Features (4 features)**: Node types, control flow degrees, dataflow connections

**Source Code Extraction**:
- Extracts actual string values from Java source files using AST parsing (Eclipse JDT) when available
- Falls back to regex-based extraction if AST parsing unavailable
- Analyzes surrounding code context for usage patterns

**Implementation Files**:
- `signature_string_feature_extractor.py`: Core feature extraction with analyzers
- `source_code_feature_extractor.py`: Source code access and string extraction utilities
- `signature_string_checker.py`: Enhanced checker with integrated feature extraction
- `test_signature_string_features.py`: Comprehensive unit tests (16 tests, all passing)

**Training Scripts**:
- `annotation_type_rl_signature_string_fullyqualified.py`: Training for @FullyQualifiedName
- `annotation_type_rl_signature_string_binary.py`: Training for @BinaryName
- `annotation_type_rl_signature_string_fielddescriptor.py`: Training for @FieldDescriptor

All training scripts support 7 base models (GCN, HGT, GBT, Causal, Enhanced Causal, GCSN, DG2N) with 30-feature input dimensions.

## Usage Examples

### Evaluate Single Checker

```bash
# Evaluate Lower Bound Checker on specific projects
python3 evaluate_multi_checker.py \
    --checker lower_bound \
    --projects guava jfreechart
```

### Evaluate All Checkers

```bash
# Evaluate all checkers on all configured projects
python3 run_multi_checker_evaluations.py
```

### Verify Infrastructure

```bash
# Run comprehensive verification tests
python3 verify_multi_checker_infrastructure.py

# Run integration tests
python3 test_checker_integration.py
```

### Identify Suitable Projects

```bash
# Identify projects suitable for each checker
python3 identify_checker_projects.py
```

### Prepare Projects for Evaluation

```bash
# Prepare all projects for all checkers
python3 prepare_checker_projects.py
```

## Adding a New Checker

To add support for a new Checker Framework checker:

### Step 1: Implement CheckerInterface

Create a new file `{checker_name}_checker.py`:

```python
from checker_interface import CheckerInterface
from checker_registry import register_checker

@register_checker
class MyNewChecker(CheckerInterface):
    def get_checker_name(self) -> str:
        return "MyNewChecker"
    
    def get_checker_processor(self) -> str:
        return "org.checkerframework.checker.mine.MyChecker"
    
    def get_annotation_types(self) -> List[str]:
        return ['@Annotation1', '@Annotation2']
    
    def parse_warnings(self, warnings_file: str) -> List[Dict[str, Any]]:
        # Implement checker-specific warning parsing
        pass
    
    def extract_features(self, cfg_data: Dict[str, Any], node: Dict[str, Any]) -> List[float]:
        # Implement checker-specific feature extraction
        pass
    
    def validate_annotation(self, annotation_type: str, location: Dict[str, Any]) -> bool:
        # Implement annotation validation logic
        pass
    
    def get_training_data_source(self) -> str:
        return '/path/to/checker/test/suite/'
    
    def get_warning_patterns(self) -> List[str]:
        return ['pattern1', 'pattern2']
```

### Step 2: Register Checker

The `@register_checker` decorator automatically registers the checker. Alternatively, register manually in `checker_registry.py`:

```python
from my_new_checker import MyNewChecker
register_checker('my_new', MyNewChecker)
```

### Step 3: Add Configuration

Add checker configuration to `checker_evaluation_config.py`:

```python
CHECKER_CONFIGS = {
    # ... existing checkers ...
    'my_new': {
        'name': 'My New Checker',
        'processor': 'org.checkerframework.checker.mine.MyChecker',
        'test_suite': '/path/to/test/suite',
        'annotation_types': ['@Annotation1', '@Annotation2'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n'],
        'expected_models': 14,  # 7 base models × 2 annotation types
        'evaluation_projects': ['project1', 'project2'],
        'model_naming_pattern': '{annotation}_{model}',
    }
}
```

### Step 4: Create Training Scripts

Create training scripts following the pattern of existing checkers:
- `train_my_new_models.py`: Orchestrator for training all models
- `annotation_type_rl_my_new_annotation1.py`: Training script for first annotation type
- `annotation_type_rl_my_new_annotation2.py`: Training script for second annotation type

### Step 5: Verify Integration

Run verification tests:

```bash
python3 verify_multi_checker_infrastructure.py
python3 test_checker_integration.py
```

## Project Identification

The `identify_checker_projects.py` script analyzes projects to identify suitable evaluation targets:

- **SQL Quotes Checker**: Looks for SQL-related code patterns (executeQuery, PreparedStatement, string concatenation with SQL)
- **Signature String Checker**: Looks for reflection/signature patterns (Class.forName, Class.getName, MethodDescriptor)

Projects are scored based on pattern matches and marked as suitable if they exceed thresholds.

## Evaluation Workflow

### Complete Evaluation Pipeline

1. **Project Preparation**: Run checker on projects to generate warnings files
2. **Warning Validation**: Verify warnings contain actual checker warnings (not just compilation errors)
3. **Slice Generation**: Generate slices using Soot slicer based on warnings
4. **CFG Generation**: Convert slices to CFGs using Checker Framework CFG Builder
5. **Prediction Generation**: Run trained models on CFGs (if models available)
6. **Metrics Computation**: Compute precision, recall, F1, and warning reduction
7. **Report Generation**: Generate comprehensive evaluation reports

### Handling Projects with No Warnings

The system gracefully handles projects with no checker warnings:
- Reports status as `no_warnings` instead of failing
- Continues evaluation for other projects
- Includes in summary reports with appropriate status

### Handling Missing Models

When models are not available for a checker:
- Evaluation continues with warning generation and validation
- Status reported as `no_models_available`
- Reports indicate which checkers have models and which don't

## Reports

### Multi-Checker Evaluation Report

Location: `multi_checker_results/MULTI_CHECKER_EVALUATION_REPORT.md`

Contains:
- Summary statistics across all checkers
- Per-checker results with status breakdown
- Cross-checker comparison table
- Per-project evaluation status

### Verification Report

Location: `MULTI_CHECKER_VERIFICATION_REPORT.md`

Contains:
- Infrastructure verification test results
- Component compliance status
- Configuration validation results

### JSON Results

Location: `multi_checker_results/multi_checker_evaluation_results.json`

Machine-readable results in JSON format for programmatic analysis.

## Troubleshooting

### Checker Not Found

**Error**: `Checker 'X' not found in registry`

**Solution**: Ensure checker implementation is imported and registered. Check that `@register_checker` decorator is applied or manual registration is performed.

### Test Suite Not Found

**Warning**: `Test suite not found for checker X`

**Solution**: Verify test suite path in `checker_evaluation_config.py`. Check that Checker Framework is installed at expected location.

### No Warnings Generated

**Status**: `no_warnings`

**Explanation**: This is normal - projects may be well-annotated or not trigger warnings for the specific checker. The system handles this gracefully.

### Models Not Available

**Status**: `no_models_available`

**Solution**: Train models for the checker using the training scripts. Verify models exist in `models_annotation_types/` directory.

## Known Limitations

1. **SQL Quotes Checker**: Test suite not found in current Checker Framework installation. Models cannot be trained until test suite is available.

2. **Model Training**: SQL Quotes and Signature String checkers require model training before full evaluation can proceed.

3. **Warning Parsing**: Some checkers may have warning formats that require custom parsing logic beyond the generic fallback.

4. **Project Suitability**: Project identification uses pattern matching heuristics. Manual verification may be needed for edge cases.

## Future Enhancements

- Automatic model training integration
- Support for additional checkers (Null Checker, Interning Checker, etc.)
- Enhanced project identification with ML-based scoring
- Cross-checker model transfer learning
- Unified metrics comparison across checkers

## Related Documentation

- **Verification Report**: `MULTI_CHECKER_VERIFICATION_REPORT.md`
- **Evaluation Results**: `multi_checker_results/MULTI_CHECKER_EVALUATION_REPORT.md`
- **Checker Interface**: `checker_interface.py`
- **Configuration**: `checker_evaluation_config.py`

