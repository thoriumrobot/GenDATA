# Semantic Augmentation Evaluation System Guide

## Overview

The Semantic Augmentation Evaluation System provides comprehensive testing and analysis of the 27 semantic augmentation methods (17 Enhanced + 10 Simple) used in the GenDATA pipeline. This system evaluates which augmentations apply to Checker Framework test cases, measures their individual contributions to model performance, and verifies that each augmentation behaves as intended.

## 🎯 **Evaluation Components**

### **1. SemanticAugmentationEvaluator**
**File**: `semantic_augmentation_evaluator.py`

**Purpose**: Analyzes which augmentations apply to Checker Framework test cases and measures their effectiveness.

**Key Features**:
- **Coverage Analysis**: Determines which augmentations apply to each test case
- **Complexity Analysis**: Automatically selects Enhanced vs Simple augmentation systems
- **Transformation Effectiveness**: Measures success rates for each augmentation method
- **System Selection Statistics**: Tracks usage of Enhanced vs Simple systems

### **2. AblationStudyPipeline**
**File**: `ablation_study_pipeline.py`

**Purpose**: Systematically tests individual augmentation contributions by training models with and without specific augmentations.

**Key Features**:
- **Individual Method Testing**: Tests each of the 27 augmentation methods separately
- **F1 Score Measurement**: Measures performance impact of each augmentation
- **Statistical Significance Testing**: Determines if performance differences are significant
- **Confidence Intervals**: Provides statistical confidence in results

### **3. SemanticAugmentationTestSuite**
**File**: `semantic_augmentation_test_suite.py`

**Purpose**: Comprehensive test suite to verify that each augmentation method behaves correctly.

**Key Features**:
- **Behavior Verification**: Tests that augmentations apply as expected
- **Pattern Matching**: Validates that expected transformations occur
- **Semantic Equivalence**: Ensures transformations preserve semantics
- **Compilation Validation**: Verifies transformed code remains valid Java

### **4. EvaluationRunner**
**File**: `run_semantic_augmentation_evaluation.py`

**Purpose**: Orchestrates the complete evaluation process and generates comprehensive reports.

**Key Features**:
- **Complete Evaluation**: Runs all evaluation components
- **Focused Analysis**: Allows running individual components
- **Report Generation**: Creates detailed evaluation reports
- **Results Aggregation**: Combines results from all components

## 🚀 **Usage Examples**

### **Complete Evaluation**
Run all evaluation components:
```bash
cd /home/ubuntu/GenDATA
python run_semantic_augmentation_evaluation.py --run_all
```

### **Checker Framework Coverage Analysis Only**
Analyze which augmentations apply to Checker Framework test cases:
```bash
python run_semantic_augmentation_evaluation.py --run_coverage
```

### **Ablation Studies Only**
Test individual augmentation contributions:
```bash
python run_semantic_augmentation_evaluation.py --run_ablation
```

### **Test Cases Only**
Verify augmentation behavior:
```bash
python run_semantic_augmentation_evaluation.py --run_tests
```

### **Custom Configuration**
```bash
python run_semantic_augmentation_evaluation.py \
    --checker_framework_dir /path/to/checker/framework/tests/ \
    --output_dir /path/to/results/ \
    --run_all
```

## 📊 **Evaluation Results**

### **Checker Framework Coverage Analysis**
**Output**: `evaluation_results/analysis_results/checker_framework_coverage.json`

**Metrics**:
- **Total Files Analyzed**: Number of Java files processed
- **Enhanced System Usage**: Files using enhanced augmentation (17 methods)
- **Simple System Usage**: Files using simple augmentation (10 methods)
- **Transformation Coverage**: Percentage of files each augmentation applies to
- **Transformation Effectiveness**: Success rates for each augmentation method

**Example Results**:
```json
{
  "total_files": 150,
  "enhanced_files": 45,
  "simple_files": 105,
  "transformation_coverage": {
    "simple_method_calls": {
      "count": 120,
      "percentage": 80.0
    },
    "simple_assignments": {
      "count": 135,
      "percentage": 90.0
    }
  }
}
```

### **Ablation Study Results**
**Output**: `evaluation_results/ablation_studies/ablation_results.json`

**Metrics**:
- **F1 Score Differences**: Performance impact of each augmentation
- **Statistical Significance**: Whether differences are statistically meaningful
- **Confidence Intervals**: Statistical confidence in results
- **Training Times**: Time cost of each augmentation method

**Example Results**:
```json
[
  {
    "augmentation_name": "simple_method_calls",
    "baseline_f1": 0.85,
    "ablated_f1": 0.83,
    "f1_difference": 0.02,
    "confidence_interval": [0.015, 0.025],
    "statistical_significance": true
  }
]
```

### **Test Case Results**
**Output**: `evaluation_results/test_results/semantic_augmentation_test_results.json`

**Metrics**:
- **Total Tests**: Number of test cases executed
- **Passed Tests**: Tests that passed all checks
- **Failed Tests**: Tests that failed behavior verification
- **Success Rate**: Overall test success percentage

**Example Results**:
```json
{
  "total_tests": 27,
  "failures": 2,
  "errors": 0,
  "success_rate": 0.93
}
```

## 📈 **Generated Reports**

### **Comprehensive Evaluation Report**
**File**: `evaluation_results/comprehensive_evaluation_report.md`

**Contents**:
- **Executive Summary**: High-level findings and metrics
- **Most Impactful Augmentations**: Top-performing augmentation methods
- **Transformation Coverage Analysis**: Which augmentations are most applicable
- **Recommendations**: Actionable insights based on results

### **Individual Component Reports**
- **Coverage Report**: `analysis_results/checker_framework_coverage.json`
- **Ablation Report**: `ablation_studies/ablation_study_report.md`
- **Test Report**: `test_results/semantic_augmentation_test_results.json`

## 🔧 **Configuration Options**

### **SemanticAugmentationEvaluator**
```python
evaluator = SemanticAugmentationEvaluator(
    checker_framework_tests_dir='/path/to/tests/',
    output_dir='/path/to/output/'
)
```

### **AblationStudyPipeline**
```python
pipeline = AblationStudyPipeline(
    cfwr_root='/path/to/cfwr/',
    output_dir='/path/to/output/'
)
```

### **EvaluationRunner**
```bash
python run_semantic_augmentation_evaluation.py \
    --checker_framework_dir /path/to/tests/ \
    --cfwr_root /path/to/cfwr/ \
    --output_dir /path/to/output/ \
    --run_all
```

## 🎯 **Key Metrics and Interpretations**

### **Transformation Coverage**
- **High Coverage (>70%)**: Augmentation applies to most test cases
- **Medium Coverage (30-70%)**: Augmentation applies to moderate number of test cases
- **Low Coverage (<30%)**: Augmentation applies to few test cases

### **F1 Score Differences**
- **Positive (>0.01)**: Augmentation improves model performance
- **Near Zero (-0.01 to 0.01)**: Augmentation has minimal impact
- **Negative (<-0.01)**: Augmentation degrades model performance

### **Statistical Significance**
- **Significant**: Performance difference is statistically meaningful (p < 0.05)
- **Not Significant**: Performance difference could be due to chance

### **Test Case Success Rate**
- **High (>90%)**: Augmentations behave correctly most of the time
- **Medium (70-90%)**: Augmentations have some issues
- **Low (<70%)**: Augmentations have significant behavioral problems

## 📋 **Best Practices**

### **Running Evaluations**
1. **Start with Coverage Analysis**: Understand which augmentations apply to your data
2. **Run Ablation Studies**: Identify most impactful augmentation methods
3. **Verify with Test Cases**: Ensure augmentations behave correctly
4. **Generate Comprehensive Report**: Get actionable insights

### **Interpreting Results**
1. **Focus on Significant Results**: Pay attention to statistically significant findings
2. **Consider Coverage**: High-impact augmentations with low coverage may be less important
3. **Balance Performance vs. Coverage**: Optimize for augmentations with both high impact and high coverage
4. **Monitor Test Success Rates**: Ensure augmentations don't break semantic equivalence

### **Optimization Recommendations**
1. **High Impact + High Coverage**: Prioritize these augmentations
2. **High Impact + Low Coverage**: Consider expanding applicability
3. **Low Impact + High Coverage**: Consider optimizing or removing
4. **Low Impact + Low Coverage**: Candidates for removal

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Ensure you're in the GenDATA directory
cd /home/ubuntu/GenDATA
python run_semantic_augmentation_evaluation.py --run_all
```

#### **File Not Found Errors**
```bash
# Check that Checker Framework tests exist
ls /home/ubuntu/checker-framework/checker/tests/index/
```

#### **Permission Errors**
```bash
# Ensure output directory is writable
chmod 755 /home/ubuntu/GenDATA/evaluation_results/
```

### **Performance Issues**

#### **Slow Ablation Studies**
- Reduce training episodes in `ablation_study_pipeline.py`
- Use smaller evaluation datasets
- Run ablation studies in parallel

#### **Memory Issues**
- Process files in smaller batches
- Use temporary files for large datasets
- Monitor system memory usage

## 📚 **Advanced Usage**

### **Custom Test Cases**
Add custom test cases to `semantic_augmentation_test_suite.py`:
```python
custom_test_case = TestCase(
    name="custom_transformation",
    description="Test custom transformation",
    input_code="public class Test { ... }",
    expected_patterns=["pattern1", "pattern2"],
    should_transform=True,
    semantic_equivalence_checks=["check1", "check2"],
    system_type="enhanced"
)
```

### **Custom Complexity Analysis**
Modify complexity indicators in `semantic_augmentation_evaluator.py`:
```python
complexity_indicators = [
    'for (', 'while (', 'stream()', 'lambda', '->',
    'custom_pattern',  # Add custom patterns
    # ... existing indicators
]
```

### **Custom Ablation Studies**
Run ablation studies for specific methods:
```python
pipeline = AblationStudyPipeline(cfwr_root, output_dir)
results = pipeline.run_ablation_studies(['method1', 'method2'])
```

## 🎉 **Conclusion**

The Semantic Augmentation Evaluation System provides comprehensive analysis of the 27 semantic augmentation methods used in the GenDATA pipeline. By understanding which augmentations apply to different code types, measuring their individual contributions to model performance, and verifying their correct behavior, you can optimize the augmentation system for maximum effectiveness.

This evaluation system ensures that the semantic augmentation methods are:
- **Applicable**: Work on the target code types
- **Effective**: Improve model performance
- **Correct**: Preserve semantic equivalence
- **Reliable**: Behave consistently across different inputs

Use this system to continuously monitor and improve the semantic augmentation pipeline for optimal annotation placement model performance.
