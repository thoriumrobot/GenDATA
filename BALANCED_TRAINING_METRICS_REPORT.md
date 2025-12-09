# Balanced Training Metrics Report

This report contains training and validation metrics for all checkers trained on balanced datasets.

## Overview

| Checker | Expected Models | Trained Models | Best Accuracy (Mean) | Final Accuracy (Mean) |
|---------|----------------|----------------|----------------------|----------------------|
| Lower Bound Checker | 21 | 21 | 0.00% | 0.00% |
| SQL Quotes Checker | 14 | 14 | 100.00% | 100.00% |
| Signature String Checker | 21 | 21 | 91.50% | 91.50% |

## Per-Checker Details

### Lower Bound Checker

#### Aggregate Training Metrics

#### Per-Model Metrics

| Model | Annotation Type | Best Accuracy | Final Accuracy |
|-------|-----------------|----------------|----------------|
| gtenegativeone_causal_model |  | 0.00% | 0.00% |
| gtenegativeone_dg2n_model |  | 0.00% | 0.00% |
| gtenegativeone_enhanced_causal_model |  | 0.00% | 0.00% |
| gtenegativeone_gbt_model |  | 0.00% | 0.00% |
| gtenegativeone_gcn_model |  | 0.00% | 0.00% |
| gtenegativeone_gcsn_model |  | 0.00% | 0.00% |
| gtenegativeone_hgt_model |  | 0.00% | 0.00% |
| nonnegative_causal_model |  | 0.00% | 0.00% |
| nonnegative_dg2n_model |  | 0.00% | 0.00% |
| nonnegative_enhanced_causal_model |  | 0.00% | 0.00% |
| nonnegative_gbt_model |  | 0.00% | 0.00% |
| nonnegative_gcn_model |  | 0.00% | 0.00% |
| nonnegative_gcsn_model |  | 0.00% | 0.00% |
| nonnegative_hgt_model |  | 0.00% | 0.00% |
| positive_causal_model |  | 0.00% | 0.00% |
| positive_dg2n_model |  | 0.00% | 0.00% |
| positive_enhanced_causal_model |  | 0.00% | 0.00% |
| positive_gbt_model |  | 0.00% | 0.00% |
| positive_gcn_model |  | 0.00% | 0.00% |
| positive_gcsn_model |  | 0.00% | 0.00% |
| positive_hgt_model |  | 0.00% | 0.00% |

### SQL Quotes Checker

#### Dataset Statistics

- **Total Examples**: 2,000
- **Positive Examples**: 1,000
- **Negative Examples**: 1,000
- **Balance Ratio**: 0.500 (50.0% positive)

**Per-Annotation Type Statistics:**

- @SqlEvenQuotes: 500 positive, 500 negative (balance: 0.500)
- @SqlOddQuotes: 500 positive, 500 negative (balance: 0.500)

#### Aggregate Training Metrics

**Best Validation Accuracy:**
- Mean: 100.00%
- Median: 100.00%
- Min: 100.00%
- Max: 100.00%

**Final Validation Accuracy:**
- Mean: 100.00%
- Median: 100.00%
- Min: 100.00%
- Max: 100.00%

#### Per-Model Metrics

| Model | Annotation Type | Best Accuracy | Final Accuracy |
|-------|-----------------|----------------|----------------|
| sqlevenquotes_causal | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_dg2n | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_enhanced_causal | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_gbt | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_gcn | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_gcsn | @SqlEvenQuotes | 100.00% | 100.00% |
| sqlevenquotes_hgt | @SqlEvenQuotes | 100.00% | 100.00% |
| sqloddquotes_causal | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_dg2n | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_enhanced_causal | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_gbt | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_gcn | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_gcsn | @SqlOddQuotes | 100.00% | 100.00% |
| sqloddquotes_hgt | @SqlOddQuotes | 100.00% | 100.00% |

### Signature String Checker

#### Dataset Statistics

- **Total Examples**: 3,000
- **Positive Examples**: 1,500
- **Negative Examples**: 1,500
- **Balance Ratio**: 0.500 (50.0% positive)

**Per-Annotation Type Statistics:**

- @FullyQualifiedName: 500 positive, 500 negative (balance: 0.500)
- @BinaryName: 500 positive, 500 negative (balance: 0.500)
- @FieldDescriptor: 500 positive, 500 negative (balance: 0.500)

#### Aggregate Training Metrics

**Best Validation Accuracy:**
- Mean: 91.50%
- Median: 99.00%
- Min: 75.50%
- Max: 100.00%

**Final Validation Accuracy:**
- Mean: 91.50%
- Median: 99.00%
- Min: 75.50%
- Max: 100.00%

#### Per-Model Metrics

| Model | Annotation Type | Best Accuracy | Final Accuracy |
|-------|-----------------|----------------|----------------|
| binaryname_causal | @BinaryName | 75.50% | 75.50% |
| binaryname_dg2n | @BinaryName | 75.50% | 75.50% |
| binaryname_enhanced_causal | @BinaryName | 75.50% | 75.50% |
| binaryname_gbt | @BinaryName | 75.50% | 75.50% |
| binaryname_gcn | @BinaryName | 75.50% | 75.50% |
| binaryname_gcsn | @BinaryName | 75.50% | 75.50% |
| binaryname_hgt | @BinaryName | 75.50% | 75.50% |
| fielddescriptor_causal | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_dg2n | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_enhanced_causal | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_gbt | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_gcn | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_gcsn | @FieldDescriptor | 100.00% | 100.00% |
| fielddescriptor_hgt | @FieldDescriptor | 100.00% | 100.00% |
| fullyqualifiedname_causal | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_dg2n | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_enhanced_causal | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_gbt | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_gcn | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_gcsn | @FullyQualifiedName | 99.00% | 99.00% |
| fullyqualifiedname_hgt | @FullyQualifiedName | 99.00% | 99.00% |

## Notes

- All models were trained on balanced datasets (50% positive, 50% negative examples)
- Best accuracy refers to the highest validation accuracy achieved during training
- Final accuracy refers to the validation accuracy at the end of training
- Models are saved with `_balanced` suffix to distinguish from non-balanced models
