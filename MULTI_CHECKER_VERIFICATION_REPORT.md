# Multi-Checker Infrastructure Verification Report

**Generated**: 1765213019.5888457

## Summary

- Total Tests: 20
- Passed: 20
- Failed: 0
- Success Rate: 100.0%

## Test Results

### LowerBoundChecker Interface Compliance

**Status**: ✅ PASS

**Info**:
- All 0 abstract methods are implemented
- Checker instance created successfully
- get_checker_name() = 'LowerBound'
- get_checker_processor() = 'org.checkerframework.checker.index.IndexChecker'
- get_annotation_types() = ['@Positive', '@NonNegative', '@GTENegativeOne']
- get_training_data_source() = '/home/ubuntu/checker-framework/checker/tests/index/'
- get_warning_patterns() = 6 patterns
- parse_warnings() handles missing file correctly (returned 0 warnings)
- extract_features() returned 6 features
- validate_annotation() works correctly


### SqlQuotesChecker Interface Compliance

**Status**: ✅ PASS

**Info**:
- All 0 abstract methods are implemented
- Checker instance created successfully
- get_checker_name() = 'SqlQuotes'
- get_checker_processor() = 'org.checkerframework.checker.quotes.QuotesChecker'
- get_annotation_types() = ['@SqlEvenQuotes', '@SqlOddQuotes']
- get_training_data_source() = '/home/ubuntu/checker-framework/checker/tests/quotes/'
- get_warning_patterns() = 5 patterns
- parse_warnings() handles missing file correctly (returned 0 warnings)
- extract_features() returned 7 features
- validate_annotation() works correctly


### SignatureStringChecker Interface Compliance

**Status**: ✅ PASS

**Info**:
- All 0 abstract methods are implemented
- Checker instance created successfully
- get_checker_name() = 'SignatureString'
- get_checker_processor() = 'org.checkerframework.checker.signature.qual.SignatureChecker'
- get_annotation_types() = ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor']
- get_training_data_source() = '/home/ubuntu/checker-framework/checker/tests/signature/'
- get_warning_patterns() = 6 patterns
- parse_warnings() handles missing file correctly (returned 0 warnings)
- extract_features() returned 8 features
- validate_annotation() works correctly


### List Checkers

**Status**: ✅ PASS

**Info**:
- Found 7 registered checkers: ['lower_bound', 'lowerbound', 'index', 'sql_quotes', 'sqlquotes', 'signature_string', 'signaturestring']


### Get Checker: lower_bound

**Status**: ✅ PASS

**Info**:
- Successfully retrieved checker 'lower_bound'
- Case-insensitive retrieval works


### Get Checker: sql_quotes

**Status**: ✅ PASS

**Info**:
- Successfully retrieved checker 'sql_quotes'
- Case-insensitive retrieval works


### Get Checker: signature_string

**Status**: ✅ PASS

**Info**:
- Successfully retrieved checker 'signature_string'
- Case-insensitive retrieval works


### Is Checker Registered

**Status**: ✅ PASS

**Info**:
- is_checker_registered('lower_bound') = True
- is_checker_registered('sql_quotes') = True
- is_checker_registered('signature_string') = True


### Get Unknown Checker

**Status**: ✅ PASS

**Info**:
- get_checker() correctly returns None for unknown checker


### Checker Selection by Name

**Status**: ✅ PASS

**Info**:
- Successfully selected checker 'lower_bound' -> org.checkerframework.checker.index.IndexChecker


### SQL Quotes Checker Selection

**Status**: ✅ PASS

**Info**:
- Successfully selected SQL Quotes checker


### Signature String Checker Selection

**Status**: ✅ PASS

**Info**:
- Successfully selected Signature String checker


### Fallback to Default Processor

**Status**: ✅ PASS

**Info**:
- Correctly fell back to default processor


### Checker-Specific Warning Parsing

**Status**: ✅ PASS

**Info**:
- Successfully parsed warnings file: 1 warnings


### Get All Checker Names

**Status**: ✅ PASS

**Info**:
- Found 3 checkers: ['lower_bound', 'sql_quotes', 'signature_string']


### Get Config: lower_bound

**Status**: ✅ PASS

**Info**:
- Configuration complete for lower_bound


### Get Config: sql_quotes

**Status**: ✅ PASS

**Info**:
- Configuration complete for sql_quotes


### Get Config: signature_string

**Status**: ✅ PASS

**Info**:
- Configuration complete for signature_string


### Build Model Name

**Status**: ✅ PASS

**Info**:
- build_model_name() works correctly: 'positive_gcn'


### Get Evaluation Projects

**Status**: ✅ PASS

**Info**:
- Found 6 evaluation projects for lower_bound: ['guava', 'jfreechart', 'plume-lib', 'agrona', 'hipparchus', 'eclipse-collections']


