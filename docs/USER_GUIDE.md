# GenDATA Semantic Transformation User Guide

## Overview

The GenDATA project provides a comprehensive semantic transformation system for Java code that preserves program semantics while generating diverse code variants. This system is designed to support machine learning training by creating semantically equivalent but structurally different code patterns.

## Quick Start

### Basic Usage

```java
import cfwr.jdt.SemanticTransformer;

// Create a transformer instance
SemanticTransformer transformer = new SemanticTransformer();

// Transform code with specific transformations
String originalCode = """
    public class Example {
        public void method() {
            for (int i = 0; i < 10; i++) {
                System.out.println(i);
            }
        }
    }
    """;

List<String> transformations = Arrays.asList("loop_conversion", "mathematical_expression");
String transformedCode = transformer.transformCode(originalCode, transformations, "enhanced");

System.out.println(transformedCode);
```

### Command Line Usage

```bash
# Transform a single file
java -jar build/libs/jdt-transformer-all.jar \
    --input input.java \
    --output output.java \
    --transformations "loop_conversion,mathematical_expression" \
    --mode enhanced \
    --seed 42

# Transform multiple files
find src/ -name "*.java" | while read file; do
    java -jar build/libs/jdt-transformer-all.jar \
        --input "$file" \
        --output "transformed/$file" \
        --transformations "loop_conversion,guard_reversal" \
        --mode enhanced
done
```

## Available Transformations

### Enhanced Transformations (17 types)

These transformations apply sophisticated code transformations that preserve semantics while changing structure.

#### 1. Loop Conversion (`loop_conversion`)
Converts between for and while loops while preserving semantics.

**Example:**
```java
// Before
for (int i = 0; i < 10; i++) {
    System.out.println(i);
}

// After
int i = 0;
while (i < 10) {
    System.out.println(i);
    i++;
}
```

#### 2. Guard Reversal (`guard_reversal`)
Reverses conditional logic using De Morgan's laws.

**Example:**
```java
// Before
if (a && b) {
    doSomething();
}

// After
if (!(!a || !b)) {
    doSomething();
}
```

#### 3. Mathematical Expression (`mathematical_expression`)
Applies mathematical properties while preserving semantics.

**Example:**
```java
// Before
int result = 5 + 3 * 2;

// After
int result = 3 * 2 + 5; // Commutativity
```

#### 4. Logical Expression (`logical_expression`)
Applies boolean algebra transformations.

**Example:**
```java
// Before
boolean flag = a && b && a;

// After
boolean flag = a && b; // Idempotence
```

#### 5. Ternary Operator (`ternary_operator`)
Converts between ternary operators and if-else statements.

**Example:**
```java
// Before
String result = condition ? "yes" : "no";

// After
String result;
if (condition) {
    result = "yes";
} else {
    result = "no";
}
```

#### 6. Switch Statement (`switch_statement`)
Transforms switch statements to if-else chains and vice versa.

#### 7. Variable Operation (`variable_operation`)
Transforms variable operations and assignments.

#### 8. Method Extraction (`method_extraction`)
Extracts repeated code patterns into methods.

#### 9. Conditional Expression (`conditional_expression`)
Transforms conditional expressions and statements.

#### 10. Array Access Pattern (`array_access_pattern`)
Transforms array access patterns and operations.

#### 11. String Concatenation (`string_concatenation`)
Transforms string concatenation patterns.

#### 12. Numeric Literal (`numeric_literal`)
Transforms numeric literals and constants.

#### 13. Exception Handling (`exception_handling`)
Transforms exception handling patterns.

#### 14. Lambda Expression (`lambda_expression`)
Transforms lambda expressions and functional interfaces.

#### 15. Stream API (`stream_api`)
Transforms Stream API operations.

#### 16. Builder Pattern (`builder_pattern`)
Transforms builder pattern implementations.

#### 17. Functional Conversion (`functional_conversion`)
Converts between functional and imperative styles.

### Simple Transformations (10 types)

These transformations apply basic code transformations with minimal complexity.

#### 18. Simple Method Call (`simple_method_call`)
Transforms simple method calls and invocations.

#### 19. Simple Assignment (`simple_assignment`)
Transforms simple assignment operations.

#### 20. Simple Conditional (`simple_conditional`)
Transforms simple conditional statements.

#### 21. Simple Array Access (`simple_array_access`)
Transforms simple array access operations.

#### 22. Simple Return Statement (`simple_return_statement`)
Transforms simple return statements.

#### 23. Simple Variable Declaration (`simple_variable_declaration`)
Transforms simple variable declarations.

#### 24. Simple Constructor Call (`simple_constructor_call`)
Transforms simple constructor calls.

#### 25. Simple Field Access (`simple_field_access`)
Transforms simple field access operations.

#### 26. Simple String Operation (`simple_string_operation`)
Transforms simple string operations.

#### 27. Simple Numeric Operation (`simple_numeric_operation`)
Transforms simple numeric operations.

### Random Transformations (3 types)

These transformations add random elements to the code for diversity.

#### 28. Random Method Insertion (`random_method_insertion`)
Inserts random method calls at appropriate locations.

#### 29. Random Statement Insertion (`random_statement_insertion`)
Inserts random statements at appropriate locations.

#### 30. Random Expression Insertion (`random_expression_insertion`)
Inserts random expressions at appropriate locations.

### New Advanced Transformations (8 types)

These are the newly added transformation types that provide advanced code transformation capabilities.

#### 31. Bitwise Operation (`bitwise_operation`)
Transforms bitwise operations using bitwise algebra properties.

**Example:**
```java
// Before
int result = a & b;

// After
int result = b & a; // Commutativity
```

#### 32. Comparison Operation (`comparison_operation`)
Transforms comparison operations using comparison algebra.

**Example:**
```java
// Before
boolean flag = a < b;

// After
boolean flag = b > a; // Equivalent comparison
```

#### 33. Type Conversion (`type_conversion`)
Transforms type conversions and casts.

**Example:**
```java
// Before
int x = (int) 42;

// After
int x = 42; // Remove redundant cast
```

#### 34. Null Check Pattern (`null_check_pattern`)
Transforms null check patterns.

**Example:**
```java
// Before
if (obj != null) {
    obj.doSomething();
}

// After
if (!Objects.isNull(obj)) {
    obj.doSomething();
}
```

#### 35. Constant Folding (`constant_folding`)
Applies constant folding optimizations.

**Example:**
```java
// Before
int result = 5 + 3;

// After
int result = 8; // Constant folding
```

#### 36. Dead Code Insertion (`dead_code_insertion`)
Inserts dead code that doesn't affect program semantics.

**Example:**
```java
// Before
public void method() {
    doSomething();
}

// After
public void method() {
    doSomething();
    0; // Dead code insertion
}
```

#### 37. Method Chain Transformation (`method_chain_transformation`)
Transforms method chaining patterns.

**Example:**
```java
// Before
String result = obj.method1().method2();

// After
String temp = obj.method1();
String result = temp.method2();
```

#### 38. Variable Renaming (`variable_renaming`)
Transforms variable names to add variety.

**Example:**
```java
// Before
int count = 0;

// After
int newCount = 0;
```

## Transformation Modes

### Enhanced Mode (`enhanced`)
Applies sophisticated transformations with higher complexity and semantic preservation.

```java
String result = transformer.transformCode(code, transformations, "enhanced");
```

### Simple Mode (`simple`)
Applies basic transformations with minimal complexity.

```java
String result = transformer.transformCode(code, transformations, "simple");
```

## Compatibility Matrix

### Incompatible Transformations
- `loop_conversion` ❌ `guard_reversal` - May create conflicting conditional structures

### Compatible Groups
1. **Mathematical Group**: `mathematical_expression`, `simple_numeric_operation`, `numeric_literal`, `bitwise_operation`, `constant_folding`
2. **Logical Group**: `logical_expression`, `simple_conditional`, `conditional_expression`, `comparison_operation`
3. **Loop Group**: `loop_conversion`, `functional_conversion`, `stream_api`
4. **String Group**: `string_concatenation`, `simple_string_operation`
5. **Array Group**: `array_access_pattern`, `simple_array_access`
6. **Method Group**: `method_extraction`, `simple_method_call`, `builder_pattern`, `method_chain_transformation`
7. **Exception Group**: `exception_handling`, `lambda_expression`
8. **Random Group**: `random_method_insertion`, `random_statement_insertion`, `random_expression_insertion`, `dead_code_insertion`
9. **Type Group**: `type_conversion`, `variable_renaming`, `null_check_pattern`

## Best Practices

### 1. Transformation Selection
- Use compatible transformation groups together
- Avoid incompatible transformations in the same transformation set
- Start with simple transformations and gradually add complex ones

### 2. Code Quality
- Always validate that transformed code compiles
- Test transformed code to ensure semantic equivalence
- Use appropriate transformation modes for your use case

### 3. Performance Considerations
- Simple transformations are faster than enhanced transformations
- Mathematical expressions can be slow due to complex AST manipulation
- Consider using fewer transformations for large codebases

### 4. Error Handling
- The system handles null inputs gracefully
- Invalid code is returned unchanged
- Check the diagnostics report for transformation details

## Examples

### Example 1: Loop and Mathematical Transformations

```java
String code = """
    public class Calculator {
        public int calculate() {
            int sum = 0;
            for (int i = 0; i < 10; i++) {
                sum = sum + i * 2;
            }
            return sum;
        }
    }
    """;

List<String> transformations = Arrays.asList(
    "loop_conversion", 
    "mathematical_expression", 
    "simple_assignment"
);

SemanticTransformer transformer = new SemanticTransformer();
String result = transformer.transformCode(code, transformations, "enhanced");
```

### Example 2: Logical and Comparison Transformations

```java
String code = """
    public class Validator {
        public boolean isValid(int value) {
            return value > 0 && value < 100;
        }
    }
    """;

List<String> transformations = Arrays.asList(
    "logical_expression",
    "comparison_operation",
    "null_check_pattern"
);

SemanticTransformer transformer = new SemanticTransformer();
String result = transformer.transformCode(code, transformations, "enhanced");
```

### Example 3: Multiple Compatible Transformations

```java
String code = """
    public class Processor {
        public String process(String input) {
            String result = "";
            for (int i = 0; i < input.length(); i++) {
                char c = input.charAt(i);
                if (c != ' ') {
                    result = result + c;
                }
            }
            return result;
        }
    }
    """;

List<String> transformations = Arrays.asList(
    "loop_conversion",
    "string_concatenation", 
    "simple_conditional",
    "array_access_pattern",
    "type_conversion"
);

SemanticTransformer transformer = new SemanticTransformer();
String result = transformer.transformCode(code, transformations, "enhanced");
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Check for incompatible transformations
   - Verify input code is valid Java
   - Use simpler transformation sets

2. **Performance Issues**
   - Reduce number of transformations
   - Use simple mode instead of enhanced
   - Avoid mathematical_expression for large codebases

3. **Semantic Equivalence Issues**
   - Test transformed code thoroughly
   - Use appropriate transformation combinations
   - Check diagnostics report for details

### Getting Help

- Check the transformation behavior matrix for detailed information
- Review the test suite for usage examples
- Examine the diagnostics report for transformation details

## Advanced Features

### Diagnostics System

The transformation system includes a comprehensive diagnostics system that records:

- Transformation events and decisions
- Performance metrics
- Error information
- Success/failure rates

```java
SemanticTransformer transformer = new SemanticTransformer();
String result = transformer.transformCode(code, transformations, "enhanced");

// Get diagnostics report
TransformationDiagnostics.DiagnosticReport report = transformer.getDiagnosticsReport();
System.out.println(report);
```

### Meta-Testing

The system includes meta-testing capabilities to validate:

- Test infrastructure correctness
- Transformation correctness
- Test coverage validation
- Test quality validation

### Performance Benchmarking

Built-in performance benchmarking provides:

- Time per transformation
- Memory usage metrics
- Scalability analysis
- Success rate measurements

## Conclusion

The GenDATA semantic transformation system provides a powerful and flexible way to generate diverse code variants while preserving program semantics. With 38 different transformation types, comprehensive compatibility checking, and advanced features like diagnostics and meta-testing, it's well-suited for machine learning training and code analysis applications.

For more detailed information, see the [Developer Guide](DEVELOPER_GUIDE.md) and [Transformation Behavior Matrix](transformation_behavior_matrix.md).
