# GenDATA Semantic Transformation Developer Guide

## Overview

This guide provides detailed information for developers working on the GenDATA semantic transformation system and multi-checker annotation placement. It covers the architecture, implementation details, extension points, and best practices for contributing to the project.

## Architecture

### Core Components

```
GenDATA/
├── src/main/java/cfwr/jdt/
│   ├── SemanticTransformer.java          # Main transformation engine
│   ├── TransformationDiagnostics.java    # Diagnostics and reporting
│   └── utils/                            # Utility classes
├── src/test/java/cfwr/jdt/
│   ├── transformations/                  # Individual transformation tests
│   ├── integration/                      # Integration tests
│   ├── performance/                      # Performance tests
│   └── meta/                             # Meta-testing suite
└── docs/                                 # Documentation
```

### Key Classes

#### SemanticTransformer
The main transformation engine that orchestrates all transformations.

**Key Methods:**
- `transformCode(String, List<String>, String)` - Main transformation entry point
- `validateTransformationCompatibility(List<String>)` - Compatibility checking
- `applyTransformation(String, CompilationUnit, ASTRewrite)` - Apply individual transformation

#### TransformationDiagnostics
Records and reports on transformation events, decisions, and performance.

**Key Methods:**
- `recordTransformationStart(String, String, String)` - Record transformation start
- `recordDecision(String, String, boolean)` - Record transformation decision
- `recordTransformationEnd(String, boolean, String, long, String)` - Record transformation end
- `generateReport()` - Generate comprehensive report

## Implementation Details

### Transformation Pipeline

1. **Input Validation**
   - Null and empty string checks
   - Code compilation validation
   - Transformation compatibility checking

2. **AST Parsing**
   - Parse Java code into Eclipse JDT AST
   - Create ASTRewrite for modifications
   - Validate AST structure

3. **Transformation Application**
   - Apply transformations in order
   - Record diagnostics for each transformation
   - Handle errors gracefully

4. **Output Generation**
   - Generate transformed code from AST
   - Validate compilation of result
   - Return transformed code or original on failure

### AST Visitor Pattern

All transformations use the AST Visitor pattern to traverse and modify the AST:

```java
cu.accept(new ASTVisitor() {
    @Override
    public boolean visit(ForStatement node) {
        // Transform for loops
        return true; // Continue visiting children
    }
    
    @Override
    public boolean visit(InfixExpression node) {
        // Transform expressions
        return true;
    }
});
```

### ASTRewrite Usage

Transformations modify the AST using ASTRewrite:

```java
private void transformExpression(InfixExpression expr, ASTRewrite rewrite) {
    AST ast = expr.getAST();
    
    // Create new expression
    InfixExpression newExpr = ast.newInfixExpression();
    newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
    newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
    newExpr.setOperator(expr.getOperator());
    
    // Replace original with new
    rewrite.replace(expr, newExpr, null);
}
```

## Adding New Transformations

### Step 1: Add to Switch Statement

Add your transformation to the main switch statement in `SemanticTransformer.java`:

```java
case "my_transformation":
    return applyMyTransformation(cu, rewrite);
```

### Step 2: Implement Transformation Method

Create the transformation method following the standard pattern:

```java
/**
 * Transform [description of what your transformation does].
 */
private boolean applyMyTransformation(CompilationUnit cu, ASTRewrite rewrite) {
    AtomicBoolean changed = new AtomicBoolean(false);
    
    cu.accept(new ASTVisitor() {
        @Override
        public boolean visit([TargetNodeType] node) {
            if (random.nextDouble() < probability) {
                transformMyPattern(node, rewrite);
                changed.set(true);
            }
            return true;
        }
    });
    
    return changed.get();
}
```

### Step 3: Implement Helper Methods

Create helper methods for specific transformations:

```java
/**
 * Transform [specific pattern].
 */
private void transformMyPattern([TargetNodeType] node, ASTRewrite rewrite) {
    AST ast = node.getAST();
    
    // Create new AST nodes
    // Apply transformations
    // Replace original node
    rewrite.replace(node, newNode, null);
}
```

### Step 4: Add to Compatibility Matrix

Update the compatibility matrix in the constructor:

```java
// Add incompatible transformations
INCOMPATIBLE_TRANSFORMATIONS.put("my_transformation", 
    Arrays.asList("incompatible_transformation1", "incompatible_transformation2"));
```

### Step 5: Create Tests

Create comprehensive tests for your transformation:

```java
@Test
public void testMyTransformation_BasicCase() {
    String originalCode = """
        // Test code here
        """;
    
    String transformedCode = transformer.transformCode(originalCode, 
        Arrays.asList("my_transformation"), "enhanced");
    
    assertCompiles(transformedCode, "Transformation should produce compilable code");
    assertTransformationApplied(originalCode, transformedCode, 
        "Transformation should change code");
}
```

### Step 6: Update Documentation

Update the transformation behavior matrix and user guide with your new transformation.

## Best Practices

### 1. Semantic Preservation
Always ensure transformations preserve program semantics:

```java
// Good: Preserves semantics
if (a && b) -> if (b && a) // Commutativity

// Bad: Changes semantics
if (a && b) -> if (a || b) // Not equivalent
```

### 2. Error Handling
Handle errors gracefully and provide meaningful diagnostics:

```java
try {
    // Transformation logic
    rewrite.replace(node, newNode, null);
    changed.set(true);
} catch (Exception e) {
    // Log error but don't fail the entire transformation
    debug("transformation_error", "Error in transformation: " + e.getMessage());
}
```

### 3. Randomness Control
Use controlled randomness for consistent behavior:

```java
// Good: Controlled probability
if (random.nextDouble() < 0.3) {
    // Apply transformation 30% of the time
}

// Bad: Always apply or never apply
if (true) { // Always applies
if (false) { // Never applies
```

### 4. AST Safety
Always copy AST nodes when modifying:

```java
// Good: Copy nodes
newNode.setLeftOperand((Expression) ASTNode.copySubtree(ast, originalNode.getLeftOperand()));

// Bad: Direct assignment (can cause corruption)
newNode.setLeftOperand(originalNode.getLeftOperand());
```

### 5. Performance Considerations
Consider performance implications of transformations:

```java
// Good: Early return for simple cases
if (!isComplexExpression(node)) {
    return false;
}

// Good: Limit transformation scope
if (random.nextDouble() < 0.1) { // Low probability for expensive transformations
    applyExpensiveTransformation(node, rewrite);
}
```

## Testing Framework

### Test Structure

The testing framework is organized into several layers:

1. **Individual Transformation Tests** (`src/test/java/cfwr/jdt/transformations/`)
   - One test file per transformation
   - 10+ test cases per transformation
   - Exhaustive coverage of edge cases

2. **Integration Tests** (`src/test/java/cfwr/jdt/transformations/integration/`)
   - Test transformation combinations
   - Real-world code validation
   - Performance testing

3. **Meta-Tests** (`src/test/java/cfwr/jdt/transformations/meta/`)
   - Test the test infrastructure
   - Validate transformation correctness
   - Coverage and quality validation

### Test Utilities

#### TransformationTestBase
Base class providing common testing utilities:

```java
public abstract class TransformationTestBase {
    protected SemanticTransformer transformer;
    
    protected void assertCompiles(String code, String message);
    protected void assertSemanticallyEquivalent(String original, String transformed, String message);
    protected void assertTransformationApplied(String original, String transformed, String message);
}
```

#### CompilationValidator
Validates that transformed code compiles:

```java
public class CompilationValidator {
    public boolean compiles(String code);
    public List<String> getCompilationErrors(String code);
}
```

#### SemanticEquivalenceChecker
Checks semantic equivalence between original and transformed code:

```java
public class SemanticEquivalenceChecker {
    public boolean areEquivalent(String original, String transformed);
    public double calculateSimilarity(String original, String transformed);
}
```

### Writing Tests

#### Basic Test Pattern

```java
@Test
public void testTransformation_CaseName() {
    // Arrange
    String originalCode = """
        // Original code here
        """;
    
    // Act
    String transformedCode = transformer.transformCode(originalCode, 
        Arrays.asList("transformation_name"), "enhanced");
    
    // Assert
    assertCompiles(transformedCode, "Transformation should produce compilable code");
    assertSemanticallyEquivalent(originalCode, transformedCode, 
        "Transformation should preserve semantics");
    assertTransformationApplied(originalCode, transformedCode, 
        "Transformation should change code");
}
```

#### Edge Case Testing

```java
@Test
public void testTransformation_EdgeCase() {
    // Test edge cases like:
    // - Empty code
    // - Null inputs
    // - Complex nested structures
    // - Boundary conditions
}
```

#### Integration Testing

```java
@Test
public void testTransformationCombination() {
    // Test multiple transformations together
    // Verify compatibility
    // Check performance
}
```

## Debugging and Diagnostics

### Debug Output

The system provides comprehensive debug output:

```java
debug("transformation_start", "Starting transformation: " + transformation);
debug("transformation_decision", "Applied transformation due to: " + reason);
debug("transformation_end", "Transformation completed in " + duration + "ms");
debug("transformation_error", "Error in transformation: " + error);
```

### Diagnostics System

Access detailed diagnostics information:

```java
SemanticTransformer transformer = new SemanticTransformer();
String result = transformer.transformCode(code, transformations, mode);

// Get detailed diagnostics
TransformationDiagnostics diagnostics = transformer.getDiagnostics();
TransformationDiagnostics.DiagnosticReport report = diagnostics.generateReport();

// Access specific information
List<TransformationDiagnostics.TransformationEvent> events = report.events;
Map<String, Long> performanceMetrics = report.performanceMetrics;
```

### Performance Monitoring

Monitor transformation performance:

```java
long startTime = System.currentTimeMillis();
String result = transformer.transformCode(code, transformations, mode);
long duration = System.currentTimeMillis() - startTime;

// Record performance metric
transformer.getDiagnostics().recordPerformanceMetric("total_transformation_time", duration);
```

## Extension Points

### Custom Transformations

Create custom transformations by extending the base classes:

```java
public class CustomTransformation extends TransformationBase {
    @Override
    public boolean apply(CompilationUnit cu, ASTRewrite rewrite) {
        // Custom transformation logic
        return false;
    }
    
    @Override
    public String getName() {
        return "custom_transformation";
    }
}
```

### Custom Diagnostics

Extend the diagnostics system:

```java
public class CustomDiagnostics extends TransformationDiagnostics {
    @Override
    public void recordCustomEvent(String eventType, String details) {
        // Custom event recording
        super.recordCustomEvent(eventType, details);
    }
}
```

### Custom Test Utilities

Create custom test utilities:

```java
public class CustomTestBase extends TransformationTestBase {
    protected void assertCustomProperty(String code, String property) {
        // Custom assertion logic
    }
}
```

## Performance Optimization

### Optimization Strategies

1. **Early Returns**
   ```java
   if (!shouldTransform(node)) {
       return false;
   }
   ```

2. **Probability Controls**
   ```java
   if (random.nextDouble() < 0.1) { // Low probability for expensive operations
       applyExpensiveTransformation(node, rewrite);
   }
   ```

3. **Caching**
   ```java
   private Map<String, Boolean> compilationCache = new HashMap<>();
   
   private boolean compilesCached(String code) {
       return compilationCache.computeIfAbsent(code, this::compiles);
   }
   ```

4. **Batch Processing**
   ```java
   // Process multiple transformations in one pass
   cu.accept(new ASTVisitor() {
       // Handle multiple transformation types
   });
   ```

### Profiling

Use the built-in profiling capabilities:

```java
// Enable profiling
transformer.enableProfiling();

// Run transformations
String result = transformer.transformCode(code, transformations, mode);

// Get profiling results
Map<String, Long> profile = transformer.getProfilingResults();
```

## Contributing Guidelines

### Code Style

Follow the established code style:

- Use 4 spaces for indentation
- Use meaningful variable and method names
- Add comprehensive JavaDoc comments
- Follow the existing naming conventions

### Commit Messages

Use descriptive commit messages:

```
feat: add bitwise operation transformation

- Implements bitwise AND, OR, XOR transformations
- Adds commutativity for bitwise operations
- Includes comprehensive test coverage
- Updates transformation behavior matrix
```

### Pull Request Process

1. Create a feature branch
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit pull request with detailed description

### Testing Requirements

- All new transformations must have comprehensive tests
- Test coverage must be maintained above 80%
- Integration tests must pass
- Performance benchmarks must not regress

## Troubleshooting

### Common Issues

1. **AST Corruption**
   - Always copy AST nodes when modifying
   - Use ASTRewrite for all modifications
   - Validate AST structure after changes

2. **Compilation Failures**
   - Check for syntax errors in generated code
   - Validate variable scoping
   - Ensure proper type handling

3. **Semantic Equivalence Issues**
   - Test transformations thoroughly
   - Verify edge cases
   - Check for side effects

4. **Performance Issues**
   - Profile transformations
   - Use appropriate probabilities
   - Consider early returns

### Debug Tools

1. **AST Visualization**
   ```java
   // Print AST structure
   cu.accept(new ASTVisitor() {
       @Override
       public boolean visit(ASTNode node) {
           System.out.println(node.getClass().getSimpleName() + ": " + node);
           return true;
       }
   });
   ```

2. **Transformation Tracing**
   ```java
   // Enable detailed tracing
   transformer.enableTracing();
   ```

3. **Memory Profiling**
   ```java
   // Monitor memory usage
   Runtime runtime = Runtime.getRuntime();
   long memoryBefore = runtime.totalMemory() - runtime.freeMemory();
   // ... run transformations
   long memoryAfter = runtime.totalMemory() - runtime.freeMemory();
   ```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Use ML to predict best transformations
   - Learn from transformation success rates
   - Optimize transformation sequences

2. **Parallel Processing**
   - Parallel transformation application
   - Concurrent compilation validation
   - Distributed processing support

3. **Advanced Semantic Analysis**
   - Deeper semantic equivalence checking
   - Program dependence graph analysis
   - Symbolic execution validation

4. **IDE Integration**
   - Eclipse plugin
   - IntelliJ plugin
   - VS Code extension

### Research Areas

1. **Genetic Algorithms**
   - Use GA to evolve transformation sequences
   - Optimize transformation parameters
   - Discover new transformation patterns

2. **Static Analysis Integration**
   - Use static analysis to guide transformations
   - Detect transformation opportunities
   - Validate transformation safety

3. **Semantic Preservation**
   - Better semantic equivalence checking
   - Formal verification of transformations
   - Automated test generation

## Multi-Checker Annotation Placement Implementation

### Architecture

The multi-checker annotation placement system uses a unified architecture:

1. **MultiCheckerPredictor** (`multi_checker_predictor.py`): Unified predictor for all checkers
2. **ComprehensiveAnnotationPlacer** (`place_annotations.py`): Annotation placement engine
3. **SimpleAnnotationTypePipeline** (`simple_annotation_type_pipeline.py`): Pipeline with automatic checker detection

### Key Components

#### MultiCheckerPredictor

```python
from multi_checker_predictor import MultiCheckerPredictor

# Initialize for a specific checker
predictor = MultiCheckerPredictor(
    checker_name='lower_bound',  # or 'sql_quotes', 'signature_string'
    models_dir=None,  # Uses checker-specific directory by default
    device='auto'
)

# Load models
predictor.load_checker_models()

# Predict for a location
prediction = predictor.predict_for_location(cfg_data, node, line_number, threshold=0.3)
# Returns: {'annotation_type': '@Positive', 'confidence': 0.85, ...} or None
```

**Key Features:**
- Confidence-based selection: Selects highest confidence annotation
- Checker-specific model loading: Loads from checker-specific directories
- Unified interface: Same API for all checkers

#### Confidence-Based Selection

The system implements confidence-based selection:

1. **Run All Models**: For each location, all annotation type models for the checker are evaluated
2. **Collect Positive Predictions**: Gather predictions where model says "yes" (confidence >= threshold)
3. **Select Highest Confidence**: Choose the annotation with highest confidence
4. **Place Single Annotation**: Only one annotation is placed per location

**Implementation:**
```python
# In MultiCheckerPredictor.predict_for_location()
predictions = []
for annotation_type in self.annotation_types:
    for base_model in self.base_models:
        is_positive, confidence, reason = self._get_model_prediction(...)
        if is_positive and confidence >= threshold:
            predictions.append({
                'annotation_type': annotation_type,
                'confidence': confidence,
                'model_type': base_model,
                ...
            })

# Select highest confidence
if predictions:
    best_prediction = max(predictions, key=lambda p: p['confidence'])
    return best_prediction
```

#### Checker Detection

The pipeline automatically detects checker from warnings file path:

```python
# In SimpleAnnotationTypePipeline._determine_checker_name()
if 'lower_bound' in warnings_file_lower or 'index1' in warnings_file_lower:
    return 'lower_bound'
elif 'sql_quotes' in warnings_file_lower or 'quotes' in warnings_file_lower:
    return 'sql_quotes'
elif 'signature_string' in warnings_file_lower or 'signature' in warnings_file_lower:
    return 'signature_string'
```

### Adding Support for New Checkers

To add support for a new checker:

1. **Update `checker_evaluation_config.py`**:
   ```python
   'new_checker': {
       'name': 'New Checker',
       'processor': 'org.checkerframework.checker.new.NewChecker',
       'annotation_types': ['@Annotation1', '@Annotation2'],
       'base_models': ['gcn', 'hgt', 'gbt', ...],
       ...
   }
   ```

2. **Train Models**: Use balanced training scripts to train models for all annotation types

3. **Update Feature Extraction**: Ensure `ImprovedBalancedDatasetGenerator` supports the checker

4. **Test**: Verify prediction and placement work correctly

### Path Configuration

The system uses `GEN_DATA_ROOT` from `checker_evaluation_config.py` instead of hardcoded paths:

```python
from checker_evaluation_config import GEN_DATA_ROOT

# Checker-specific directories
if checker_name == 'lower_bound':
    models_dir = GEN_DATA_ROOT / 'models_annotation_types'
else:
    models_dir = GEN_DATA_ROOT / f'models_annotation_types_{checker_name}'
```

### Single Annotation Verification

The placement system includes explicit verification:

```python
# In ComprehensiveAnnotationPlacer.process_predictions()
# Group by location and select highest confidence
predictions_by_location = {}
for pred in predictions:
    location_key = (file_path, pred.line_number)
    predictions_by_location[location_key].append(pred)

# Select highest confidence per location
for location_key, location_predictions in predictions_by_location.items():
    best_prediction = max(location_predictions, key=lambda p: p.confidence)
    # Only one annotation placed per location
```

## Conclusion

The GenDATA semantic transformation system provides a robust, extensible foundation for Java code transformation. By following the guidelines and best practices outlined in this guide, developers can effectively contribute to the project and create new transformations that maintain high quality and performance standards.

For more information, see the [User Guide](USER_GUIDE.md) and [Transformation Behavior Matrix](transformation_behavior_matrix.md).
