package cfwr.jdt.transformations.simple;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Simple String Operation transformation.
 * Tests various patterns and edge cases for simple string operation.
 */
@DisplayName("Simple String Operation Transformation Tests")
class SimpleStringOperationTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "simple_string_operation";
    private static final String MODE = "simple";
    
    @Nested
    @DisplayName("Basic Transformations")
    class BasicTransformations {
        
        @Test
        @DisplayName("Simple simple string operation transformation")
        public void testSimpleStringOperation_Case1_SimpleTransformation() {
            String method = """
                public void simpleTransformation() {
                    // Test case implementation
                    System.out.println("Simple simple string operation test");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple simple string operation transformation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple simple string operation transformation should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with parameters")
        public void testSimpleStringOperation_Case2_WithParameters() {
            String method = """
                public void transformationWithParameters(int x, String str) {
                    // Test case with parameters
                    System.out.println("x=" + x + ", str=" + str);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with parameters should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with parameters should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with return value")
        public void testSimpleStringOperation_Case3_WithReturnValue() {
            String method = """
                public int transformationWithReturnValue(int x) {
                    return x * 2;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with return value should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with return value should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Formatting Variations")
    class FormattingVariations {
        
        @Test
        @DisplayName("Simple String Operation spacing variations")
        public void testSimpleStringOperation_Case4_SpacingVariations() {
            String method = """
                public void spacingVariations() {
                    // Test different spacing patterns
                    int x=5;
                    int y = 10;
                    String s="test";
                    String t = "hello";
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation spacing variations should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation spacing variations should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation parenthesization")
        public void testSimpleStringOperation_Case5_Parenthesization() {
            String method = """
                public void parenthesization() {
                    // Test parenthesization patterns
                    int result = 5 + 3 * 2;
                    String message = "Hello" + "World";
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation parenthesization should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation parenthesization should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with comments")
        public void testSimpleStringOperation_Case6_WithComments() {
            String method = """
                public void withComments() {
                    // Single line comment
                    int x = 5; /* inline comment */
                    
                    /*
                     * Multi-line comment
                     */
                    String str = "test";
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with comments should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with comments should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Cases")
    class ComplexCases {
        
        @Test
        @DisplayName("Complex simple string operation pattern")
        public void testSimpleStringOperation_Case7_ComplexPattern() {
            String method = """
                public void complexPattern() {
                    // Complex test case
                    int[] array = {1, 2, 3, 4, 5};
                    for (int i = 0; i < array.length; i++) {
                        if (array[i] > 2) {
                            System.out.println("Large value: " + array[i]);
                        }
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Complex simple string operation pattern should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex simple string operation pattern should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with nested structures")
        public void testSimpleStringOperation_Case8_NestedStructures() {
            String method = """
                public void nestedStructures() {
                    try {
                        for (int i = 0; i < 10; i++) {
                            if (i % 2 == 0) {
                                System.out.println("Even: " + i);
                            } else {
                                System.out.println("Odd: " + i);
                            }
                        }
                    } catch (Exception e) {
                        System.out.println("Error: " + e.getMessage());
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with nested structures should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with nested structures should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with method calls")
        public void testSimpleStringOperation_Case9_WithMethodCalls() {
            String method = """
                public void withMethodCalls(int x, int y) {
                    int result = calculate(x, y);
                    System.out.println("Result: " + result);
                }
                
                private int calculate(int a, int b) {
                    return a + b * 2;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with method calls should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Empty method body")
        public void testSimpleStringOperation_Case10_EmptyMethodBody() {
            String method = """
                public void emptyMethodBody() {
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Empty method body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Empty method body should preserve semantics");
        }
        
        @Test
        @DisplayName("Single statement method")
        public void testSimpleStringOperation_Case11_SingleStatementMethod() {
            String method = """
                public int singleStatementMethod() {
                    return 42;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Single statement method should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single statement method should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with exception handling")
        public void testSimpleStringOperation_Case12_WithExceptionHandling() {
            String method = """
                public void withExceptionHandling() {
                    try {
                        riskyOperation();
                    } catch (RuntimeException e) {
                        System.out.println("Caught: " + e.getMessage());
                    } finally {
                        cleanup();
                    }
                }
                
                private void riskyOperation() {
                    throw new RuntimeException("Test exception");
                }
                
                private void cleanup() {
                    System.out.println("Cleanup completed");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with exception handling should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with exception handling should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with null values")
        public void testSimpleStringOperation_Case13_WithNullValues() {
            String method = """
                public void withNullValues(String str, Integer num) {
                    if (str != null && num != null) {
                        System.out.println("str=" + str + ", num=" + num);
                    } else {
                        System.out.println("Null values detected");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with null values should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with null values should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with array operations")
        public void testSimpleStringOperation_Case14_WithArrayOperations() {
            String method = """
                public void withArrayOperations() {
                    int[] numbers = {1, 2, 3, 4, 5};
                    String[] names = {"Alice", "Bob", "Charlie"};
                    
                    for (int i = 0; i < numbers.length; i++) {
                        System.out.println(numbers[i] + ": " + names[i % names.length]);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with array operations should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with array operations should preserve semantics");
        }
        
        @Test
        @DisplayName("Simple String Operation with generic types")
        public void testSimpleStringOperation_Case15_WithGenericTypes() {
            String method = """
                public <T> void withGenericTypes(T value) {
                    if (value instanceof String) {
                        System.out.println("String: " + value);
                    } else if (value instanceof Integer) {
                        System.out.println("Integer: " + value);
                    } else {
                        System.out.println("Other: " + value);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple String Operation with generic types should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple String Operation with generic types should preserve semantics");
        }
    }
}