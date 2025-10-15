package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Builder Pattern transformation.
 * Tests various patterns and edge cases for builder pattern.
 */
@DisplayName("Builder Pattern Transformation Tests")
class BuilderPatternTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "builder_pattern";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Basic Transformations")
    class BasicTransformations {
        
        @Test
        @DisplayName("Simple builder pattern transformation")
        public void testBuilderPattern_Case1_SimpleTransformation() {
            String method = """
                public void simpleTransformation() {
                    // Test case implementation
                    System.out.println("Simple builder pattern test");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple builder pattern transformation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple builder pattern transformation should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with parameters")
        public void testBuilderPattern_Case2_WithParameters() {
            String method = """
                public void transformationWithParameters(int x, String str) {
                    // Test case with parameters
                    System.out.println("x=" + x + ", str=" + str);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Builder Pattern with parameters should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with parameters should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with return value")
        public void testBuilderPattern_Case3_WithReturnValue() {
            String method = """
                public int transformationWithReturnValue(int x) {
                    return x * 2;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Builder Pattern with return value should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with return value should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Cases")
    class ComplexCases {
        
        @Test
        @DisplayName("Complex builder pattern pattern")
        public void testBuilderPattern_Case4_ComplexPattern() {
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
            assertCompiles(transformed, "Complex builder pattern pattern should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex builder pattern pattern should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with nested structures")
        public void testBuilderPattern_Case5_NestedStructures() {
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
            assertCompiles(transformed, "Builder Pattern with nested structures should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with nested structures should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with method calls")
        public void testBuilderPattern_Case6_WithMethodCalls() {
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
            assertCompiles(transformed, "Builder Pattern with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with method calls should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Empty method body")
        public void testBuilderPattern_Case7_EmptyMethodBody() {
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
        public void testBuilderPattern_Case8_SingleStatementMethod() {
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
        @DisplayName("Builder Pattern with exception handling")
        public void testBuilderPattern_Case9_WithExceptionHandling() {
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
            assertCompiles(transformed, "Builder Pattern with exception handling should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with exception handling should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with null values")
        public void testBuilderPattern_Case10_WithNullValues() {
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
            assertCompiles(transformed, "Builder Pattern with null values should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with null values should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with array operations")
        public void testBuilderPattern_Case11_WithArrayOperations() {
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
            assertCompiles(transformed, "Builder Pattern with array operations should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with array operations should preserve semantics");
        }
        
        @Test
        @DisplayName("Builder Pattern with generic types")
        public void testBuilderPattern_Case12_WithGenericTypes() {
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
            assertCompiles(transformed, "Builder Pattern with generic types should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Builder Pattern with generic types should preserve semantics");
        }
    }
}