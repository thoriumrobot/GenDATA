package cfwr.jdt.transformations.random;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Random Statement Insertion transformation.
 * Tests various patterns and edge cases for random statement insertion.
 */
@DisplayName("Random Statement Insertion Transformation Tests")
class RandomStatementInsertionTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "random_statement_insertion";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Basic Random Insertions")
    class BasicRandomInsertions {
        
        @Test
        @DisplayName("Simple random statement insertion transformation")
        public void testRandomStatementInsertion_Case1_SimpleTransformation() {
            String method = """
                public void simpleTransformation() {
                    // Test case implementation
                    System.out.println("Simple random statement insertion test");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Simple random statement insertion transformation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple random statement insertion transformation should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with parameters")
        public void testRandomStatementInsertion_Case2_WithParameters() {
            String method = """
                public void transformationWithParameters(int x, String str) {
                    // Test case with parameters
                    System.out.println("x=" + x + ", str=" + str);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Random Statement Insertion with parameters should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with parameters should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with return value")
        public void testRandomStatementInsertion_Case3_WithReturnValue() {
            String method = """
                public int transformationWithReturnValue(int x) {
                    return x * 2;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Random Statement Insertion with return value should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with return value should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Insertion Patterns")
    class InsertionPatterns {
        
        @Test
        @DisplayName("Random Statement Insertion in method body")
        public void testRandomStatementInsertion_Case4_InMethodBody() {
            String method = """
                public void inMethodBody() {
                    // Original method body
                    int x = 5;
                    System.out.println("x=" + x);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Random Statement Insertion in method body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion in method body should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion in loop")
        public void testRandomStatementInsertion_Case5_InLoop() {
            String method = """
                public void inLoop() {
                    for (int i = 0; i < 5; i++) {
                        System.out.println("i=" + i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Random Statement Insertion in loop should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion in loop should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion in conditional")
        public void testRandomStatementInsertion_Case6_InConditional() {
            String method = """
                public void inConditional(int x) {
                    if (x > 0) {
                        System.out.println("Positive: " + x);
                    } else {
                        System.out.println("Non-positive: " + x);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertNotNull(transformed, "Transformation should return result");
            assertCompiles(transformed, "Random Statement Insertion in conditional should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion in conditional should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Cases")
    class ComplexCases {
        
        @Test
        @DisplayName("Complex random statement insertion pattern")
        public void testRandomStatementInsertion_Case7_ComplexPattern() {
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
            assertCompiles(transformed, "Complex random statement insertion pattern should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex random statement insertion pattern should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with nested structures")
        public void testRandomStatementInsertion_Case8_NestedStructures() {
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
            assertCompiles(transformed, "Random Statement Insertion with nested structures should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with nested structures should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with method calls")
        public void testRandomStatementInsertion_Case9_WithMethodCalls() {
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
            assertCompiles(transformed, "Random Statement Insertion with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with method calls should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Empty method body")
        public void testRandomStatementInsertion_Case10_EmptyMethodBody() {
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
        public void testRandomStatementInsertion_Case11_SingleStatementMethod() {
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
        @DisplayName("Random Statement Insertion with exception handling")
        public void testRandomStatementInsertion_Case12_WithExceptionHandling() {
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
            assertCompiles(transformed, "Random Statement Insertion with exception handling should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with exception handling should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with null values")
        public void testRandomStatementInsertion_Case13_WithNullValues() {
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
            assertCompiles(transformed, "Random Statement Insertion with null values should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with null values should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with array operations")
        public void testRandomStatementInsertion_Case14_WithArrayOperations() {
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
            assertCompiles(transformed, "Random Statement Insertion with array operations should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with array operations should preserve semantics");
        }
        
        @Test
        @DisplayName("Random Statement Insertion with generic types")
        public void testRandomStatementInsertion_Case15_WithGenericTypes() {
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
            assertCompiles(transformed, "Random Statement Insertion with generic types should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Random Statement Insertion with generic types should preserve semantics");
        }
    }
}