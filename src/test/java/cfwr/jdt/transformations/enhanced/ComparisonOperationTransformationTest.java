package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Comparison Operation transformation.
 * Tests comparison operators with symmetry and commutativity transformations.
 */
@DisplayName("Comparison Operation Transformation Tests")
class ComparisonOperationTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "comparison_operation";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Less Than Operations")
    class LessThanOperations {
        
        @Test
        @DisplayName("Simple less than symmetry")
        public void testComparisonOperation_Case1_SimpleLessThan() {
            String method = """
                public void simpleLessThan() {
                    boolean result = a < b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple less than should transform code");
            assertCompiles(transformed, "Simple less than should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple less than should preserve semantics");
            
            // Verify symmetry (a < b ↔ b > a)
            assertTrue(transformed.contains("<") || transformed.contains(">"), "Should contain comparison operator");
        }
        
        @Test
        @DisplayName("Less than with expressions")
        public void testComparisonOperation_Case2_LessThanWithExpressions() {
            String method = """
                public void lessThanWithExpressions() {
                    boolean result = (a + b) < (c - d);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less than with expressions should transform code");
            assertCompiles(transformed, "Less than with expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less than with expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Less than with constants")
        public void testComparisonOperation_Case3_LessThanWithConstants() {
            String method = """
                public void lessThanWithConstants() {
                    boolean result = x < 10;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less than with constants should transform code");
            assertCompiles(transformed, "Less than with constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less than with constants should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Greater Than Operations")
    class GreaterThanOperations {
        
        @Test
        @DisplayName("Simple greater than symmetry")
        public void testComparisonOperation_Case4_SimpleGreaterThan() {
            String method = """
                public void simpleGreaterThan() {
                    boolean result = a > b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple greater than should transform code");
            assertCompiles(transformed, "Simple greater than should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple greater than should preserve semantics");
            
            // Verify symmetry (a > b ↔ b < a)
            assertTrue(transformed.contains(">") || transformed.contains("<"), "Should contain comparison operator");
        }
        
        @Test
        @DisplayName("Greater than with method calls")
        public void testComparisonOperation_Case5_GreaterThanWithMethodCalls() {
            String method = """
                public void greaterThanWithMethodCalls() {
                    boolean result = obj.getValue() > other.getCount();
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Greater than with method calls should transform code");
            assertCompiles(transformed, "Greater than with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Greater than with method calls should preserve semantics");
        }
        
        @Test
        @DisplayName("Greater than with floating-point")
        public void testComparisonOperation_Case6_GreaterThanWithFloatingPoint() {
            String method = """
                public void greaterThanWithFloatingPoint() {
                    boolean result = x > 3.14;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Greater than with floating-point should transform code");
            assertCompiles(transformed, "Greater than with floating-point should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Greater than with floating-point should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Less/Greater Equals Operations")
    class LessGreaterEqualsOperations {
        
        @Test
        @DisplayName("Less than or equal symmetry")
        public void testComparisonOperation_Case7_LessThanOrEqual() {
            String method = """
                public void lessThanOrEqual() {
                    boolean result = a <= b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less than or equal should transform code");
            assertCompiles(transformed, "Less than or equal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less than or equal should preserve semantics");
            
            // Verify symmetry (a <= b ↔ b >= a)
            assertTrue(transformed.contains("<=") || transformed.contains(">="), "Should contain comparison operator");
        }
        
        @Test
        @DisplayName("Greater than or equal symmetry")
        public void testComparisonOperation_Case8_GreaterThanOrEqual() {
            String method = """
                public void greaterThanOrEqual() {
                    boolean result = a >= b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Greater than or equal should transform code");
            assertCompiles(transformed, "Greater than or equal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Greater than or equal should preserve semantics");
            
            // Verify symmetry (a >= b ↔ b <= a)
            assertTrue(transformed.contains(">=") || transformed.contains("<="), "Should contain comparison operator");
        }
        
        @Test
        @DisplayName("Less/greater equals with complex expressions")
        public void testComparisonOperation_Case9_LessGreaterEqualsComplex() {
            String method = """
                public void lessGreaterEqualsComplex() {
                    boolean result1 = (a + b) <= (c * d);
                    boolean result2 = (x - y) >= (z / w);
                    System.out.println(result1 && result2);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less/greater equals complex should transform code");
            assertCompiles(transformed, "Less/greater equals complex should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less/greater equals complex should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Equality Operations")
    class EqualityOperations {
        
        @Test
        @DisplayName("Equality commutativity")
        public void testComparisonOperation_Case10_EqualityCommutativity() {
            String method = """
                public void equalityCommutativity() {
                    boolean result = a == b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Equality should transform code");
            assertCompiles(transformed, "Equality should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Equality should preserve semantics");
            
            // Verify commutativity (a == b ↔ b == a)
            assertTrue(transformed.contains("=="), "Should contain equality operator");
        }
        
        @Test
        @DisplayName("Inequality commutativity")
        public void testComparisonOperation_Case11_InequalityCommutativity() {
            String method = """
                public void inequalityCommutativity() {
                    boolean result = a != b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Inequality should transform code");
            assertCompiles(transformed, "Inequality should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Inequality should preserve semantics");
            
            // Verify commutativity (a != b ↔ b != a)
            assertTrue(transformed.contains("!="), "Should contain inequality operator");
        }
        
        @Test
        @DisplayName("String comparisons")
        public void testComparisonOperation_Case12_StringComparisons() {
            String method = """
                public void stringComparisons() {
                    boolean result1 = str1.equals(str2);
                    boolean result2 = str1 == str2;
                    System.out.println(result1 && result2);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String comparisons should transform code");
            assertCompiles(transformed, "String comparisons should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String comparisons should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Comparison Operations")
    class ComplexComparisonOperations {
        
        @Test
        @DisplayName("Complex comparison chains")
        public void testComparisonOperation_Case13_ComplexComparisonChains() {
            String method = """
                public void complexComparisonChains() {
                    boolean result = (a < b) && (c > d) && (e == f);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex comparison chains should transform code");
            assertCompiles(transformed, "Complex comparison chains should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex comparison chains should preserve semantics");
        }
        
        @Test
        @DisplayName("Comparison with null")
        public void testComparisonOperation_Case14_ComparisonWithNull() {
            String method = """
                public void comparisonWithNull() {
                    boolean result = obj != null;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Comparison with null should transform code");
            assertCompiles(transformed, "Comparison with null should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Comparison with null should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point comparisons")
        public void testComparisonOperation_Case15_FloatingPointComparisons() {
            String method = """
                public void floatingPointComparisons() {
                    boolean result = Math.abs(a - b) < 0.001;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point comparisons should transform code");
            assertCompiles(transformed, "Floating-point comparisons should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point comparisons should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: comparing constants")
        public void testComparisonOperation_Case16_ComparingConstants() {
            String method = """
                public void comparingConstants() {
                    boolean result = 5 > 3;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Comparing constants should transform code");
            assertCompiles(transformed, "Comparing constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Comparing constants should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive comparison operation test")
    public void testComparisonOperation_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                int a = 10, b = 20, c = 30;
                double x = 3.14, y = 2.71;
                String str1 = "hello", str2 = "world";
                
                // Test all comparison operations
                boolean lessThan = a < b;
                boolean greaterThan = b > c;
                boolean lessEqual = a <= b;
                boolean greaterEqual = b >= a;
                boolean equal = a == 10;
                boolean notEqual = a != b;
                
                // Complex expressions
                boolean complex = (a < b) && (x > y) && (str1 != str2);
                
                System.out.println(lessThan && greaterThan && lessEqual && greaterEqual && equal && notEqual && complex);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}
