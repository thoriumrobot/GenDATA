package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Guard Reversal transformation.
 * Tests if-else condition flipping with various guard patterns.
 */
@DisplayName("Guard Reversal Transformation Tests")
class GuardReversalTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "guard_reversal";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Simple Guard Reversals")
    class SimpleGuardReversals {
        
        @Test
        @DisplayName("Simple if-else guard reversal")
        public void testGuardReversal_Case1_SimpleIfElse() {
            String method = """
                public String simpleGuard(int x) {
                    if (x > 0) {
                        return "positive";
                    } else {
                        return "negative";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple if-else should transform code");
            assertCompiles(transformed, "Simple if-else should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple if-else should preserve semantics");
            
            // Verify guard reversal occurred
            assertTrue(transformed.contains("x <= 0") || transformed.contains("!(x > 0)"), 
                "Guard condition should be reversed");
        }
        
        @Test
        @DisplayName("Guard with equality comparison")
        public void testGuardReversal_Case2_EqualityComparison() {
            String method = """
                public String equalityGuard(int x) {
                    if (x == 0) {
                        return "zero";
                    } else {
                        return "non-zero";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Equality comparison should transform code");
            assertCompiles(transformed, "Equality comparison should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Equality comparison should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with less than comparison")
        public void testGuardReversal_Case3_LessThanComparison() {
            String method = """
                public String lessThanGuard(int x) {
                    if (x < 10) {
                        return "small";
                    } else {
                        return "large";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less than comparison should transform code");
            assertCompiles(transformed, "Less than comparison should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less than comparison should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with greater than or equal")
        public void testGuardReversal_Case4_GreaterThanOrEqual() {
            String method = """
                public String greaterThanOrEqualGuard(int x) {
                    if (x >= 5) {
                        return "sufficient";
                    } else {
                        return "insufficient";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Greater than or equal should transform code");
            assertCompiles(transformed, "Greater than or equal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Greater than or equal should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with less than or equal")
        public void testGuardReversal_Case5_LessThanOrEqual() {
            String method = """
                public String lessThanOrEqualGuard(int x) {
                    if (x <= 100) {
                        return "valid";
                    } else {
                        return "invalid";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Less than or equal should transform code");
            assertCompiles(transformed, "Less than or equal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Less than or equal should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with not equal comparison")
        public void testGuardReversal_Case6_NotEqualComparison() {
            String method = """
                public String notEqualGuard(int x) {
                    if (x != 0) {
                        return "non-zero";
                    } else {
                        return "zero";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Not equal comparison should transform code");
            assertCompiles(transformed, "Not equal comparison should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Not equal comparison should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Guard Patterns")
    class ComplexGuardPatterns {
        
        @Test
        @DisplayName("Guard with boolean variable")
        public void testGuardReversal_Case7_BooleanVariable() {
            String method = """
                public String booleanGuard(boolean flag) {
                    if (flag) {
                        return "enabled";
                    } else {
                        return "disabled";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Boolean variable should transform code");
            assertCompiles(transformed, "Boolean variable should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Boolean variable should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with null check")
        public void testGuardReversal_Case8_NullCheck() {
            String method = """
                public String nullCheckGuard(String str) {
                    if (str != null) {
                        return str.toUpperCase();
                    } else {
                        return "null";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Null check should transform code");
            assertCompiles(transformed, "Null check should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Null check should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with instanceof check")
        public void testGuardReversal_Case9_InstanceofCheck() {
            String method = """
                public String instanceofGuard(Object obj) {
                    if (obj instanceof String) {
                        return ((String) obj).toUpperCase();
                    } else {
                        return obj.toString();
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Instanceof check should transform code");
            assertCompiles(transformed, "Instanceof check should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Instanceof check should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with complex boolean expression")
        public void testGuardReversal_Case10_ComplexBooleanExpression() {
            String method = """
                public String complexBooleanGuard(int x, int y) {
                    if (x > 0 && y < 100) {
                        return "valid range";
                    } else {
                        return "invalid range";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex boolean expression should transform code");
            assertCompiles(transformed, "Complex boolean expression should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex boolean expression should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with OR expression")
        public void testGuardReversal_Case11_OrExpression() {
            String method = """
                public String orExpressionGuard(int x) {
                    if (x < 0 || x > 100) {
                        return "out of range";
                    } else {
                        return "in range";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "OR expression should transform code");
            assertCompiles(transformed, "OR expression should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "OR expression should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with nested parentheses")
        public void testGuardReversal_Case12_NestedParentheses() {
            String method = """
                public String nestedParenthesesGuard(int x, int y, int z) {
                    if ((x > 0 && y < 10) || (z == 0)) {
                        return "condition met";
                    } else {
                        return "condition not met";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested parentheses should transform code");
            assertCompiles(transformed, "Nested parentheses should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested parentheses should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Multiple Guard Statements")
    class MultipleGuardStatements {
        
        @Test
        @DisplayName("Multiple sequential if-else statements")
        public void testGuardReversal_Case13_MultipleSequential() {
            String method = """
                public String multipleSequentialGuards(int x) {
                    if (x > 0) {
                        return "positive";
                    } else if (x < 0) {
                        return "negative";
                    } else {
                        return "zero";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple sequential should transform code");
            assertCompiles(transformed, "Multiple sequential should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiple sequential should preserve semantics");
        }
        
        @Test
        @DisplayName("Nested if-else statements")
        public void testGuardReversal_Case14_NestedIfElse() {
            String method = """
                public String nestedIfElseGuards(int x, int y) {
                    if (x > 0) {
                        if (y > 0) {
                            return "both positive";
                        } else {
                            return "x positive, y non-positive";
                        }
                    } else {
                        return "x non-positive";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested if-else should transform code");
            assertCompiles(transformed, "Nested if-else should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested if-else should preserve semantics");
        }
        
        @Test
        @DisplayName("If without else clause")
        public void testGuardReversal_Case15_IfWithoutElse() {
            String method = """
                public void ifWithoutElse(int x) {
                    if (x > 0) {
                        System.out.println("positive");
                    }
                    System.out.println("always printed");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // If without else should not be transformed
            assertNoTransformation(original, transformed, "If without else");
            assertCompiles(transformed, "If without else should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "If without else should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Guard with Method Calls")
    class GuardWithMethodCalls {
        
        @Test
        @DisplayName("Guard with pure method call (should transform)")
        public void testGuardReversal_Case16_PureMethodCall() {
            String method = """
                public String pureMethodCallGuard(int x) {
                    if (isPositive(x)) {
                        return "positive";
                    } else {
                        return "non-positive";
                    }
                }
                
                private boolean isPositive(int x) {
                    return x > 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Pure method calls should allow transformation
            assertTransformationApplied(original, transformed, "Pure method call should transform code");
            assertCompiles(transformed, "Pure method call should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Pure method call should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with side-effect method call (should skip)")
        public void testGuardReversal_Case17_SideEffectMethodCall() {
            String method = """
                public String sideEffectMethodCallGuard(int x) {
                    if (hasSideEffect(x)) {
                        return "side effect occurred";
                    } else {
                        return "no side effect";
                    }
                }
                
                private boolean hasSideEffect(int x) {
                    System.out.println("Side effect for " + x);
                    return x > 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Side-effect method calls should not be transformed
            // The transformation should be skipped
            assertNoTransformation(original, transformed, "Side-effect method call");
            assertCompiles(transformed, "Side-effect method call should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Side-effect method call should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with array access")
        public void testGuardReversal_Case18_ArrayAccess() {
            String method = """
                public String arrayAccessGuard(int[] array, int index) {
                    if (array[index] > 0) {
                        return "positive element";
                    } else {
                        return "non-positive element";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Array access should transform code");
            assertCompiles(transformed, "Array access should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Array access should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with field access")
        public void testGuardReversal_Case19_FieldAccess() {
            String method = """
                private int threshold = 10;
                
                public String fieldAccessGuard(int x) {
                    if (x > threshold) {
                        return "above threshold";
                    } else {
                        return "below threshold";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Field access should transform code");
            assertCompiles(transformed, "Field access should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Field access should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCases {
        
        @Test
        @DisplayName("Guard with complex nested expressions")
        public void testGuardReversal_Case20_ComplexNestedExpressions() {
            String method = """
                public String complexNestedExpressionsGuard(int x, int y, int z) {
                    if (((x + y) * z) > (x * y + z)) {
                        return "complex condition true";
                    } else {
                        return "complex condition false";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex nested expressions should transform code");
            assertCompiles(transformed, "Complex nested expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex nested expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with floating-point comparison")
        public void testGuardReversal_Case21_FloatingPointComparison() {
            String method = """
                public String floatingPointGuard(double x) {
                    if (x > 0.0) {
                        return "positive";
                    } else {
                        return "non-positive";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point comparison should transform code");
            assertCompiles(transformed, "Floating-point comparison should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point comparison should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with string comparison")
        public void testGuardReversal_Case22_StringComparison() {
            String method = """
                public String stringComparisonGuard(String str) {
                    if (str.length() > 0) {
                        return "non-empty";
                    } else {
                        return "empty";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String comparison should transform code");
            assertCompiles(transformed, "String comparison should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String comparison should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with assignment in condition (should skip)")
        public void testGuardReversal_Case23_AssignmentInCondition() {
            String method = """
                public String assignmentInConditionGuard(int x) {
                    if ((x = getValue()) > 0) {
                        return "assigned positive";
                    } else {
                        return "assigned non-positive";
                    }
                }
                
                private int getValue() {
                    return 5;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Assignment in condition should not be transformed
            assertNoTransformation(original, transformed, "Assignment in condition");
            assertCompiles(transformed, "Assignment in condition should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Assignment in condition should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with increment in condition (should skip)")
        public void testGuardReversal_Case24_IncrementInCondition() {
            String method = """
                public String incrementInConditionGuard(int x) {
                    if (++x > 0) {
                        return "incremented positive";
                    } else {
                        return "incremented non-positive";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Increment in condition should not be transformed
            assertNoTransformation(original, transformed, "Increment in condition");
            assertCompiles(transformed, "Increment in condition should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Increment in condition should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with method call chain")
        public void testGuardReversal_Case25_MethodCallChain() {
            String method = """
                public String methodCallChainGuard(String str) {
                    if (str.trim().toUpperCase().length() > 0) {
                        return "non-empty processed";
                    } else {
                        return "empty processed";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Method call chains should be transformed if they're pure
            assertTransformationApplied(original, transformed, "Method call chain should transform code");
            assertCompiles(transformed, "Method call chain should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Method call chain should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with ternary operator in condition")
        public void testGuardReversal_Case26_TernaryInCondition() {
            String method = """
                public String ternaryInConditionGuard(int x, int y) {
                    if ((x > y ? x : y) > 0) {
                        return "max positive";
                    } else {
                        return "max non-positive";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Ternary in condition should transform code");
            assertCompiles(transformed, "Ternary in condition should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Ternary in condition should preserve semantics");
        }
        
        @Test
        @DisplayName("Guard with logical operators precedence")
        public void testGuardReversal_Case27_LogicalOperatorsPrecedence() {
            String method = """
                public String logicalOperatorsPrecedenceGuard(boolean a, boolean b, boolean c) {
                    if (a && b || c) {
                        return "logical condition true";
                    } else {
                        return "logical condition false";
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical operators precedence should transform code");
            assertCompiles(transformed, "Logical operators precedence should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical operators precedence should preserve semantics");
        }
    }
}
