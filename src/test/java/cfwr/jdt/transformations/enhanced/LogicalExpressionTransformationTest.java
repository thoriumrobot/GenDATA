package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Logical Expression transformation.
 * Tests De Morgan's laws, logical operator transformations, and boolean expression manipulations.
 */
@DisplayName("Logical Expression Transformation Tests")
class LogicalExpressionTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "logical_expression";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("De Morgan's Laws")
    class DeMorgansLaws {
        
        @Test
        @DisplayName("De Morgan's law: !(a && b) → !a || !b")
        public void testLogicalExpression_Case1_DeMorganAndToOr() {
            String method = """
                public boolean deMorganAndToOr(boolean a, boolean b) {
                    return !(a && b);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "De Morgan AND to OR should transform code");
            assertCompiles(transformed, "De Morgan AND to OR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "De Morgan AND to OR should preserve semantics");
            
            // Verify De Morgan's law application
            assertTrue(transformed.contains("!a || !b") || transformed.contains("(!a) || (!b)"), 
                "Should apply De Morgan's law: !(a && b) → !a || !b");
        }
        
        @Test
        @DisplayName("De Morgan's law: !(a || b) → !a && !b")
        public void testLogicalExpression_Case2_DeMorganOrToAnd() {
            String method = """
                public boolean deMorganOrToAnd(boolean a, boolean b) {
                    return !(a || b);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "De Morgan OR to AND should transform code");
            assertCompiles(transformed, "De Morgan OR to AND should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "De Morgan OR to AND should preserve semantics");
            
            // Verify De Morgan's law application
            assertTrue(transformed.contains("!a && !b") || transformed.contains("(!a) && (!b)"), 
                "Should apply De Morgan's law: !(a || b) → !a && !b");
        }
        
        @Test
        @DisplayName("De Morgan's law with complex expressions")
        public void testLogicalExpression_Case3_DeMorganComplexExpressions() {
            String method = """
                public boolean deMorganComplexExpressions(boolean a, boolean b, boolean c) {
                    return !((a && b) || (c && !a));
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "De Morgan complex expressions should transform code");
            assertCompiles(transformed, "De Morgan complex expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "De Morgan complex expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("De Morgan's law with nested parentheses")
        public void testLogicalExpression_Case4_DeMorganNestedParentheses() {
            String method = """
                public boolean deMorganNestedParentheses(boolean a, boolean b, boolean c) {
                    return !(((a && b) || c) && (!b || a));
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "De Morgan nested parentheses should transform code");
            assertCompiles(transformed, "De Morgan nested parentheses should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "De Morgan nested parentheses should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Double Negation Elimination")
    class DoubleNegationElimination {
        
        @Test
        @DisplayName("Double negation elimination: !!a → a")
        public void testLogicalExpression_Case5_DoubleNegationElimination() {
            String method = """
                public boolean doubleNegationElimination(boolean a) {
                    return !!a;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Double negation elimination should transform code");
            assertCompiles(transformed, "Double negation elimination should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Double negation elimination should preserve semantics");
            
            // Should eliminate double negation
            assertFalse(transformed.contains("!!"), "Should eliminate double negation");
        }
        
        @Test
        @DisplayName("Triple negation: !!!a → !a")
        public void testLogicalExpression_Case6_TripleNegation() {
            String method = """
                public boolean tripleNegation(boolean a) {
                    return !!!a;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Triple negation should transform code");
            assertCompiles(transformed, "Triple negation should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Triple negation should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Logical Operator Commutativity")
    class LogicalOperatorCommutativity {
        
        @Test
        @DisplayName("AND commutativity: a && b → b && a")
        public void testLogicalExpression_Case7_AndCommutativity() {
            String method = """
                public boolean andCommutativity(boolean a, boolean b) {
                    return a && b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "AND commutativity should transform code");
            assertCompiles(transformed, "AND commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "AND commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("OR commutativity: a || b → b || a")
        public void testLogicalExpression_Case8_OrCommutativity() {
            String method = """
                public boolean orCommutativity(boolean a, boolean b) {
                    return a || b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "OR commutativity should transform code");
            assertCompiles(transformed, "OR commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "OR commutativity should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Boolean Literal Simplification")
    class BooleanLiteralSimplification {
        
        @Test
        @DisplayName("AND with true: a && true → a")
        public void testLogicalExpression_Case9_AndWithTrue() {
            String method = """
                public boolean andWithTrue(boolean a) {
                    return a && true;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "AND with true should transform code");
            assertCompiles(transformed, "AND with true should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "AND with true should preserve semantics");
        }
        
        @Test
        @DisplayName("AND with false: a && false → false")
        public void testLogicalExpression_Case10_AndWithFalse() {
            String method = """
                public boolean andWithFalse(boolean a) {
                    return a && false;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "AND with false should transform code");
            assertCompiles(transformed, "AND with false should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "AND with false should preserve semantics");
        }
        
        @Test
        @DisplayName("OR with true: a || true → true")
        public void testLogicalExpression_Case11_OrWithTrue() {
            String method = """
                public boolean orWithTrue(boolean a) {
                    return a || true;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "OR with true should transform code");
            assertCompiles(transformed, "OR with true should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "OR with true should preserve semantics");
        }
        
        @Test
        @DisplayName("OR with false: a || false → a")
        public void testLogicalExpression_Case12_OrWithFalse() {
            String method = """
                public boolean orWithFalse(boolean a) {
                    return a || false;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "OR with false should transform code");
            assertCompiles(transformed, "OR with false should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "OR with false should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Logical Expressions")
    class ComplexLogicalExpressions {
        
        @Test
        @DisplayName("Mixed logical operators with parentheses")
        public void testLogicalExpression_Case13_MixedLogicalOperators() {
            String method = """
                public boolean mixedLogicalOperators(boolean a, boolean b, boolean c, boolean d) {
                    return (a && b) || (c && d) && (!a || b);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Mixed logical operators should transform code");
            assertCompiles(transformed, "Mixed logical operators should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Mixed logical operators should preserve semantics");
        }
        
        @Test
        @DisplayName("Logical expression with comparison operators")
        public void testLogicalExpression_Case14_LogicalWithComparisons() {
            String method = """
                public boolean logicalWithComparisons(int x, int y, int z) {
                    return (x > 0 && y < 10) || (z == 0 && x != y);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical with comparisons should transform code");
            assertCompiles(transformed, "Logical with comparisons should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical with comparisons should preserve semantics");
        }
        
        @Test
        @DisplayName("Logical expression with method calls")
        public void testLogicalExpression_Case15_LogicalWithMethodCalls() {
            String method = """
                public boolean logicalWithMethodCalls(String str, int value) {
                    return (str != null && !str.isEmpty()) || (value > 0 && isValid(value));
                }
                
                private boolean isValid(int v) {
                    return v < 100;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical with method calls should transform code");
            assertCompiles(transformed, "Logical with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical with method calls should preserve semantics");
        }
        
        @Test
        @DisplayName("Logical expression with ternary operators")
        public void testLogicalExpression_Case16_LogicalWithTernary() {
            String method = """
                public boolean logicalWithTernary(boolean a, boolean b, int x) {
                    return (a && b) || (x > 0 ? true : false);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical with ternary should transform code");
            assertCompiles(transformed, "Logical with ternary should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical with ternary should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Short-Circuit Evaluation")
    class ShortCircuitEvaluation {
        
        @Test
        @DisplayName("Short-circuit AND evaluation")
        public void testLogicalExpression_Case17_ShortCircuitAnd() {
            String method = """
                public boolean shortCircuitAnd(boolean a) {
                    return a && getBooleanValue();
                }
                
                private boolean getBooleanValue() {
                    System.out.println("getBooleanValue called");
                    return true;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Short-circuit AND should transform code");
            assertCompiles(transformed, "Short-circuit AND should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Short-circuit AND should preserve semantics");
        }
        
        @Test
        @DisplayName("Short-circuit OR evaluation")
        public void testLogicalExpression_Case18_ShortCircuitOr() {
            String method = """
                public boolean shortCircuitOr(boolean a) {
                    return a || getBooleanValue();
                }
                
                private boolean getBooleanValue() {
                    System.out.println("getBooleanValue called");
                    return false;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Short-circuit OR should transform code");
            assertCompiles(transformed, "Short-circuit OR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Short-circuit OR should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Single boolean variable")
        public void testLogicalExpression_Case19_SingleBooleanVariable() {
            String method = """
                public boolean singleBooleanVariable(boolean a) {
                    return a;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Single variable should not be transformed
            assertNoTransformation(original, transformed, "Single boolean variable");
            assertCompiles(transformed, "Single boolean variable should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single boolean variable should preserve semantics");
        }
        
        @Test
        @DisplayName("Logical expression with null checks")
        public void testLogicalExpression_Case20_LogicalWithNullChecks() {
            String method = """
                public boolean logicalWithNullChecks(String str1, String str2) {
                    return (str1 != null && str2 != null) || (str1 == null && str2 == null);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical with null checks should transform code");
            assertCompiles(transformed, "Logical with null checks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical with null checks should preserve semantics");
        }
        
        @Test
        @DisplayName("Logical expression with instanceof checks")
        public void testLogicalExpression_Case21_LogicalWithInstanceof() {
            String method = """
                public boolean logicalWithInstanceof(Object obj1, Object obj2) {
                    return (obj1 instanceof String && obj2 instanceof String) || 
                           (obj1 instanceof Integer && obj2 instanceof Integer);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Logical with instanceof should transform code");
            assertCompiles(transformed, "Logical with instanceof should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Logical with instanceof should preserve semantics");
        }
        
        @Test
        @DisplayName("Very complex nested logical expression")
        public void testLogicalExpression_Case22_VeryComplexNestedExpression() {
            String method = """
                public boolean veryComplexNestedExpression(boolean a, boolean b, boolean c, boolean d, boolean e) {
                    return (((a && b) || (c && d)) && (!a || (b && c))) || 
                           ((d && e) && (!b || (c && d))) && (a || (!e && b));
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Very complex nested expression should transform code");
            assertCompiles(transformed, "Very complex nested expression should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Very complex nested expression should preserve semantics");
        }
    }
}
