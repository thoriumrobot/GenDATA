package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Mathematical Expression transformation.
 * Tests mathematical property applications like commutativity, identity elements, and associativity.
 */
@DisplayName("Mathematical Expression Transformation Tests")
class MathematicalExpressionTransformationTest extends TransformationTestBase {    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "mathematical_expression";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Commutativity Transformations")
    class CommutativityTransformations {
        
        @Test
        @DisplayName("Addition commutativity (a+b → b+a)")
        public void testMathematicalExpression_Case1_AdditionCommutativity() {
            String method = """
                public int additionCommutativity(int a, int b) {
                    return a + b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Addition commutativity should transform code");
            assertCompiles(transformed, "Addition commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Addition commutativity should preserve semantics");
            
            // Verify commutativity was applied
            assertTrue(transformed.contains("b + a") || transformed.contains("a + b"), 
                "Should contain commutative addition");
        }
        
        @Test
        @DisplayName("Multiplication commutativity")
        public void testMathematicalExpression_Case2_MultiplicationCommutativity() {
            String method = """
                public int multiplicationCommutativity(int a, int b) {
                    return a * b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiplication commutativity should transform code");
            assertCompiles(transformed, "Multiplication commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiplication commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise OR commutativity")
        public void testMathematicalExpression_Case3_BitwiseOrCommutativity() {
            String method = """
                public int bitwiseOrCommutativity(int a, int b) {
                    return a | b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise OR commutativity should transform code");
            assertCompiles(transformed, "Bitwise OR commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise OR commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise XOR commutativity")
        public void testMathematicalExpression_Case4_BitwiseXorCommutativity() {
            String method = """
                public int bitwiseXorCommutativity(int a, int b) {
                    return a ^ b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise XOR commutativity should transform code");
            assertCompiles(transformed, "Bitwise XOR commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise XOR commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise AND commutativity")
        public void testMathematicalExpression_Case5_BitwiseAndCommutativity() {
            String method = """
                public int bitwiseAndCommutativity(int a, int b) {
                    return a & b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise AND commutativity should transform code");
            assertCompiles(transformed, "Bitwise AND commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise AND commutativity should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Identity Element Transformations")
    class IdentityElementTransformations {
        
        @Test
        @DisplayName("Addition identity element (x+0 → x)")
        public void testMathematicalExpression_Case6_AdditionIdentity() {
            String method = """
                public int additionIdentity(int x) {
                    return x + 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Addition identity should transform code");
            assertCompiles(transformed, "Addition identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Addition identity should preserve semantics");
            
            // Should simplify to just x
            assertTrue(transformed.contains("return x;") || transformed.contains("return (x);"), 
                "Should simplify x+0 to x");
        }
        
        @Test
        @DisplayName("Multiplication identity element (x*1 → x)")
        public void testMathematicalExpression_Case7_MultiplicationIdentity() {
            String method = """
                public int multiplicationIdentity(int x) {
                    return x * 1;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiplication identity should transform code");
            assertCompiles(transformed, "Multiplication identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiplication identity should preserve semantics");
        }
        
        @Test
        @DisplayName("Subtraction identity element (x-0 → x)")
        public void testMathematicalExpression_Case8_SubtractionIdentity() {
            String method = """
                public int subtractionIdentity(int x) {
                    return x - 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Subtraction identity should transform code");
            assertCompiles(transformed, "Subtraction identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Subtraction identity should preserve semantics");
        }
        
        @Test
        @DisplayName("Division identity element (x/1 → x)")
        public void testMathematicalExpression_Case9_DivisionIdentity() {
            String method = """
                public int divisionIdentity(int x) {
                    return x / 1;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Division identity should transform code");
            assertCompiles(transformed, "Division identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Division identity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise OR identity element (x|0 → x)")
        public void testMathematicalExpression_Case10_BitwiseOrIdentity() {
            String method = """
                public int bitwiseOrIdentity(int x) {
                    return x | 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise OR identity should transform code");
            assertCompiles(transformed, "Bitwise OR identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise OR identity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise XOR identity element (x^0 → x)")
        public void testMathematicalExpression_Case11_BitwiseXorIdentity() {
            String method = """
                public int bitwiseXorIdentity(int x) {
                    return x ^ 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise XOR identity should transform code");
            assertCompiles(transformed, "Bitwise XOR identity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise XOR identity should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Zero Element Transformations")
    class ZeroElementTransformations {
        
        @Test
        @DisplayName("Multiplication by zero (x*0 → 0)")
        public void testMathematicalExpression_Case12_MultiplicationByZero() {
            String method = """
                public int multiplicationByZero(int x) {
                    return x * 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiplication by zero should transform code");
            assertCompiles(transformed, "Multiplication by zero should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiplication by zero should preserve semantics");
            
            // Should simplify to 0
            assertTrue(transformed.contains("return 0;") || transformed.contains("return (0);"), 
                "Should simplify x*0 to 0");
        }
        
        @Test
        @DisplayName("Bitwise AND with zero (x&0 → 0)")
        public void testMathematicalExpression_Case13_BitwiseAndZero() {
            String method = """
                public int bitwiseAndZero(int x) {
                    return x & 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise AND zero should transform code");
            assertCompiles(transformed, "Bitwise AND zero should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise AND zero should preserve semantics");
        }
        
        @Test
        @DisplayName("Left shift by zero (x<<0 → x)")
        public void testMathematicalExpression_Case14_LeftShiftZero() {
            String method = """
                public int leftShiftZero(int x) {
                    return x << 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Left shift zero should transform code");
            assertCompiles(transformed, "Left shift zero should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Left shift zero should preserve semantics");
        }
        
        @Test
        @DisplayName("Right shift by zero (x>>0 → x)")
        public void testMathematicalExpression_Case15_RightShiftZero() {
            String method = """
                public int rightShiftZero(int x) {
                    return x >> 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Right shift zero should transform code");
            assertCompiles(transformed, "Right shift zero should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Right shift zero should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Associativity Transformations")
    class AssociativityTransformations {
        
        @Test
        @DisplayName("Addition associativity")
        public void testMathematicalExpression_Case16_AdditionAssociativity() {
            String method = """
                public int additionAssociativity(int a, int b, int c) {
                    return a + b + c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Addition associativity should transform code");
            assertCompiles(transformed, "Addition associativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Addition associativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Multiplication associativity")
        public void testMathematicalExpression_Case17_MultiplicationAssociativity() {
            String method = """
                public int multiplicationAssociativity(int a, int b, int c) {
                    return a * b * c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiplication associativity should transform code");
            assertCompiles(transformed, "Multiplication associativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiplication associativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise OR associativity")
        public void testMathematicalExpression_Case18_BitwiseOrAssociativity() {
            String method = """
                public int bitwiseOrAssociativity(int a, int b, int c) {
                    return a | b | c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise OR associativity should transform code");
            assertCompiles(transformed, "Bitwise OR associativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise OR associativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise AND associativity")
        public void testMathematicalExpression_Case19_BitwiseAndAssociativity() {
            String method = """
                public int bitwiseAndAssociativity(int a, int b, int c) {
                    return a & b & c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Bitwise AND associativity should transform code");
            assertCompiles(transformed, "Bitwise AND associativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Bitwise AND associativity should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Complex Mathematical Expressions")
    class ComplexMathematicalExpressions {
        
        @Test
        @DisplayName("Complex nested expressions")
        public void testMathematicalExpression_Case20_ComplexNestedExpressions() {
            String method = """
                public int complexNestedExpressions(int a, int b, int c, int d) {
                    return (a + b) * (c - d) + (a * b) - (c / d);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex nested expressions should transform code");
            assertCompiles(transformed, "Complex nested expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex nested expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Mixed operators expression")
        public void testMathematicalExpression_Case21_MixedOperatorsExpression() {
            String method = """
                public int mixedOperatorsExpression(int a, int b, int c) {
                    return a + b * c - a / b + a % c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Mixed operators expression should transform code");
            assertCompiles(transformed, "Mixed operators expression should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Mixed operators expression should preserve semantics");
        }
        
        @Test
        @DisplayName("Expression with parentheses")
        public void testMathematicalExpression_Case22_ExpressionWithParentheses() {
            String method = """
                public int expressionWithParentheses(int a, int b, int c) {
                    return (a + (b * c)) - ((a / b) + c);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Expression with parentheses should transform code");
            assertCompiles(transformed, "Expression with parentheses should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Expression with parentheses should preserve semantics");
        }
        
        @Test
        @DisplayName("Expression with method calls")
        public void testMathematicalExpression_Case23_ExpressionWithMethodCalls() {
            String method = """
                public int expressionWithMethodCalls(int a, int b) {
                    return a + getValue() * b - calculate(a, b);
                }
                
                private int getValue() {
                    return 5;
                }
                
                private int calculate(int x, int y) {
                    return x * y;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Expression with method calls should transform code");
            assertCompiles(transformed, "Expression with method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Expression with method calls should preserve semantics");
        }
        
        @Test
        @DisplayName("Expression with field access")
        public void testMathematicalExpression_Case24_ExpressionWithFieldAccess() {
            String method = """
                private int multiplier = 2;
                private int offset = 10;
                
                public int expressionWithFieldAccess(int a, int b) {
                    return a * multiplier + b - offset;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Expression with field access should transform code");
            assertCompiles(transformed, "Expression with field access should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Expression with field access should preserve semantics");
        }
        
        @Test
        @DisplayName("Expression with array access")
        public void testMathematicalExpression_Case25_ExpressionWithArrayAccess() {
            String method = """
                public int expressionWithArrayAccess(int[] array, int index) {
                    return array[index] + array[index + 1] * array[index - 1];
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Expression with array access should transform code");
            assertCompiles(transformed, "Expression with array access should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Expression with array access should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Floating-Point Expressions")
    class FloatingPointExpressions {
        
        @Test
        @DisplayName("Floating-point addition commutativity")
        public void testMathematicalExpression_Case26_FloatingPointAdditionCommutativity() {
            String method = """
                public double floatingPointAdditionCommutativity(double a, double b) {
                    return a + b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point addition commutativity should transform code");
            assertCompiles(transformed, "Floating-point addition commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point addition commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point multiplication commutativity")
        public void testMathematicalExpression_Case27_FloatingPointMultiplicationCommutativity() {
            String method = """
                public double floatingPointMultiplicationCommutativity(double a, double b) {
                    return a * b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point multiplication commutativity should transform code");
            assertCompiles(transformed, "Floating-point multiplication commutativity should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point multiplication commutativity should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point identity elements")
        public void testMathematicalExpression_Case28_FloatingPointIdentityElements() {
            String method = """
                public double floatingPointIdentityElements(double x) {
                    return x + 0.0 + x * 1.0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point identity elements should transform code");
            assertCompiles(transformed, "Floating-point identity elements should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point identity elements should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point division")
        public void testMathematicalExpression_Case29_FloatingPointDivision() {
            String method = """
                public double floatingPointDivision(double a, double b) {
                    return a / b;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point division should transform code");
            assertCompiles(transformed, "Floating-point division should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point division should preserve semantics");
        }
        
        @Test
        @DisplayName("Mixed integer and floating-point operations")
        public void testMathematicalExpression_Case30_MixedIntegerFloatingPoint() {
            String method = """
                public double mixedIntegerFloatingPoint(int a, double b) {
                    return a + b * 2.0 - a / 1.0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Mixed integer and floating-point operations should transform code");
            assertCompiles(transformed, "Mixed integer and floating-point operations should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Mixed integer and floating-point operations should preserve semantics");
        }
    }
}
