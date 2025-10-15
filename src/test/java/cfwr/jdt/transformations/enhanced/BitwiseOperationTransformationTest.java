package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Bitwise Operation transformation.
 * Tests bitwise AND, OR, XOR, and NOT operations with various patterns.
 */
@DisplayName("Bitwise Operation Transformation Tests")
class BitwiseOperationTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "bitwise_operation";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Bitwise AND Operations")
    class BitwiseAndOperations {
        
        @Test
        @DisplayName("Simple bitwise AND commutativity")
        public void testBitwiseOperation_Case1_SimpleAndCommutativity() {
            String method = """
                public void simpleAnd() {
                    int a = 5, b = 3;
                    int result = a & b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple AND should transform code");
            assertCompiles(transformed, "Simple AND should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple AND should preserve semantics");
            
            // Verify commutativity (a & b ↔ b & a)
            assertTrue(transformed.contains("&"), "Should contain bitwise AND operation");
        }
        
        @Test
        @DisplayName("Complex bitwise AND with parentheses")
        public void testBitwiseOperation_Case2_ComplexAndWithParentheses() {
            String method = """
                public void complexAnd() {
                    int a = 5, b = 3, c = 7;
                    int result = (a & b) & c;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex AND should transform code");
            assertCompiles(transformed, "Complex AND should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex AND should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise AND with constants")
        public void testBitwiseOperation_Case3_AndWithConstants() {
            String method = """
                public void andWithConstants() {
                    int result = 0xFF & 0x0F;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "AND with constants should transform code");
            assertCompiles(transformed, "AND with constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "AND with constants should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Bitwise OR Operations")
    class BitwiseOrOperations {
        
        @Test
        @DisplayName("Simple bitwise OR commutativity")
        public void testBitwiseOperation_Case4_SimpleOrCommutativity() {
            String method = """
                public void simpleOr() {
                    int a = 5, b = 3;
                    int result = a | b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple OR should transform code");
            assertCompiles(transformed, "Simple OR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple OR should preserve semantics");
            
            // Verify commutativity (a | b ↔ b | a)
            assertTrue(transformed.contains("|"), "Should contain bitwise OR operation");
        }
        
        @Test
        @DisplayName("Nested bitwise OR operations")
        public void testBitwiseOperation_Case5_NestedOrOperations() {
            String method = """
                public void nestedOr() {
                    int a = 5, b = 3, c = 7;
                    int result = a | (b | c);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested OR should transform code");
            assertCompiles(transformed, "Nested OR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested OR should preserve semantics");
        }
        
        @Test
        @DisplayName("Bitwise OR with mixed types")
        public void testBitwiseOperation_Case6_OrWithMixedTypes() {
            String method = """
                public void orWithMixedTypes() {
                    int a = 5, b = 3;
                    long result = (long)a | b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "OR with mixed types should transform code");
            assertCompiles(transformed, "OR with mixed types should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "OR with mixed types should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Bitwise XOR Operations")
    class BitwiseXorOperations {
        
        @Test
        @DisplayName("Simple bitwise XOR commutativity")
        public void testBitwiseOperation_Case7_SimpleXorCommutativity() {
            String method = """
                public void simpleXor() {
                    int a = 5, b = 3;
                    int result = a ^ b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple XOR should transform code");
            assertCompiles(transformed, "Simple XOR should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple XOR should preserve semantics");
            
            // Verify commutativity (a ^ b ↔ b ^ a)
            assertTrue(transformed.contains("^"), "Should contain bitwise XOR operation");
        }
        
        @Test
        @DisplayName("XOR with multiple operands")
        public void testBitwiseOperation_Case8_XorWithMultipleOperands() {
            String method = """
                public void xorWithMultiple() {
                    int a = 5, b = 3, c = 7;
                    int result = a ^ b ^ c;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "XOR with multiple operands should transform code");
            assertCompiles(transformed, "XOR with multiple operands should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "XOR with multiple operands should preserve semantics");
        }
        
        @Test
        @DisplayName("XOR with expressions")
        public void testBitwiseOperation_Case9_XorWithExpressions() {
            String method = """
                public void xorWithExpressions() {
                    int a = 5, b = 3, c = 7, d = 2;
                    int result = (a + b) ^ (c - d);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "XOR with expressions should transform code");
            assertCompiles(transformed, "XOR with expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "XOR with expressions should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Bitwise NOT Operations")
    class BitwiseNotOperations {
        
        @Test
        @DisplayName("Simple bitwise NOT transformation")
        public void testBitwiseOperation_Case10_SimpleNotTransformation() {
            String method = """
                public void simpleNot() {
                    int x = 5;
                    int result = ~x;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple NOT should transform code");
            assertCompiles(transformed, "Simple NOT should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple NOT should preserve semantics");
            
            // Verify NOT transformation (~x → (-x) - 1)
            assertTrue(transformed.contains("~") || transformed.contains("-"), "Should contain NOT operation or negation");
        }
        
        @Test
        @DisplayName("NOT with complex expression")
        public void testBitwiseOperation_Case11_NotWithComplexExpression() {
            String method = """
                public void notWithComplex() {
                    int a = 3, b = 7;
                    int result = ~(a + b);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "NOT with complex expression should transform code");
            assertCompiles(transformed, "NOT with complex expression should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "NOT with complex expression should preserve semantics");
        }
        
        @Test
        @DisplayName("NOT with constants")
        public void testBitwiseOperation_Case12_NotWithConstants() {
            String method = """
                public void notWithConstants() {
                    int result = ~0xFF;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "NOT with constants should transform code");
            assertCompiles(transformed, "NOT with constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "NOT with constants should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("Mixed Bitwise Operations")
    class MixedBitwiseOperations {
        
        @Test
        @DisplayName("Mixed bitwise and arithmetic")
        public void testBitwiseOperation_Case13_MixedBitwiseAndArithmetic() {
            String method = """
                public void mixedBitwise() {
                    int a = 5, b = 3, c = 7, d = 2;
                    int result = (a & b) + (c | d);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Mixed bitwise should transform code");
            assertCompiles(transformed, "Mixed bitwise should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Mixed bitwise should preserve semantics");
        }
        
        @Test
        @DisplayName("Multiple bitwise operations")
        public void testBitwiseOperation_Case14_MultipleBitwiseOperations() {
            String method = """
                public void multipleBitwise() {
                    int a = 5, b = 3, c = 7, d = 2, e = 9, f = 4;
                    int result1 = a & b;
                    int result2 = c | d;
                    int result3 = e ^ f;
                    System.out.println(result1 + result2 + result3);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple bitwise should transform code");
            assertCompiles(transformed, "Multiple bitwise should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiple bitwise should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single operand")
        public void testBitwiseOperation_Case15_SingleOperand() {
            String method = """
                public void singleOperand() {
                    int a = 10;
                    int result = ~a;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single operand should transform code");
            assertCompiles(transformed, "Single operand should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single operand should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: constants only")
        public void testBitwiseOperation_Case16_ConstantsOnly() {
            String method = """
                public void constantsOnly() {
                    int result = 0xFF & 0x0F;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Constants only should transform code");
            assertCompiles(transformed, "Constants only should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Constants only should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive bitwise operation test")
    public void testBitwiseOperation_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                int a = 0xFF, b = 0x0F, c = 0x33, d = 0x55, e = 0xAA;
                
                // Test all bitwise operations
                int andResult = a & b;
                int orResult = b | c;
                int xorResult = a ^ c;
                int notResult = ~a;
                
                // Complex expression
                int complex = (a & b) | (c ^ d) & (~e);
                
                System.out.println(andResult + orResult + xorResult + notResult + complex);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}
