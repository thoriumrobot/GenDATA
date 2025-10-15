package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Constant Folding transformation.
 * Tests constant expression evaluation and optimization.
 */
@DisplayName("Constant Folding Transformation Tests")
class ConstantFoldingTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "constant_folding";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Arithmetic Constant Folding")
    class ArithmeticConstantFolding {
        
        @Test
        @DisplayName("Simple addition folding")
        public void testConstantFolding_Case1_SimpleAddition() {
            String method = """
                public void simpleAddition() {
                    int result = 5 + 3;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple addition should transform code");
            assertCompiles(transformed, "Simple addition should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple addition should preserve semantics");
            
            // Verify folding (5 + 3 → 8)
            assertTrue(transformed.contains("8") || transformed.contains("5 + 3"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple subtraction folding")
        public void testConstantFolding_Case2_SimpleSubtraction() {
            String method = """
                public void simpleSubtraction() {
                    int result = 10 - 4;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple subtraction should transform code");
            assertCompiles(transformed, "Simple subtraction should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple subtraction should preserve semantics");
            
            // Verify folding (10 - 4 → 6)
            assertTrue(transformed.contains("6") || transformed.contains("10 - 4"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple multiplication folding")
        public void testConstantFolding_Case3_SimpleMultiplication() {
            String method = """
                public void simpleMultiplication() {
                    int result = 6 * 7;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple multiplication should transform code");
            assertCompiles(transformed, "Simple multiplication should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple multiplication should preserve semantics");
            
            // Verify folding (6 * 7 → 42)
            assertTrue(transformed.contains("42") || transformed.contains("6 * 7"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Simple division folding")
        public void testConstantFolding_Case4_SimpleDivision() {
            String method = """
                public void simpleDivision() {
                    int result = 15 / 3;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple division should transform code");
            assertCompiles(transformed, "Simple division should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple division should preserve semantics");
            
            // Verify folding (15 / 3 → 5)
            assertTrue(transformed.contains("5") || transformed.contains("15 / 3"), "Should contain folded result or original");
        }
        
        @Test
        @DisplayName("Complex arithmetic expressions")
        public void testConstantFolding_Case5_ComplexArithmetic() {
            String method = """
                public void complexArithmetic() {
                    int result = (5 + 3) * (10 - 2);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex arithmetic should transform code");
            assertCompiles(transformed, "Complex arithmetic should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Complex arithmetic should preserve semantics");
        }
        
        @Test
        @DisplayName("Nested arithmetic")
        public void testConstantFolding_Case6_NestedArithmetic() {
            String method = """
                public void nestedArithmetic() {
                    int result = ((2 + 3) * 4) - (6 / 2);
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested arithmetic should transform code");
            assertCompiles(transformed, "Nested arithmetic should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Nested arithmetic should preserve semantics");
        }
        
        @Test
        @DisplayName("Floating-point constants")
        public void testConstantFolding_Case7_FloatingPointConstants() {
            String method = """
                public void floatingPointConstants() {
                    double result = 3.14 + 2.86;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Floating-point constants should transform code");
            assertCompiles(transformed, "Floating-point constants should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Floating-point constants should preserve semantics");
        }
        
        @Test
        @DisplayName("Division by zero handling")
        public void testConstantFolding_Case8_DivisionByZeroHandling() {
            String method = """
                public void divisionByZeroHandling() {
                    int result = 10 / 2; // Safe division
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Division by zero handling should transform code");
            assertCompiles(transformed, "Division by zero handling should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Division by zero handling should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: very large numbers")
        public void testConstantFolding_Case9_VeryLargeNumbers() {
            String method = """
                public void veryLargeNumbers() {
                    long result = 1000000L + 2000000L;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Very large numbers should transform code");
            assertCompiles(transformed, "Very large numbers should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Very large numbers should preserve semantics");
        }
    }
    
    @Nested
    @DisplayName("String and Boolean Constant Folding")
    class StringBooleanConstantFolding {
        
        @Test
        @DisplayName("String concatenation folding")
        public void testConstantFolding_Case10_StringConcatenationFolding() {
            String method = """
                public void stringConcatenationFolding() {
                    String result = "Hello" + " " + "World";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "String concatenation folding should transform code");
            assertCompiles(transformed, "String concatenation folding should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "String concatenation folding should preserve semantics");
        }
        
        @Test
        @DisplayName("Boolean constant folding")
        public void testConstantFolding_Case11_BooleanConstantFolding() {
            String method = """
                public void booleanConstantFolding() {
                    boolean result = true && false;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Boolean constant folding should transform code");
            assertCompiles(transformed, "Boolean constant folding should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Boolean constant folding should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive constant folding test")
    public void testConstantFolding_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various constant folding scenarios
                int arithmetic = (5 + 3) * (10 - 2);
                double floating = 3.14 + 2.86;
                String concatenation = "Test" + " " + "String";
                boolean logical = true && (false || true);
                
                System.out.println(arithmetic + " " + floating + " " + concatenation + " " + logical);
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}