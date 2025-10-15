package cfwr.jdt.transformations.random;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Dead Code Insertion transformation.
 * Tests insertion of harmless dead code statements.
 */
@DisplayName("Dead Code Insertion Transformation Tests")
class DeadCodeInsertionTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "dead_code_insertion";
    private static final String MODE = "random";
    
    @Nested
    @DisplayName("Dead Code Insertion Operations")
    class DeadCodeInsertionOperations {
        
        @Test
        @DisplayName("Insert numeric literal")
        public void testDeadCodeInsertion_Case1_InsertNumericLiteral() {
            String method = """
                public void insertNumericLiteral() {
                    int result = a + b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert numeric literal should transform code");
            assertCompiles(transformed, "Insert numeric literal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert numeric literal should preserve semantics");
            
            // Verify dead code insertion (should contain 0; or similar)
            assertTrue(transformed.contains("0;") || transformed.contains("1;") || transformed.contains("2;"), "Should contain dead numeric literal");
        }
        
        @Test
        @DisplayName("Insert boolean literal")
        public void testDeadCodeInsertion_Case2_InsertBooleanLiteral() {
            String method = """
                public void insertBooleanLiteral() {
                    boolean result = a > b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert boolean literal should transform code");
            assertCompiles(transformed, "Insert boolean literal should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert boolean literal should preserve semantics");
            
            // Verify dead code insertion (should contain false; or true;)
            assertTrue(transformed.contains("false;") || transformed.contains("true;"), "Should contain dead boolean literal");
        }
        
        @Test
        @DisplayName("Insert empty string")
        public void testDeadCodeInsertion_Case3_InsertEmptyString() {
            String method = """
                public void insertEmptyString() {
                    String result = "test";
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert empty string should transform code");
            assertCompiles(transformed, "Insert empty string should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert empty string should preserve semantics");
            
            // Verify dead code insertion (should contain "";)
            assertTrue(transformed.contains("\"\";"), "Should contain dead empty string");
        }
        
        @Test
        @DisplayName("Insert in method body")
        public void testDeadCodeInsertion_Case4_InsertInMethodBody() {
            String method = """
                public void insertInMethodBody() {
                    int x = 10;
                    int y = 20;
                    int result = x + y;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in method body should transform code");
            assertCompiles(transformed, "Insert in method body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in method body should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert in if block")
        public void testDeadCodeInsertion_Case5_InsertInIfBlock() {
            String method = """
                public void insertInIfBlock() {
                    if (condition) {
                        System.out.println("True");
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in if block should transform code");
            assertCompiles(transformed, "Insert in if block should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in if block should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert in loop body")
        public void testDeadCodeInsertion_Case6_InsertInLoopBody() {
            String method = """
                public void insertInLoopBody() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert in loop body should transform code");
            assertCompiles(transformed, "Insert in loop body should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert in loop body should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert multiple dead statements")
        public void testDeadCodeInsertion_Case7_InsertMultipleDeadStatements() {
            String method = """
                public void insertMultipleDeadStatements() {
                    int result = 0;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert multiple dead statements should transform code");
            assertCompiles(transformed, "Insert multiple dead statements should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert multiple dead statements should preserve semantics");
        }
        
        @Test
        @DisplayName("Insert at different positions")
        public void testDeadCodeInsertion_Case8_InsertAtDifferentPositions() {
            String method = """
                public void insertAtDifferentPositions() {
                    int a = 1;
                    int b = 2;
                    int result = a + b;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Insert at different positions should transform code");
            assertCompiles(transformed, "Insert at different positions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Insert at different positions should preserve semantics");
        }
        
        @Test
        @DisplayName("Verify no semantic impact")
        public void testDeadCodeInsertion_Case9_VerifyNoSemanticImpact() {
            String method = """
                public void verifyNoSemanticImpact() {
                    int result = calculate();
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Verify no semantic impact should transform code");
            assertCompiles(transformed, "Verify no semantic impact should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Verify no semantic impact should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: empty blocks")
        public void testDeadCodeInsertion_Case10_EmptyBlocks() {
            String method = """
                public void emptyBlocks() {
                    if (condition) {
                        // Empty block
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Empty blocks should transform code");
            assertCompiles(transformed, "Empty blocks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Empty blocks should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single-statement blocks")
        public void testDeadCodeInsertion_Case11_SingleStatementBlocks() {
            String method = """
                public void singleStatementBlocks() {
                    if (condition) {
                        return;
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single-statement blocks should transform code");
            assertCompiles(transformed, "Single-statement blocks should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single-statement blocks should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive dead code insertion test")
    public void testDeadCodeInsertion_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various dead code insertion scenarios
                int a = 1;
                if (condition) {
                    int b = 2;
                }
                for (int i = 0; i < 5; i++) {
                    System.out.println(i);
                }
                return a;
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}