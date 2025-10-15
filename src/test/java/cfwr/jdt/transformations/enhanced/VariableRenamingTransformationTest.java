package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Variable Renaming transformation.
 * Tests renaming of local variables with generated names.
 */
@DisplayName("Variable Renaming Transformation Tests")
class VariableRenamingTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "variable_renaming";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("Simple Variable Renaming")
    class SimpleVariableRenaming {
        
        @Test
        @DisplayName("Simple variable renaming")
        public void testVariableRenaming_Case1_SimpleVariableRenaming() {
            String method = """
                public void simpleVariableRenaming() {
                    int count = 0;
                    System.out.println(count);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple variable renaming should transform code");
            assertCompiles(transformed, "Simple variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple variable renaming should preserve semantics");
            
            // Verify variable renaming (should contain new variable name)
            assertFalse(transformed.contains("count") || transformed.contains("newCount") || transformed.contains("var1"), "Should contain renamed variable");
        }
        
        @Test
        @DisplayName("Local variable renaming")
        public void testVariableRenaming_Case2_LocalVariableRenaming() {
            String method = """
                public void localVariableRenaming() {
                    int temp = 10;
                    int result = temp + 5;
                    return result;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Local variable renaming should transform code");
            assertCompiles(transformed, "Local variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Local variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Parameter renaming")
        public void testVariableRenaming_Case3_ParameterRenaming() {
            String method = """
                public void parameterRenaming(int value, String name) {
                    System.out.println(value + " " + name);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Parameter renaming should transform code");
            assertCompiles(transformed, "Parameter renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Parameter renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Loop variable renaming")
        public void testVariableRenaming_Case4_LoopVariableRenaming() {
            String method = """
                public void loopVariableRenaming() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Loop variable renaming should transform code");
            assertCompiles(transformed, "Loop variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Loop variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Multiple variable renaming")
        public void testVariableRenaming_Case5_MultipleVariableRenaming() {
            String method = """
                public void multipleVariableRenaming() {
                    int a = 1;
                    int b = 2;
                    int c = a + b;
                    System.out.println(c);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple variable renaming should transform code");
            assertCompiles(transformed, "Multiple variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Multiple variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Scoped variable renaming")
        public void testVariableRenaming_Case6_ScopedVariableRenaming() {
            String method = """
                public void scopedVariableRenaming() {
                    int outer = 10;
                    if (condition) {
                        int inner = 20;
                        System.out.println(outer + inner);
                    }
                    System.out.println(outer);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Scoped variable renaming should transform code");
            assertCompiles(transformed, "Scoped variable renaming should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Scoped variable renaming should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable shadowing")
        public void testVariableRenaming_Case7_VariableShadowing() {
            String method = """
                public void variableShadowing() {
                    int x = 10;
                    if (condition) {
                        int x = 20; // Shadowing
                        System.out.println(x);
                    }
                    System.out.println(x);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable shadowing should transform code");
            assertCompiles(transformed, "Variable shadowing should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable shadowing should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable in expressions")
        public void testVariableRenaming_Case8_VariableInExpressions() {
            String method = """
                public void variableInExpressions() {
                    int x = 5;
                    int y = 10;
                    int result = x * y + x - y;
                    System.out.println(result);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable in expressions should transform code");
            assertCompiles(transformed, "Variable in expressions should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable in expressions should preserve semantics");
        }
        
        @Test
        @DisplayName("Variable in method calls")
        public void testVariableRenaming_Case9_VariableInMethodCalls() {
            String method = """
                public void variableInMethodCalls() {
                    String name = "test";
                    int length = name.length();
                    String upper = name.toUpperCase();
                    System.out.println(upper + " " + length);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Variable in method calls should transform code");
            assertCompiles(transformed, "Variable in method calls should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Variable in method calls should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: single character variables")
        public void testVariableRenaming_Case10_SingleCharacterVariables() {
            String method = """
                public void singleCharacterVariables() {
                    int a = 1;
                    int b = 2;
                    int c = a + b;
                    return c;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single character variables should transform code");
            assertCompiles(transformed, "Single character variables should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Single character variables should preserve semantics");
        }
        
        @Test
        @DisplayName("Edge case: reserved keywords nearby")
        public void testVariableRenaming_Case11_ReservedKeywordsNearby() {
            String method = """
                public void reservedKeywordsNearby() {
                    int class_var = 1;
                    int public_var = 2;
                    int static_var = 3;
                    System.out.println(class_var + public_var + static_var);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Reserved keywords nearby should transform code");
            assertCompiles(transformed, "Reserved keywords nearby should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Reserved keywords nearby should preserve semantics");
        }
    }
    
    @Test
    @DisplayName("Comprehensive variable renaming test")
    public void testVariableRenaming_ComprehensiveTest() {
        String method = """
            public void comprehensiveTest() {
                // Test various variable renaming scenarios
                int count = 0;
                String name = "test";
                
                for (int i = 0; i < 5; i++) {
                    int temp = i * 2;
                    count += temp;
                }
                
                if (count > 10) {
                    String result = name + "_" + count;
                    System.out.println(result);
                }
            }
            """;
        
        String original = createTestClass(method);
        String transformed = applyTransformation(original, TRANSFORMATION, MODE);
        
        assertTransformationApplied(original, transformed, "Comprehensive test should transform code");
        assertCompiles(transformed, "Comprehensive test should produce compilable code");
        assertSemanticallyEquivalent(original, transformed, "Comprehensive test should preserve semantics");
    }
}