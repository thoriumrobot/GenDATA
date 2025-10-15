package cfwr.jdt.transformations.enhanced;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for Loop Conversion transformation.
 * Tests for-to-while and while-to-for conversions with various loop patterns.
 */
@DisplayName("Loop Conversion Transformation Tests")
class LoopConversionTransformationTest extends TransformationTestBase {
    
    @Override
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    private static final String TRANSFORMATION = "loop_conversion";
    private static final String MODE = "enhanced";
    
    @Nested
    @DisplayName("For-to-While Conversions")
    class ForToWhileConversions {
        
        @Test
        @DisplayName("Simple for loop to while loop conversion")
        public void testLoopConversion_Case1_SimpleForToWhile() {
            String method = """
                public void simpleLoop() {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple for-to-while conversion should transform code");
            assertCompiles(transformed, "Simple for-to-while conversion should produce compilable code");
            assertSemanticallyEquivalent(original, transformed, "Simple for-to-while conversion should preserve semantics");
            
            // Verify while loop structure
            assertTrue(transformed.contains("while ("), "Should contain while loop");
            assertTrue(transformed.contains("int i = 0"), "Should contain initialization");
            assertTrue(transformed.contains("i++"), "Should contain increment");
        }
        
        @Test
        @DisplayName("For loop with complex initialization")
        public void testLoopConversion_Case2_ComplexInitialization() {
            String method = """
                public void complexInitLoop() {
                    int x = 5;
                    for (int i = x * 2, j = 0; i < 20 && j < 10; i++, j += 2) {
                        System.out.println(i + j);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex initialization");
            assertCompiles(transformed, "Complex initialization");
            assertSemanticallyEquivalent(original, transformed, "Complex initialization");
        }
        
        @Test
        @DisplayName("For loop with multiple update expressions")
        public void testLoopConversion_Case3_MultipleUpdates() {
            String method = """
                public void multipleUpdatesLoop() {
                    for (int i = 0, j = 10; i < j; i++, j--) {
                        System.out.println("i=" + i + ", j=" + j);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple updates");
            assertCompiles(transformed, "Multiple updates");
            assertSemanticallyEquivalent(original, transformed, "Multiple updates");
        }
        
        @Test
        @DisplayName("Enhanced for-each loop preservation")
        public void testLoopConversion_Case4_ForEachPreservation() {
            String method = """
                public void forEachLoop(int[] array) {
                    for (int element : array) {
                        System.out.println(element);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // For-each loops should not be converted to while loops
            // They should remain as for-each loops
            assertTrue(transformed.contains("for (int element : array)"), 
                "Enhanced for-each loop should be preserved");
            assertCompiles(transformed, "For-each preservation");
            assertSemanticallyEquivalent(original, transformed, "For-each preservation");
        }
        
        @Test
        @DisplayName("Empty for loop body")
        public void testLoopConversion_Case5_EmptyLoopBody() {
            String method = """
                public void emptyLoopBody() {
                    for (int i = 0; i < 1000000; i++) {
                        // Empty loop for timing
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Empty loop body");
            assertCompiles(transformed, "Empty loop body");
            assertSemanticallyEquivalent(original, transformed, "Empty loop body");
        }
        
        @Test
        @DisplayName("For loop with break statement")
        public void testLoopConversion_Case6_LoopWithBreak() {
            String method = """
                public void loopWithBreak() {
                    for (int i = 0; i < 100; i++) {
                        if (i == 50) {
                            break;
                        }
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Loop with break");
            assertCompiles(transformed, "Loop with break");
            assertSemanticallyEquivalent(original, transformed, "Loop with break");
            assertTrue(transformed.contains("break"), "Break statement should be preserved");
        }
        
        @Test
        @DisplayName("For loop with continue statement")
        public void testLoopConversion_Case7_LoopWithContinue() {
            String method = """
                public void loopWithContinue() {
                    for (int i = 0; i < 10; i++) {
                        if (i % 2 == 0) {
                            continue;
                        }
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Loop with continue");
            assertCompiles(transformed, "Loop with continue");
            assertSemanticallyEquivalent(original, transformed, "Loop with continue");
            assertTrue(transformed.contains("continue"), "Continue statement should be preserved");
        }
        
        @Test
        @DisplayName("For loop with labeled break")
        public void testLoopConversion_Case8_LabeledBreak() {
            String method = """
                public void labeledBreakLoop() {
                    outer: for (int i = 0; i < 5; i++) {
                        for (int j = 0; j < 5; j++) {
                            if (i + j > 6) {
                                break outer;
                            }
                            System.out.println(i + "," + j);
                        }
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Labeled break");
            assertCompiles(transformed, "Labeled break");
            assertSemanticallyEquivalent(original, transformed, "Labeled break");
            assertTrue(transformed.contains("break outer"), "Labeled break should be preserved");
        }
        
        @Test
        @DisplayName("For loop with complex condition")
        public void testLoopConversion_Case9_ComplexCondition() {
            String method = """
                public void complexConditionLoop() {
                    for (int i = 0; i < 100 && i % 7 != 0; i += 3) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex condition");
            assertCompiles(transformed, "Complex condition");
            assertSemanticallyEquivalent(original, transformed, "Complex condition");
        }
        
        @Test
        @DisplayName("Nested for loops")
        public void testLoopConversion_Case10_NestedLoops() {
            String method = """
                public void nestedLoops() {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            System.out.println(i + "," + j);
                        }
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Nested loops");
            assertCompiles(transformed, "Nested loops");
            assertSemanticallyEquivalent(original, transformed, "Nested loops");
        }
        
        @Test
        @DisplayName("For loop with method calls in condition")
        public void testLoopConversion_Case11_MethodCallCondition() {
            String method = """
                public void methodCallCondition() {
                    for (int i = 0; i < getMaxValue(); i++) {
                        System.out.println(i);
                    }
                }
                
                private int getMaxValue() {
                    return 10;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Method call condition");
            assertCompiles(transformed, "Method call condition");
            assertSemanticallyEquivalent(original, transformed, "Method call condition");
        }
        
        @Test
        @DisplayName("For loop with array access in condition")
        public void testLoopConversion_Case12_ArrayAccessCondition() {
            String method = """
                public void arrayAccessCondition(int[] bounds) {
                    for (int i = 0; i < bounds.length && bounds[i] > 0; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Array access condition");
            assertCompiles(transformed, "Array access condition");
            assertSemanticallyEquivalent(original, transformed, "Array access condition");
        }
    }
    
    @Nested
    @DisplayName("While-to-For Conversions")
    class WhileToForConversions {
        
        @Test
        @DisplayName("Simple while loop to for loop conversion")
        public void testLoopConversion_Case13_SimpleWhileToFor() {
            String method = """
                public void simpleWhileLoop() {
                    int i = 0;
                    while (i < 10) {
                        System.out.println(i);
                        i++;
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Simple while-to-for");
            assertCompiles(transformed, "Simple while-to-for");
            assertSemanticallyEquivalent(original, transformed, "Simple while-to-for");
        }
        
        @Test
        @DisplayName("While loop with complex initialization")
        public void testLoopConversion_Case14_ComplexWhileInit() {
            String method = """
                public void complexWhileInit() {
                    int i = calculateStart();
                    while (i < calculateEnd()) {
                        System.out.println(i);
                        i += calculateStep();
                    }
                }
                
                private int calculateStart() { return 0; }
                private int calculateEnd() { return 10; }
                private int calculateStep() { return 1; }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex while init");
            assertCompiles(transformed, "Complex while init");
            assertSemanticallyEquivalent(original, transformed, "Complex while init");
        }
        
        @Test
        @DisplayName("While loop with multiple counter variables")
        public void testLoopConversion_Case15_MultipleCounters() {
            String method = """
                public void multipleCounters() {
                    int i = 0, j = 10;
                    while (i < j) {
                        System.out.println("i=" + i + ", j=" + j);
                        i++;
                        j--;
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Multiple counters");
            assertCompiles(transformed, "Multiple counters");
            assertSemanticallyEquivalent(original, transformed, "Multiple counters");
        }
        
        @Test
        @DisplayName("While loop with break and continue")
        public void testLoopConversion_Case16_WhileWithBreakContinue() {
            String method = """
                public void whileWithBreakContinue() {
                    int i = 0;
                    while (i < 100) {
                        if (i == 50) break;
                        if (i % 2 == 0) {
                            i++;
                            continue;
                        }
                        System.out.println(i);
                        i++;
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "While with break/continue");
            assertCompiles(transformed, "While with break/continue");
            assertSemanticallyEquivalent(original, transformed, "While with break/continue");
        }
        
        @Test
        @DisplayName("Infinite while loop (should not convert)")
        public void testLoopConversion_Case17_InfiniteWhileLoop() {
            String method = """
                public void infiniteWhileLoop() {
                    while (true) {
                        System.out.println("Infinite loop");
                        if (someCondition()) break;
                    }
                }
                
                private boolean someCondition() {
                    return Math.random() > 0.5;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // Infinite while loops should not be converted to for loops
            assertTrue(transformed.contains("while (true)"), 
                "Infinite while loop should be preserved");
            assertCompiles(transformed, "Infinite while loop");
            assertSemanticallyEquivalent(original, transformed, "Infinite while loop");
        }
        
        @Test
        @DisplayName("While loop with complex condition")
        public void testLoopConversion_Case18_ComplexWhileCondition() {
            String method = """
                public void complexWhileCondition() {
                    int i = 0;
                    while (i < 100 && i % 7 != 0 && isValid(i)) {
                        System.out.println(i);
                        i += 3;
                    }
                }
                
                private boolean isValid(int value) {
                    return value >= 0;
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Complex while condition");
            assertCompiles(transformed, "Complex while condition");
            assertSemanticallyEquivalent(original, transformed, "Complex while condition");
        }
    }
    
    @Nested
    @DisplayName("Edge Cases and Error Handling")
    class EdgeCases {
        
        @Test
        @DisplayName("Loop with no body (empty block)")
        public void testLoopConversion_Case19_EmptyBlock() {
            String method = """
                public void emptyBlockLoop() {
                    for (int i = 0; i < 10; i++) {
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Empty block");
            assertCompiles(transformed, "Empty block");
            assertSemanticallyEquivalent(original, transformed, "Empty block");
        }
        
        @Test
        @DisplayName("Loop with single statement (no braces)")
        public void testLoopConversion_Case20_SingleStatement() {
            String method = """
                public void singleStatementLoop() {
                    for (int i = 0; i < 10; i++)
                        System.out.println(i);
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Single statement");
            assertCompiles(transformed, "Single statement");
            assertSemanticallyEquivalent(original, transformed, "Single statement");
        }
        
        @Test
        @DisplayName("Loop with variable shadowing")
        public void testLoopConversion_Case21_VariableShadowing() {
            String method = """
                public void variableShadowingLoop(int i) {
                    for (int i = 0; i < 10; i++) {
                        System.out.println(i);
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // This should not compile due to variable shadowing
            assertDoesNotCompile(original, "Variable shadowing should not compile");
            assertDoesNotCompile(transformed, "Variable shadowing should not compile");
        }
        
        @Test
        @DisplayName("Loop with unreachable code after break")
        public void testLoopConversion_Case22_UnreachableCode() {
            String method = """
                public void unreachableCodeLoop() {
                    for (int i = 0; i < 10; i++) {
                        if (i == 5) {
                            break;
                            System.out.println("Unreachable");
                        }
                    }
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            // This should not compile due to unreachable code
            assertDoesNotCompile(original, "Unreachable code should not compile");
            assertDoesNotCompile(transformed, "Unreachable code should not compile");
        }
        
        @Test
        @DisplayName("Loop with exception handling")
        public void testLoopConversion_Case23_ExceptionHandling() {
            String method = """
                public void exceptionHandlingLoop() {
                    for (int i = 0; i < 10; i++) {
                        try {
                            riskyOperation(i);
                        } catch (Exception e) {
                            System.out.println("Error: " + e.getMessage());
                            break;
                        }
                    }
                }
                
                private void riskyOperation(int value) throws Exception {
                    if (value == 7) throw new Exception("Lucky number");
                }
                """;
            
            String original = createTestClass(method);
            String transformed = applyTransformation(original, TRANSFORMATION, MODE);
            
            assertTransformationApplied(original, transformed, "Exception handling");
            assertCompiles(transformed, "Exception handling");
            assertSemanticallyEquivalent(original, transformed, "Exception handling");
        }
    }
}
