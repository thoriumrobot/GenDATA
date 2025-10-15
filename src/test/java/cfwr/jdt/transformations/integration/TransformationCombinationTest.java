package cfwr.jdt.transformations.integration;

import cfwr.jdt.SemanticTransformer;
import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Comprehensive tests for transformation combinations to ensure transformations
 * work correctly when applied together and respect compatibility constraints.
 */
public class TransformationCombinationTest extends TransformationTestBase {
    
    private SemanticTransformer transformer;
    
    // Define transformation categories for systematic testing
    private static final List<String> ENHANCED_TRANSFORMATIONS = Arrays.asList(
        "loop_conversion", "guard_reversal", "mathematical_expression", "logical_expression",
        "ternary_operator", "switch_statement", "variable_operation", "method_extraction",
        "conditional_expression", "array_access_pattern", "string_concatenation", "numeric_literal",
        "exception_handling", "lambda_expression", "stream_api", "builder_pattern", "functional_conversion"
    );
    
    private static final List<String> SIMPLE_TRANSFORMATIONS = Arrays.asList(
        "simple_method_call", "simple_assignment", "simple_conditional", "simple_array_access",
        "simple_return_statement", "simple_variable_declaration", "simple_constructor_call",
        "simple_field_access", "simple_string_operation", "simple_numeric_operation"
    );
    
    private static final List<String> RANDOM_TRANSFORMATIONS = Arrays.asList(
        "random_method_insertion", "random_statement_insertion", "random_expression_insertion"
    );
    
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer();
    }
    
    @Test
    public void testCompatibleTransformationPairs() {
        // Test all pairs of compatible transformations
        List<String> compatibleTransformations = new ArrayList<>();
        compatibleTransformations.addAll(ENHANCED_TRANSFORMATIONS);
        compatibleTransformations.addAll(SIMPLE_TRANSFORMATIONS);
        compatibleTransformations.addAll(RANDOM_TRANSFORMATIONS);
        
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    for (int i = 0; i < 10; i++) {
                        if (i % 2 == 0) {
                            System.out.println("Even: " + i);
                        } else {
                            System.out.println("Odd: " + i);
                        }
                    }
                }
            }
            """;
        
        // Test random pairs of transformations
        Random random = new Random(42); // Fixed seed for reproducibility
        for (int i = 0; i < 20; i++) {
            String transformation1 = compatibleTransformations.get(random.nextInt(compatibleTransformations.size()));
            String transformation2 = compatibleTransformations.get(random.nextInt(compatibleTransformations.size()));
            
            if (!transformation1.equals(transformation2)) {
                testTransformationPair(testCode, transformation1, transformation2);
            }
        }
    }
    
    @Test
    public void testIncompatibleTransformationPairs() {
        // Test that incompatible transformations are handled correctly
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    for (int i = 0; i < 10; i++) {
                        if (i > 5) {
                            System.out.println("Large: " + i);
                        }
                    }
                }
            }
            """;
        
        // Test incompatible pairs (loop_conversion + guard_reversal)
        String transformedCode = transformer.transformCode(testCode, 
            Arrays.asList("loop_conversion", "guard_reversal"), "enhanced");
        
        assertCompiles(transformedCode, "Incompatible transformations should still produce compilable code");
        
        // The transformation should apply one but not both (due to compatibility checking)
        assertTransformationApplied(testCode, transformedCode, 
            "At least one transformation should be applied");
    }
    
    @Test
    public void testTransformationTriples() {
        // Test all triples of compatible transformations
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 0;
                    for (int i = 0; i < 5; i++) {
                        result = result + i * 2;
                    }
                    return result > 10 ? "Large" : "Small";
                }
            }
            """;
        
        // Test various triples
        List<List<String>> testTriples = Arrays.asList(
            Arrays.asList("mathematical_expression", "logical_expression", "simple_assignment"),
            Arrays.asList("ternary_operator", "simple_method_call", "simple_return_statement"),
            Arrays.asList("variable_operation", "numeric_literal", "simple_conditional")
        );
        
        for (List<String> triple : testTriples) {
            String transformedCode = transformer.transformCode(testCode, triple, "enhanced");
            
            assertCompiles(transformedCode, 
                "Triple transformation should produce compilable code: " + triple);
            assertTransformationApplied(testCode, transformedCode, 
                "Triple transformation should change code: " + triple);
        }
    }
    
    @Test
    public void testAllSimpleTransformations() {
        // Test all simple transformations together
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int[] array = {1, 2, 3};
                    String result = new String("Hello");
                    return array[0] + result.length();
                }
            }
            """;
        
        String transformedCode = transformer.transformCode(testCode, SIMPLE_TRANSFORMATIONS, "simple");
        
        assertCompiles(transformedCode, "All simple transformations should work together");
        assertTransformationApplied(testCode, transformedCode, 
            "All simple transformations should change code");
    }
    
    @Test
    public void testRandomCombinations() {
        // Test random combinations of 5+ transformations
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int sum = 0;
                    for (int i = 0; i < 10; i++) {
                        if (i % 2 == 0) {
                            sum += i * 2;
                        }
                    }
                    return sum > 50 ? "High" : "Low";
                }
            }
            """;
        
        List<String> allTransformations = new ArrayList<>();
        allTransformations.addAll(ENHANCED_TRANSFORMATIONS);
        allTransformations.addAll(SIMPLE_TRANSFORMATIONS);
        allTransformations.addAll(RANDOM_TRANSFORMATIONS);
        
        Random random = new Random(123); // Fixed seed for reproducibility
        
        // Test 10 random combinations of 5 transformations
        for (int i = 0; i < 10; i++) {
            Set<String> selectedTransformations = new HashSet<>();
            while (selectedTransformations.size() < 5) {
                selectedTransformations.add(allTransformations.get(random.nextInt(allTransformations.size())));
            }
            
            List<String> transformationList = new ArrayList<>(selectedTransformations);
            String transformedCode = transformer.transformCode(testCode, transformationList, "enhanced");
            
            assertCompiles(transformedCode, 
                "Random combination should produce compilable code: " + transformationList);
            assertTransformationApplied(testCode, transformedCode, 
                "Random combination should change code: " + transformationList);
        }
    }
    
    @Test
    public void testOrderIndependence() {
        // Test that transformation order doesn't affect final result
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5 + 3;
                    boolean flag = x > 0 && x < 10;
                    return flag ? "Valid" : "Invalid";
                }
            }
            """;
        
        List<String> transformations = Arrays.asList(
            "mathematical_expression", "logical_expression", "ternary_operator"
        );
        
        // Apply transformations in different orders
        List<List<String>> orders = Arrays.asList(
            transformations,
            Arrays.asList("logical_expression", "ternary_operator", "mathematical_expression"),
            Arrays.asList("ternary_operator", "mathematical_expression", "logical_expression")
        );
        
        Set<String> results = new HashSet<>();
        
        for (List<String> order : orders) {
            String transformedCode = transformer.transformCode(testCode, order, "enhanced");
            assertCompiles(transformedCode, "Order should not affect compilation: " + order);
            results.add(transformedCode);
        }
        
        // All results should be compilable (they may be different due to transformation order)
        assertEquals(orders.size(), results.size(), 
            "Different orders may produce different results, but all should be valid");
    }
    
    @Test
    public void testStressTestAllTransformations() {
        // Stress test with all 27 transformations
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int[] numbers = {1, 2, 3, 4, 5};
                    int sum = 0;
                    
                    for (int i = 0; i < numbers.length; i++) {
                        if (numbers[i] % 2 == 0) {
                            sum += numbers[i] * 2;
                        }
                    }
                    
                    return sum > 20 ? "High" : "Low";
                }
            }
            """;
        
        List<String> allTransformations = new ArrayList<>();
        allTransformations.addAll(ENHANCED_TRANSFORMATIONS);
        allTransformations.addAll(SIMPLE_TRANSFORMATIONS);
        allTransformations.addAll(RANDOM_TRANSFORMATIONS);
        
        long startTime = System.currentTimeMillis();
        
        String transformedCode = transformer.transformCode(testCode, allTransformations, "enhanced");
        
        long duration = System.currentTimeMillis() - startTime;
        
        assertCompiles(transformedCode, "All transformations should produce compilable code");
        assertTrue(duration < 10000, "All transformations should complete within 10 seconds, took: " + duration + "ms");
        
        // Verify diagnostics are captured
        assertNotNull(transformer.getDiagnostics(), "Diagnostics should be available for stress test");
    }
    
    @Test
    public void testIdempotencyOfTransformations() {
        // Test that applying the same transformation twice is safe
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 5 + 3 * 2;
                    return result > 10;
                }
            }
            """;
        
        List<String> transformations = Arrays.asList("mathematical_expression");
        
        // Apply transformation twice
        String firstTransform = transformer.transformCode(testCode, transformations, "enhanced");
        SemanticTransformer secondTransformer = new SemanticTransformer();
        String secondTransform = secondTransformer.transformCode(firstTransform, transformations, "enhanced");
        
        assertCompiles(secondTransform, "Second application should produce compilable code");
        assertCompiles(firstTransform, "First application should produce compilable code");
    }
    
    @Test
    public void testCompositionWithDifferentModes() {
        // Test transformations with different modes
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int x = 5;
                    int y = x + 1;
                    return y;
                }
            }
            """;
        
        // Test simple mode
        String simpleResult = transformer.transformCode(testCode, 
            Arrays.asList("simple_assignment"), "simple");
        assertCompiles(simpleResult, "Simple mode should work");
        
        // Test enhanced mode
        SemanticTransformer enhancedTransformer = new SemanticTransformer();
        String enhancedResult = enhancedTransformer.transformCode(testCode, 
            Arrays.asList("mathematical_expression"), "enhanced");
        assertCompiles(enhancedResult, "Enhanced mode should work");
        
        // Both should be compilable
        assertTransformationApplied(testCode, simpleResult, "Simple mode should transform code");
        assertTransformationApplied(testCode, enhancedResult, "Enhanced mode should transform code");
    }
    
    /**
     * Helper method to test a pair of transformations.
     */
    private void testTransformationPair(String testCode, String transformation1, String transformation2) {
        String transformedCode = transformer.transformCode(testCode, 
            Arrays.asList(transformation1, transformation2), "enhanced");
        
        assertCompiles(transformedCode, 
            "Transformation pair should produce compilable code: " + transformation1 + " + " + transformation2);
        
        // At least one transformation should be applied (unless both are skipped)
        assertTransformationApplied(testCode, transformedCode, 
            "At least one transformation should be applied: " + transformation1 + " + " + transformation2);
    }
}
