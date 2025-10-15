package cfwr.jdt.transformations;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.Timeout;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Integration test suite for all 27 semantic transformations.
 * Tests transformation combinations, order independence, idempotency, and performance.
 */
@DisplayName("All Transformations Integration Tests")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AllTransformationsIntegrationTest extends TransformationTestBase {
    
    private static final String MODE = "enhanced";
    
    // All 27 transformation names
    private static final List<String> ALL_TRANSFORMATIONS = Arrays.asList(
        // Enhanced transformations (17)
        "loop_conversion", "guard_reversal", "mathematical_expression", "logical_expression",
        "ternary_operator", "switch_statement", "variable_operation", "method_extraction",
        "conditional_expression", "array_access_pattern", "string_concatenation", "numeric_literal",
        "exception_handling", "lambda_expression", "stream_api", "builder_pattern", "functional_conversion",
        
        // Simple transformations (10)
        "simple_method_call", "simple_assignment", "simple_conditional", "simple_array_access",
        "simple_return_statement", "simple_variable_declaration", "simple_constructor_call",
        "simple_field_access", "simple_string_operation", "simple_numeric_operation",
        
        // Random transformations (3)
        "random_method_insertion", "random_statement_insertion", "random_expression_insertion"
    );
    
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    @Nested
    @DisplayName("Individual Transformation Tests")
    class IndividualTransformationTests {
        
        @Test
        @DisplayName("Test all 27 transformations individually")
        void testAllTransformationsIndividually() {
            String testMethod = """
                public void testMethod(int x, String str) {
                    for (int i = 0; i < x; i++) {
                        if (i > 0) {
                            System.out.println(str + ": " + i);
                        }
                    }
                }
                """;
            
            String original = createTestClass(testMethod);
            
            for (String transformation : ALL_TRANSFORMATIONS) {
                String transformed = applyTransformation(original, transformation, MODE);
                
                assertNotNull(transformed, "Transformation " + transformation + " should return result");
                assertCompiles(transformed, "Transformation " + transformation);
                assertSemanticallyEquivalent(original, transformed, "Transformation " + transformation);
                
                logTestExecution("Individual_" + transformation, original, transformed, true);
            }
        }
    }
    
    @Nested
    @DisplayName("Transformation Combinations")
    class TransformationCombinations {
        
        @Test
        @DisplayName("Test transformation pairs")
        void testTransformationPairs() {
            String testMethod = """
                public int testMethod(int x, int y) {
                    int result = 0;
                    for (int i = 0; i < x; i++) {
                        if (i > 0) {
                            result += y * i;
                        }
                    }
                    return result;
                }
                """;
            
            String original = createTestClass(testMethod);
            
            // Test combinations of compatible transformations
            List<List<String>> compatiblePairs = Arrays.asList(
                Arrays.asList("loop_conversion", "guard_reversal"),
                Arrays.asList("mathematical_expression", "logical_expression"),
                Arrays.asList("ternary_operator", "switch_statement"),
                Arrays.asList("variable_operation", "method_extraction"),
                Arrays.asList("simple_assignment", "simple_method_call"),
                Arrays.asList("simple_conditional", "simple_return_statement")
            );
            
            for (List<String> pair : compatiblePairs) {
                String transformed = applyTransformations(original, pair, MODE);
                
                assertNotNull(transformed, "Combination " + pair + " should return result");
                assertCompiles(transformed, "Combination " + pair);
                assertSemanticallyEquivalent(original, transformed, "Combination " + pair);
                
                logTestExecution("Pair_" + String.join("_", pair), original, transformed, true);
            }
        }
        
        @Test
        @DisplayName("Test transformation triples")
        void testTransformationTriples() {
            String testMethod = """
                public String testMethod(int x) {
                    if (x > 0) {
                        return x > 10 ? "large" : "small";
                    } else {
                        return "negative";
                    }
                }
                """;
            
            String original = createTestClass(testMethod);
            
            // Test combinations of three transformations
            List<List<String>> compatibleTriples = Arrays.asList(
                Arrays.asList("guard_reversal", "ternary_operator", "mathematical_expression"),
                Arrays.asList("simple_assignment", "simple_conditional", "simple_return_statement"),
                Arrays.asList("loop_conversion", "variable_operation", "simple_method_call")
            );
            
            for (List<String> triple : compatibleTriples) {
                String transformed = applyTransformations(original, triple, MODE);
                
                assertNotNull(transformed, "Triple " + triple + " should return result");
                assertCompiles(transformed, "Triple " + triple);
                assertSemanticallyEquivalent(original, transformed, "Triple " + triple);
                
                logTestExecution("Triple_" + String.join("_", triple), original, transformed, true);
            }
        }
    }
    
    @Nested
    @DisplayName("Order Independence Tests")
    class OrderIndependenceTests {
        
        @Test
        @DisplayName("Test transformation order independence")
        void testTransformationOrderIndependence() {
            String testMethod = """
                public int testMethod(int x, int y) {
                    int result = x + y;
                    if (result > 10) {
                        result *= 2;
                    }
                    return result;
                }
                """;
            
            String original = createTestClass(testMethod);
            
            // Test different orders of the same transformations
            List<String> transformations = Arrays.asList("guard_reversal", "mathematical_expression", "variable_operation");
            
            List<List<String>> orders = Arrays.asList(
                Arrays.asList("guard_reversal", "mathematical_expression", "variable_operation"),
                Arrays.asList("mathematical_expression", "guard_reversal", "variable_operation"),
                Arrays.asList("variable_operation", "guard_reversal", "mathematical_expression"),
                Arrays.asList("mathematical_expression", "variable_operation", "guard_reversal")
            );
            
            List<String> results = new ArrayList<>();
            
            for (List<String> order : orders) {
                String transformed = applyTransformations(original, order, MODE);
                results.add(transformed);
                
                assertNotNull(transformed, "Order " + order + " should return result");
                assertCompiles(transformed, "Order " + order);
                assertSemanticallyEquivalent(original, transformed, "Order " + order);
                
                logTestExecution("Order_" + String.join("_", order), original, transformed, true);
            }
            
            // All results should be semantically equivalent
            for (int i = 1; i < results.size(); i++) {
                assertSemanticallyEquivalent(results.get(0), results.get(i), 
                    "Results from different orders should be equivalent");
            }
        }
    }
    
    @Nested
    @DisplayName("Idempotency Tests")
    class IdempotencyTests {
        
        @Test
        @DisplayName("Test transformation idempotency")
        void testTransformationIdempotency() {
            String testMethod = """
                public void testMethod(int x) {
                    if (x > 0) {
                        System.out.println("Positive: " + x);
                    } else {
                        System.out.println("Non-positive: " + x);
                    }
                }
                """;
            
            String original = createTestClass(testMethod);
            
            // Test that applying the same transformation twice yields the same result
            for (String transformation : Arrays.asList("guard_reversal", "mathematical_expression", "simple_assignment")) {
                String firstTransform = applyTransformation(original, transformation, MODE);
                String secondTransform = applyTransformation(firstTransform, transformation, MODE);
                
                assertNotNull(firstTransform, "First transformation should return result");
                assertNotNull(secondTransform, "Second transformation should return result");
                assertCompiles(firstTransform, "First transformation");
                assertCompiles(secondTransform, "Second transformation");
                assertSemanticallyEquivalent(original, firstTransform, "First transformation");
                assertSemanticallyEquivalent(original, secondTransform, "Second transformation");
                assertSemanticallyEquivalent(firstTransform, secondTransform, "Idempotency for " + transformation);
                
                logTestExecution("Idempotency_" + transformation, firstTransform, secondTransform, true);
            }
        }
    }
    
    @Nested
    @DisplayName("Real-World Code Samples")
    class RealWorldCodeSamples {
        
        @Test
        @DisplayName("Test with real-world code patterns")
        void testWithRealWorldCodePatterns() {
            // Sample 1: Data processing method
            String dataProcessingMethod = """
                public List<String> processData(List<Integer> numbers) {
                    List<String> results = new ArrayList<>();
                    for (Integer num : numbers) {
                        if (num != null && num > 0) {
                            String result = num > 100 ? "large" : "small";
                            results.add(result);
                        }
                    }
                    return results;
                }
                """;
            
            String original1 = createTestClass(dataProcessingMethod);
            String transformed1 = applyTransformations(original1, 
                Arrays.asList("loop_conversion", "guard_reversal", "ternary_operator"), MODE);
            
            assertNotNull(transformed1, "Data processing transformation should return result");
            assertCompiles(transformed1, "Data processing transformation");
            assertSemanticallyEquivalent(original1, transformed1, "Data processing transformation");
            
            // Sample 2: Mathematical calculation method
            String calculationMethod = """
                public double calculateArea(double radius) {
                    if (radius <= 0) {
                        throw new IllegalArgumentException("Radius must be positive");
                    }
                    return Math.PI * radius * radius;
                }
                """;
            
            String original2 = createTestClass(calculationMethod);
            String transformed2 = applyTransformations(original2, 
                Arrays.asList("guard_reversal", "mathematical_expression", "exception_handling"), MODE);
            
            assertNotNull(transformed2, "Calculation transformation should return result");
            assertCompiles(transformed2, "Calculation transformation");
            assertSemanticallyEquivalent(original2, transformed2, "Calculation transformation");
            
            // Sample 3: String manipulation method
            String stringMethod = """
                public String formatMessage(String name, int count) {
                    String message = "";
                    if (count == 1) {
                        message = "Hello " + name;
                    } else if (count > 1) {
                        message = "Hello " + name + " (count: " + count + ")";
                    }
                    return message;
                }
                """;
            
            String original3 = createTestClass(stringMethod);
            String transformed3 = applyTransformations(original3, 
                Arrays.asList("guard_reversal", "string_concatenation", "simple_assignment"), MODE);
            
            assertNotNull(transformed3, "String manipulation transformation should return result");
            assertCompiles(transformed3, "String manipulation transformation");
            assertSemanticallyEquivalent(original3, transformed3, "String manipulation transformation");
        }
    }
    
    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {
        
        @Test
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        @DisplayName("Test transformation performance")
        void testTransformationPerformance() {
            String largeMethod = """
                public void largeMethod(int[] data) {
                    int sum = 0;
                    int product = 1;
                    int max = Integer.MIN_VALUE;
                    int min = Integer.MAX_VALUE;
                    
                    for (int i = 0; i < data.length; i++) {
                        int value = data[i];
                        sum += value;
                        product *= value;
                        
                        if (value > max) {
                            max = value;
                        }
                        if (value < min) {
                            min = value;
                        }
                    }
                    
                    System.out.println("Sum: " + sum);
                    System.out.println("Product: " + product);
                    System.out.println("Max: " + max);
                    System.out.println("Min: " + min);
                }
                """;
            
            String original = createTestClass(largeMethod);
            
            // Measure performance for multiple transformations
            long startTime = System.currentTimeMillis();
            
            for (int i = 0; i < 10; i++) {
                String transformed = applyTransformations(original, 
                    Arrays.asList("loop_conversion", "guard_reversal", "mathematical_expression", 
                                 "variable_operation", "simple_assignment"), MODE);
                
                assertNotNull(transformed, "Performance test transformation should return result");
                assertCompiles(transformed, "Performance test transformation");
            }
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            // Performance should be reasonable (less than 30 seconds for 10 iterations)
            assertTrue(duration < 30000, "Transformation performance should be reasonable: " + duration + "ms");
            
            System.out.println("Performance test completed in " + duration + "ms for 10 iterations");
        }
    }
    
    @Nested
    @DisplayName("Memory Usage Tests")
    class MemoryUsageTests {
        
        @Test
        @DisplayName("Test memory usage during transformations")
        void testMemoryUsage() {
            String method = """
                public void memoryTest() {
                    List<String> list = new ArrayList<>();
                    for (int i = 0; i < 1000; i++) {
                        list.add("Item " + i);
                    }
                    
                    for (String item : list) {
                        System.out.println(item);
                    }
                }
                """;
            
            String original = createTestClass(method);
            
            // Get initial memory usage
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long initialMemory = runtime.totalMemory() - runtime.freeMemory();
            
            // Apply transformations
            for (int i = 0; i < 100; i++) {
                String transformed = applyTransformations(original, 
                    Arrays.asList("loop_conversion", "string_concatenation", "simple_assignment"), MODE);
                
                assertNotNull(transformed, "Memory test transformation should return result");
                assertCompiles(transformed, "Memory test transformation");
            }
            
            // Check final memory usage
            runtime.gc();
            long finalMemory = runtime.totalMemory() - runtime.freeMemory();
            long memoryIncrease = finalMemory - initialMemory;
            
            // Memory increase should be reasonable (less than 100MB)
            assertTrue(memoryIncrease < 100 * 1024 * 1024, 
                "Memory usage should be reasonable: " + (memoryIncrease / 1024 / 1024) + "MB increase");
            
            System.out.println("Memory usage test: " + (memoryIncrease / 1024 / 1024) + "MB increase");
        }
    }
    
    @Nested
    @DisplayName("Thread Safety Tests")
    class ThreadSafetyTests {
        
        @Test
        @DisplayName("Test concurrent transformation execution")
        void testConcurrentTransformationExecution() throws InterruptedException {
            String method = """
                public int concurrentTest(int x) {
                    if (x > 0) {
                        return x * 2;
                    } else {
                        return -x;
                    }
                }
                """;
            
            String original = createTestClass(method);
            
            int numThreads = 10;
            Thread[] threads = new Thread[numThreads];
            boolean[] results = new boolean[numThreads];
            
            // Create and start threads
            for (int i = 0; i < numThreads; i++) {
                final int threadIndex = i;
                threads[i] = new Thread(() -> {
                    try {
                        String transformed = applyTransformations(original, 
                            Arrays.asList("guard_reversal", "mathematical_expression"), MODE);
                        
                        results[threadIndex] = (transformed != null && 
                                              compilationValidator.isValid(transformed) &&
                                              equivalenceChecker.areEquivalent(original, transformed));
                    } catch (Exception e) {
                        results[threadIndex] = false;
                        e.printStackTrace();
                    }
                });
                threads[i].start();
            }
            
            // Wait for all threads to complete
            for (Thread thread : threads) {
                thread.join();
            }
            
            // Check that all threads succeeded
            for (int i = 0; i < numThreads; i++) {
                assertTrue(results[i], "Thread " + i + " should complete successfully");
            }
            
            System.out.println("Thread safety test completed with " + numThreads + " threads");
        }
    }
    
    @Nested
    @DisplayName("Error Handling Tests")
    class ErrorHandlingTests {
        
        @Test
        @DisplayName("Test transformation with invalid code")
        void testTransformationWithInvalidCode() {
            String invalidCode = """
                public class InvalidClass {
                    public void invalidMethod() {
                        System.out.println("Invalid"  // Missing closing parenthesis
                    }
                }
                """;
            
            // All transformations should handle invalid code gracefully
            for (String transformation : Arrays.asList("guard_reversal", "mathematical_expression", "simple_assignment")) {
                String result = applyTransformation(invalidCode, transformation, MODE);
                
                // Should return original code when parsing fails
                assertEquals(invalidCode, result, "Invalid code should be returned unchanged for " + transformation);
            }
        }
        
        @Test
        @DisplayName("Test transformation with null input")
        void testTransformationWithNullInput() {
            String result = applyTransformation(null, "guard_reversal", MODE);
            assertNull(result, "Null input should return null result");
        }
        
        @Test
        @DisplayName("Test transformation with empty input")
        void testTransformationWithEmptyInput() {
            String result = applyTransformation("", "guard_reversal", MODE);
            assertEquals("", result, "Empty input should return empty result");
        }
        
        @Test
        @DisplayName("Test transformation with unknown transformation name")
        void testTransformationWithUnknownTransformation() {
            String method = """
                public void testMethod() {
                    System.out.println("test");
                }
                """;
            
            String original = createTestClass(method);
            String result = applyTransformation(original, "unknown_transformation", MODE);
            
            // Should return original code unchanged
            assertEquals(original, result, "Unknown transformation should return original code");
        }
    }
}
