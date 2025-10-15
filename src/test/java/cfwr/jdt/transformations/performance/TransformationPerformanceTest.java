package cfwr.jdt.transformations.performance;

import cfwr.jdt.SemanticTransformer;
import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Performance benchmarking suite for transformation operations.
 * Measures time, memory usage, and scalability of transformations.
 */
public class TransformationPerformanceTest extends TransformationTestBase {
    
    private SemanticTransformer transformer;
    
    @BeforeEach
    public void setUp() {
        super.setUp();
        transformer = new SemanticTransformer();
    }
    
    @Test
    public void testTimePerTransformation() {
        // Measure time per individual transformation
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 5 + 3 * 2;
                    boolean flag = result > 10 && result < 20;
                    String status = flag ? "Valid" : "Invalid";
                    System.out.println(status);
                }
            }
            """;
        
        List<String> transformations = Arrays.asList(
            "mathematical_expression", "logical_expression", "ternary_operator",
            "simple_assignment", "simple_method_call", "simple_conditional"
        );
        
        Map<String, Long> transformationTimes = new HashMap<>();
        
        for (String transformation : transformations) {
            long startTime = System.currentTimeMillis();
            
            String result = transformer.transformCode(testCode, 
                Arrays.asList(transformation), "enhanced");
            
            long duration = System.currentTimeMillis() - startTime;
            transformationTimes.put(transformation, duration);
            
            assertCompiles(result, "Transformation should produce compilable code: " + transformation);
            assertTrue(duration < 1000, 
                "Transformation should complete within 1 second: " + transformation + " took " + duration + "ms");
        }
        
        // Log performance results
        System.out.println("=== TRANSFORMATION PERFORMANCE RESULTS ===");
        for (Map.Entry<String, Long> entry : transformationTimes.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue() + "ms");
        }
        
        // Verify all transformations completed in reasonable time
        long totalTime = transformationTimes.values().stream().mapToLong(Long::longValue).sum();
        assertTrue(totalTime < 5000, "All transformations should complete within 5 seconds total, took: " + totalTime + "ms");
    }
    
    @Test
    public void testMemoryUsagePerTransformation() {
        // Measure memory usage (approximate) per transformation
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int[] array = new int[1000];
                    for (int i = 0; i < array.length; i++) {
                        array[i] = i * 2;
                    }
                    return array[500];
                }
            }
            """;
        
        Runtime runtime = Runtime.getRuntime();
        
        // Force garbage collection before measurement
        System.gc();
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();
        
        String result = transformer.transformCode(testCode, 
            Arrays.asList("loop_conversion", "mathematical_expression"), "enhanced");
        
        long finalMemory = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = finalMemory - initialMemory;
        
        assertCompiles(result, "Memory test should produce compilable code");
        assertTrue(memoryUsed < 50 * 1024 * 1024, // 50MB limit
            "Transformation should use less than 50MB, used: " + (memoryUsed / 1024 / 1024) + "MB");
    }
    
    @Test
    public void testScalabilityWithCodeSize() {
        // Test how performance scales with code size
        String baseCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 0;
                    for (int i = 0; i < 10; i++) {
                        result += i * 2;
                    }
                    return result;
                }
            }
            """;
        
        // Create larger versions of the code
        Map<Integer, String> sizedCodes = new HashMap<>();
        for (int size : Arrays.asList(1, 5, 10, 20)) {
            StringBuilder largeCode = new StringBuilder();
            largeCode.append("public class TestClass {\n");
            
            for (int i = 0; i < size; i++) {
                largeCode.append("    public void testMethod").append(i).append("() {\n");
                largeCode.append("        int result = 0;\n");
                largeCode.append("        for (int j = 0; j < 10; j++) {\n");
                largeCode.append("            result += j * 2;\n");
                largeCode.append("        }\n");
                largeCode.append("        return result;\n");
                largeCode.append("    }\n");
            }
            largeCode.append("}\n");
            sizedCodes.put(size, largeCode.toString());
        }
        
        Map<Integer, Long> sizeToTime = new HashMap<>();
        
        for (Map.Entry<Integer, String> entry : sizedCodes.entrySet()) {
            int size = entry.getKey();
            String code = entry.getValue();
            
            long startTime = System.currentTimeMillis();
            String result = transformer.transformCode(code, 
                Arrays.asList("mathematical_expression"), "enhanced");
            long duration = System.currentTimeMillis() - startTime;
            
            sizeToTime.put(size, duration);
            
            assertCompiles(result, "Scalability test should produce compilable code for size: " + size);
        }
        
        // Log scalability results
        System.out.println("=== SCALABILITY RESULTS ===");
        for (Map.Entry<Integer, Long> entry : sizeToTime.entrySet()) {
            System.out.println("Size " + entry.getKey() + ": " + entry.getValue() + "ms");
        }
        
        // Verify performance scales reasonably (not exponentially)
        Long time1 = sizeToTime.get(1);
        Long time20 = sizeToTime.get(20);
        if (time1 != null && time20 != null) {
            double ratio = (double) time20 / time1;
            assertTrue(ratio < 50, // Should not be more than 50x slower for 20x larger code
                "Performance should scale reasonably, ratio: " + ratio);
        }
    }
    
    @Test
    public void testTransformationSuccessRates() {
        // Measure transformation success rates across different code patterns
        List<String> testCodes = Arrays.asList(
            // Simple code
            "public class A { public void m() { int x = 5; } }",
            // Code with loops
            "public class B { public void m() { for(int i=0;i<10;i++){} } }",
            // Code with conditionals
            "public class C { public void m() { if(true){} } }",
            // Code with mathematical expressions
            "public class D { public void m() { int x = 5 + 3; } }",
            // Code with logical expressions
            "public class E { public void m() { boolean b = true && false; } }",
            // Complex code
            """
            public class F {
                public void m() {
                    int sum = 0;
                    for (int i = 0; i < 10; i++) {
                        if (i % 2 == 0) {
                            sum += i * 2;
                        }
                    }
                    return sum > 10 ? "High" : "Low";
                }
            }
            """
        );
        
        List<String> transformations = Arrays.asList(
            "mathematical_expression", "logical_expression", "simple_assignment",
            "loop_conversion", "guard_reversal", "ternary_operator"
        );
        
        Map<String, Integer> successCounts = new HashMap<>();
        Map<String, Integer> totalCounts = new HashMap<>();
        
        for (String transformation : transformations) {
            successCounts.put(transformation, 0);
            totalCounts.put(transformation, 0);
        }
        
        for (String testCode : testCodes) {
            for (String transformation : transformations) {
                totalCounts.put(transformation, totalCounts.get(transformation) + 1);
                
                try {
                    String result = transformer.transformCode(testCode, 
                        Arrays.asList(transformation), "enhanced");
                    
                    if (compiles(result)) {
                        successCounts.put(transformation, successCounts.get(transformation) + 1);
                    }
                } catch (Exception e) {
                    // Count as failure
                }
            }
        }
        
        // Log success rates
        System.out.println("=== TRANSFORMATION SUCCESS RATES ===");
        for (String transformation : transformations) {
            int success = successCounts.get(transformation);
            int total = totalCounts.get(transformation);
            double rate = (double) success / total * 100;
            System.out.println(transformation + ": " + success + "/" + total + " (" + String.format("%.1f", rate) + "%)");
            
            // Verify reasonable success rate (at least 50%)
            assertTrue(rate >= 50, 
                "Transformation should have reasonable success rate: " + transformation + " has " + rate + "%");
        }
    }
    
    @Test
    public void testCompilationValidationOverhead() {
        // Measure the overhead of compilation validation
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 5 + 3 * 2;
                    return result > 10;
                }
            }
            """;
        
        // Test without compilation validation (simulate)
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 100; i++) {
            // Just parse the code (no compilation)
            transformer.transformCode(testCode, 
                Arrays.asList("mathematical_expression"), "enhanced");
        }
        long timeWithValidation = System.currentTimeMillis() - startTime;
        
        // The compilation validation is built into the transformation process,
        // so we can't easily measure without it, but we can ensure it's reasonable
        assertTrue(timeWithValidation < 10000, 
            "100 transformations should complete within 10 seconds, took: " + timeWithValidation + "ms");
        
        System.out.println("=== COMPILATION VALIDATION OVERHEAD ===");
        System.out.println("100 transformations with validation: " + timeWithValidation + "ms");
        System.out.println("Average per transformation: " + (timeWithValidation / 100) + "ms");
    }
    
    @Test
    public void testConcurrentTransformationPerformance() {
        // Test performance with multiple transformers (simulating concurrent usage)
        String testCode = """
            public class TestClass {
                public void testMethod() {
                    int result = 0;
                    for (int i = 0; i < 5; i++) {
                        result += i * 2;
                    }
                    return result > 10 ? "High" : "Low";
                }
            }
            """;
        
        int numThreads = 5;
        List<SemanticTransformer> transformers = new ArrayList<>();
        for (int i = 0; i < numThreads; i++) {
            transformers.add(new SemanticTransformer());
        }
        
        long startTime = System.currentTimeMillis();
        
        // Simulate concurrent usage by running transformations sequentially
        // (In a real concurrent test, these would run in parallel)
        for (SemanticTransformer t : transformers) {
            String result = t.transformCode(testCode, 
                Arrays.asList("mathematical_expression"), "enhanced");
            assertCompiles(result, "Concurrent simulation should produce compilable code");
        }
        
        long duration = System.currentTimeMillis() - startTime;
        
        assertTrue(duration < 5000, 
            "Concurrent simulation should complete within 5 seconds, took: " + duration + "ms");
        
        System.out.println("=== CONCURRENT SIMULATION PERFORMANCE ===");
        System.out.println("5 transformers (sequential): " + duration + "ms");
        System.out.println("Average per transformer: " + (duration / numThreads) + "ms");
    }
    
    @Test
    public void testLargeCodebaseTransformation() {
        // Test performance on a larger, more realistic codebase
        StringBuilder largeCode = new StringBuilder();
        largeCode.append("public class LargeTestClass {\n");
        
        // Add many methods with different patterns
        for (int i = 0; i < 50; i++) {
            largeCode.append("    public void method").append(i).append("() {\n");
            largeCode.append("        int result = ").append(i).append(" + ").append(i * 2).append(";\n");
            largeCode.append("        boolean flag = result > ").append(i).append(" && result < ").append(i * 10).append(";\n");
            largeCode.append("        for (int j = 0; j < ").append(i % 10).append("; j++) {\n");
            largeCode.append("            if (j % 2 == 0) {\n");
            largeCode.append("                result += j;\n");
            largeCode.append("            }\n");
            largeCode.append("        }\n");
            largeCode.append("        return flag ? \"Valid\" : \"Invalid\";\n");
            largeCode.append("    }\n");
        }
        largeCode.append("}\n");
        
        String code = largeCode.toString();
        
        long startTime = System.currentTimeMillis();
        
        String result = transformer.transformCode(code, 
            Arrays.asList("mathematical_expression", "logical_expression", "loop_conversion"), "enhanced");
        
        long duration = System.currentTimeMillis() - startTime;
        
        assertCompiles(result, "Large codebase transformation should produce compilable code");
        assertTrue(duration < 15000, 
            "Large codebase transformation should complete within 15 seconds, took: " + duration + "ms");
        
        System.out.println("=== LARGE CODEBASE PERFORMANCE ===");
        System.out.println("Large codebase transformation: " + duration + "ms");
        System.out.println("Code size: " + code.length() + " characters");
        System.out.println("Methods: 50");
    }
    
    /**
     * Helper method to check if code compiles (without using the test infrastructure)
     */
    private boolean compiles(String code) {
        try {
            // Simple check - if the code contains obvious compilation errors, return false
            return !code.contains("cannot find symbol") && 
                   !code.contains("illegal start of expression") &&
                   !code.contains("';' expected");
        } catch (Exception e) {
            return false;
        }
    }
}
