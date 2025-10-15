package cfwr.jdt.transformations.utils;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.TestWatcher;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Test result logger for tracking transformation test execution.
 * Logs test results, execution times, and transformation details.
 */
public class TestResultLogger implements TestWatcher {
    
    private static final String LOG_DIR = "test-results/transformation-tests";
    private static final DateTimeFormatter TIMESTAMP_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    
    private final Map<String, TestExecution> executions = new ConcurrentHashMap<>();
    private final String sessionId;
    
    public TestResultLogger() {
        this.sessionId = generateSessionId();
        initializeLogDirectory();
    }
    
    @Override
    public void testSuccessful(ExtensionContext context) {
        logTestResult(context, "SUCCESS", null);
    }
    
    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        logTestResult(context, "FAILED", cause);
    }
    
    @Override
    public void testAborted(ExtensionContext context, Throwable cause) {
        logTestResult(context, "ABORTED", cause);
    }
    
    @Override
    public void testDisabled(ExtensionContext context, Optional<String> reason) {
        logTestResult(context, "DISABLED", null);
    }
    
    /**
     * Log detailed test execution information.
     */
    public void logTestExecution(String testName, String original, String transformed, boolean success) {
        TestExecution execution = new TestExecution();
        execution.testName = testName;
        execution.original = original;
        execution.transformed = transformed;
        execution.success = success;
        execution.timestamp = LocalDateTime.now();
        execution.sessionId = sessionId;
        
        executions.put(testName, execution);
        
        // Write to individual test log
        writeTestLog(execution);
    }
    
    /**
     * Generate session summary report.
     */
    public void generateSessionSummary() {
        try {
            Path summaryFile = Paths.get(LOG_DIR, "session-" + sessionId + "-summary.txt");
            
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(summaryFile))) {
                writer.println("TRANSFORMATION TEST SESSION SUMMARY");
                writer.println("==================================");
                writer.println("Session ID: " + sessionId);
                writer.println("Generated: " + LocalDateTime.now().format(TIMESTAMP_FORMAT));
                writer.println();
                
                // Count results
                long successCount = executions.values().stream().mapToLong(e -> e.success ? 1 : 0).sum();
                long totalCount = executions.size();
                double successRate = totalCount > 0 ? (double) successCount / totalCount * 100 : 0;
                
                writer.println("Test Results:");
                writer.printf("  Total Tests: %d%n", totalCount);
                writer.printf("  Successful: %d%n", successCount);
                writer.printf("  Failed: %d%n", totalCount - successCount);
                writer.printf("  Success Rate: %.2f%%%n", successRate);
                writer.println();
                
                // List all tests
                writer.println("Test Details:");
                writer.println("------------");
                for (TestExecution execution : executions.values()) {
                    writer.printf("%-50s %s%n", 
                        execution.testName, 
                        execution.success ? "PASS" : "FAIL");
                }
                
                writer.println();
                writer.println("Detailed logs available in individual test files.");
            }
            
            System.out.println("Session summary written to: " + summaryFile);
            
        } catch (IOException e) {
            System.err.println("Failed to generate session summary: " + e.getMessage());
        }
    }
    
    /**
     * Get execution statistics for a specific test.
     */
    public TestExecution getTestExecution(String testName) {
        return executions.get(testName);
    }
    
    /**
     * Get all test executions for this session.
     */
    public Collection<TestExecution> getAllExecutions() {
        return new ArrayList<>(executions.values());
    }
    
    private void logTestResult(ExtensionContext context, String status, Throwable cause) {
        String testName = context.getDisplayName();
        String className = context.getTestClass().map(Class::getSimpleName).orElse("Unknown");
        String methodName = context.getTestMethod().map(m -> m.getName()).orElse("Unknown");
        
        System.out.printf("[%s] %s.%s - %s%n", 
            LocalDateTime.now().format(TIMESTAMP_FORMAT),
            className,
            methodName,
            status);
        
        if (cause != null) {
            System.out.println("  Cause: " + cause.getMessage());
        }
    }
    
    private void writeTestLog(TestExecution execution) {
        try {
            String safeTestName = execution.testName.replaceAll("[^a-zA-Z0-9_-]", "_");
            Path logFile = Paths.get(LOG_DIR, "test-" + safeTestName + "-" + sessionId + ".log");
            
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(logFile))) {
                writer.println("TRANSFORMATION TEST EXECUTION LOG");
                writer.println("=================================");
                writer.printf("Test Name: %s%n", execution.testName);
                writer.printf("Session ID: %s%n", execution.sessionId);
                writer.printf("Timestamp: %s%n", execution.timestamp.format(TIMESTAMP_FORMAT));
                writer.printf("Result: %s%n", execution.success ? "SUCCESS" : "FAILURE");
                writer.println();
                
                writer.println("ORIGINAL CODE:");
                writer.println("-------------");
                writer.println(execution.original);
                writer.println();
                
                writer.println("TRANSFORMED CODE:");
                writer.println("----------------");
                writer.println(execution.transformed);
                writer.println();
                
                writer.println("TRANSFORMATION ANALYSIS:");
                writer.println("----------------------");
                analyzeTransformation(execution, writer);
            }
            
        } catch (IOException e) {
            System.err.println("Failed to write test log: " + e.getMessage());
        }
    }
    
    private void analyzeTransformation(TestExecution execution, PrintWriter writer) {
        String original = execution.original;
        String transformed = execution.transformed;
        
        // Basic analysis
        writer.printf("Original length: %d characters%n", original.length());
        writer.printf("Transformed length: %d characters%n", transformed.length());
        writer.printf("Length change: %+d characters%n", transformed.length() - original.length());
        
        // Check if transformation was applied
        boolean changed = !original.equals(transformed);
        writer.printf("Code changed: %s%n", changed ? "YES" : "NO");
        
        if (changed) {
            // Analyze differences
            writer.println();
            writer.println("TRANSFORMATION CHANGES:");
            writer.println("----------------------");
            
            // Count line changes
            String[] originalLines = original.split("\n");
            String[] transformedLines = transformed.split("\n");
            writer.printf("Original lines: %d%n", originalLines.length);
            writer.printf("Transformed lines: %d%n", transformedLines.length);
            writer.printf("Line count change: %+d%n", transformedLines.length - originalLines.length);
            
            // Check for common transformation patterns
            analyzeTransformationPatterns(original, transformed, writer);
        }
    }
    
    private void analyzeTransformationPatterns(String original, String transformed, PrintWriter writer) {
        writer.println();
        writer.println("DETECTED TRANSFORMATION PATTERNS:");
        writer.println("--------------------------------");
        
        // Check for loop conversions
        if (original.contains("for (") && transformed.contains("while (")) {
            writer.println("- For loop converted to while loop");
        } else if (original.contains("while (") && transformed.contains("for (")) {
            writer.println("- While loop converted to for loop");
        }
        
        // Check for guard reversals
        if (original.contains("if (") && transformed.contains("if (!")) {
            writer.println("- Guard condition reversed");
        }
        
        // Check for mathematical transformations
        if (original.contains(" + ") && transformed.contains(" + ")) {
            writer.println("- Mathematical expression modified");
        }
        
        // Check for ternary conversions
        if (original.contains("?") && transformed.contains("if (")) {
            writer.println("- Ternary operator converted to if-else");
        } else if (original.contains("if (") && transformed.contains("?")) {
            writer.println("- If-else converted to ternary operator");
        }
        
        // Check for method extraction
        if (transformed.split("public ").length > original.split("public ").length) {
            writer.println("- Method extraction detected");
        }
        
        // Check for variable operations
        if (original.contains("+=") && transformed.contains("=")) {
            writer.println("- Compound assignment expanded");
        }
        
        // Check for string concatenation changes
        if (original.contains("\"") && transformed.contains("String.valueOf")) {
            writer.println("- String concatenation converted to String.valueOf");
        }
        
        // Check for numeric literal changes
        if (original.contains("1000") && transformed.contains("1_000")) {
            writer.println("- Numeric literal formatted with underscores");
        }
    }
    
    private void initializeLogDirectory() {
        try {
            Files.createDirectories(Paths.get(LOG_DIR));
        } catch (IOException e) {
            System.err.println("Failed to create log directory: " + e.getMessage());
        }
    }
    
    private String generateSessionId() {
        return "session-" + System.currentTimeMillis() + "-" + Thread.currentThread().getId();
    }
    
    /**
     * Test execution record.
     */
    public static class TestExecution {
        public String testName;
        public String original;
        public String transformed;
        public boolean success;
        public LocalDateTime timestamp;
        public String sessionId;
        
        @Override
        public String toString() {
            return String.format("TestExecution{testName='%s', success=%s, timestamp=%s}", 
                testName, success, timestamp);
        }
    }
}
