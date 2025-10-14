package cfwr.jdt;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Unit tests for JdtParserService
 */
public class JdtParserServiceTest {
    
    @TempDir
    Path tempDir;
    
    private String testJavaCode;
    private String testWarningsContent;
    
    @BeforeEach
    void setUp() {
        testJavaCode = """
            public class TestClass {
                public void testMethod(int x) {
                    if (x > 0) {
                        System.out.println("Positive: " + x);
                    } else {
                        System.out.println("Non-positive: " + x);
                    }
                    
                    for (int i = 0; i < x; i++) {
                        System.out.println("Iteration: " + i);
                    }
                }
            }
            """;
        
        testWarningsContent = """
            TestClass.java:3:5: compiler.warn.proc.messager: [nullness] potential null pointer dereference
            TestClass.java:7:9: compiler.err.proc.messager: [nullness] null assignment to non-null field
            """;
    }
    
    @Test
    void testParseCodeLocations() throws IOException {
        // Create test files
        Path javaFile = tempDir.resolve("TestClass.java");
        Path outputFile = tempDir.resolve("output.json");
        
        Files.writeString(javaFile, testJavaCode);
        
        // Run JdtParserService
        String[] args = {
            "--operation", "parse-code-locations",
            "--input", javaFile.toString(),
            "--output", outputFile.toString()
        };
        
        JdtParserService.main(args);
        
        // Verify output
        assertTrue(Files.exists(outputFile), "Output file should be created");
        
        String output = Files.readString(outputFile);
        assertFalse(output.trim().isEmpty(), "Output should not be empty");
        assertTrue(output.contains("lineStart"), "Output should contain lineStart");
        assertTrue(output.contains("locationType"), "Output should contain locationType");
    }
    
    @Test
    void testParseWarnings() throws IOException {
        // Create test files
        Path warningsFile = tempDir.resolve("warnings.txt");
        Path outputFile = tempDir.resolve("warnings_output.json");
        
        Files.writeString(warningsFile, testWarningsContent);
        
        // Run JdtParserService
        String[] args = {
            "--operation", "parse-warnings",
            "--input", warningsFile.toString(),
            "--output", outputFile.toString()
        };
        
        JdtParserService.main(args);
        
        // Verify output
        assertTrue(Files.exists(outputFile), "Output file should be created");
        
        String output = Files.readString(outputFile);
        assertFalse(output.trim().isEmpty(), "Output should not be empty");
        assertTrue(output.contains("filePath"), "Output should contain filePath");
        assertTrue(output.contains("line"), "Output should contain line");
    }
    
    @Test
    void testParseIdentifiers() throws IOException {
        // Create test files
        Path javaFile = tempDir.resolve("TestClass.java");
        Path outputFile = tempDir.resolve("identifiers_output.json");
        
        Files.writeString(javaFile, testJavaCode);
        
        // Run JdtParserService
        String[] args = {
            "--operation", "parse-identifiers",
            "--input", javaFile.toString(),
            "--output", outputFile.toString()
        };
        
        JdtParserService.main(args);
        
        // Verify output
        assertTrue(Files.exists(outputFile), "Output file should be created");
        
        String output = Files.readString(outputFile);
        assertFalse(output.trim().isEmpty(), "Output should not be empty");
        assertTrue(output.contains("variables"), "Output should contain variables");
        assertTrue(output.contains("methods"), "Output should contain methods");
    }
    
    @Test
    void testValidateSyntax() throws IOException {
        // Create test files
        Path javaFile = tempDir.resolve("TestClass.java");
        Path outputFile = tempDir.resolve("validation_output.json");
        
        Files.writeString(javaFile, testJavaCode);
        
        // Run JdtParserService
        String[] args = {
            "--operation", "validate-syntax",
            "--input", javaFile.toString(),
            "--output", outputFile.toString()
        };
        
        JdtParserService.main(args);
        
        // Verify output
        assertTrue(Files.exists(outputFile), "Output file should be created");
        
        String output = Files.readString(outputFile);
        assertFalse(output.trim().isEmpty(), "Output should not be empty");
        assertTrue(output.contains("valid"), "Output should contain valid field");
    }
    
    @Test
    void testInvalidOperation() {
        String[] args = {
            "--operation", "invalid-operation",
            "--input", "test.java",
            "--output", "test.json"
        };
        
        // Should exit with error code
        assertThrows(RuntimeException.class, () -> {
            JdtParserService.main(args);
        });
    }
    
    @Test
    void testMissingParameters() {
        String[] args = {
            "--operation", "parse-code-locations"
            // Missing required parameters
        };
        
        // Should exit with error code
        assertThrows(RuntimeException.class, () -> {
            JdtParserService.main(args);
        });
    }
}
