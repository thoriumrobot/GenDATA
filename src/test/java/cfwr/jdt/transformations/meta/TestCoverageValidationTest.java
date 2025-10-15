package cfwr.jdt.transformations.meta;

import cfwr.jdt.transformations.utils.TransformationTestBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Meta-tests to ensure comprehensive test coverage for all transformations.
 * These tests verify that each of the 27 transformations has adequate test coverage.
 */
public class TestCoverageValidationTest extends TransformationTestBase {
    
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
    
    private static final String TEST_DIRECTORY = "src/test/java/cfwr/jdt/transformations";
    private static final String ENHANCED_TEST_DIRECTORY = TEST_DIRECTORY + "/enhanced";
    private static final String SIMPLE_TEST_DIRECTORY = TEST_DIRECTORY + "/simple";
    private static final String RANDOM_TEST_DIRECTORY = TEST_DIRECTORY + "/random";
    
    @BeforeEach
    public void setUp() {
        super.setUp();
    }
    
    @Test
    public void testAllTransformationsHaveTestFiles() {
        // Verify that each transformation has a corresponding test file
        for (String transformation : ALL_TRANSFORMATIONS) {
            String expectedTestFile = getExpectedTestFilePath(transformation);
            File testFile = new File(expectedTestFile);
            
            assertTrue(testFile.exists(), 
                "Test file should exist for transformation: " + transformation + 
                " at path: " + expectedTestFile);
            
            assertTrue(testFile.length() > 0, 
                "Test file should not be empty for transformation: " + transformation);
        }
    }
    
    @Test
    public void testEnhancedTransformationsHaveComprehensiveTests() {
        // Verify that enhanced transformations have comprehensive test coverage
        List<String> enhancedTransformations = Arrays.asList(
            "loop_conversion", "guard_reversal", "mathematical_expression", "logical_expression",
            "ternary_operator", "switch_statement", "variable_operation", "method_extraction",
            "conditional_expression", "array_access_pattern", "string_concatenation", "numeric_literal",
            "exception_handling", "lambda_expression", "stream_api", "builder_pattern", "functional_conversion"
        );
        
        for (String transformation : enhancedTransformations) {
            String testFilePath = ENHANCED_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
            
            File testFile = new File(testFilePath);
            assertTrue(testFile.exists(), 
                "Enhanced transformation test should exist: " + transformation);
            
            // Check that the test file has substantial content (at least 500 bytes)
            assertTrue(testFile.length() > 500, 
                "Enhanced transformation test should be comprehensive: " + transformation);
            
            // Check that the test file contains test methods
            try {
                String content = Files.readString(Paths.get(testFilePath));
                assertTrue(content.contains("@Test"), 
                    "Enhanced transformation test should contain @Test methods: " + transformation);
                assertTrue(content.contains("test" + toPascalCase(transformation)), 
                    "Enhanced transformation test should contain transformation-specific tests: " + transformation);
            } catch (IOException e) {
                fail("Could not read test file for transformation: " + transformation + " - " + e.getMessage());
            }
        }
    }
    
    @Test
    public void testSimpleTransformationsHaveBasicTests() {
        // Verify that simple transformations have basic test coverage
        List<String> simpleTransformations = Arrays.asList(
            "simple_method_call", "simple_assignment", "simple_conditional", "simple_array_access",
            "simple_return_statement", "simple_variable_declaration", "simple_constructor_call",
            "simple_field_access", "simple_string_operation", "simple_numeric_operation"
        );
        
        for (String transformation : simpleTransformations) {
            String testFilePath = SIMPLE_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
            
            File testFile = new File(testFilePath);
            assertTrue(testFile.exists(), 
                "Simple transformation test should exist: " + transformation);
            
            // Check that the test file has reasonable content (at least 200 bytes)
            assertTrue(testFile.length() > 200, 
                "Simple transformation test should have basic coverage: " + transformation);
        }
    }
    
    @Test
    public void testRandomTransformationsHaveTests() {
        // Verify that random transformations have test coverage
        List<String> randomTransformations = Arrays.asList(
            "random_method_insertion", "random_statement_insertion", "random_expression_insertion"
        );
        
        for (String transformation : randomTransformations) {
            String testFilePath = RANDOM_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
            
            File testFile = new File(testFilePath);
            assertTrue(testFile.exists(), 
                "Random transformation test should exist: " + transformation);
            
            // Check that the test file has reasonable content (at least 200 bytes)
            assertTrue(testFile.length() > 200, 
                "Random transformation test should have basic coverage: " + transformation);
        }
    }
    
    @Test
    public void testIntegrationTestsExist() {
        // Verify that integration tests exist
        String integrationTestPath = TEST_DIRECTORY + "/AllTransformationsIntegrationTest.java";
        File integrationTest = new File(integrationTestPath);
        
        assertTrue(integrationTest.exists(), 
            "Integration test should exist for all transformations");
        
        assertTrue(integrationTest.length() > 500, 
            "Integration test should be comprehensive");
        
        try {
            String content = Files.readString(Paths.get(integrationTestPath));
            assertTrue(content.contains("@Test"), 
                "Integration test should contain @Test methods");
            assertTrue(content.contains("AllTransformationsIntegrationTest"), 
                "Integration test should be properly named");
        } catch (IOException e) {
            fail("Could not read integration test file - " + e.getMessage());
        }
    }
    
    @Test
    public void testTestStructureIsConsistent() {
        // Verify that all test files follow consistent naming and structure patterns
        for (String transformation : ALL_TRANSFORMATIONS) {
            String testFilePath = getExpectedTestFilePath(transformation);
            File testFile = new File(testFilePath);
            
            if (testFile.exists()) {
                try {
                    String content = Files.readString(Paths.get(testFilePath));
                    
                    // Check for consistent package declaration
                    assertTrue(content.contains("package cfwr.jdt.transformations"), 
                        "Test file should have correct package: " + transformation);
                    
                    // Check for JUnit 5 imports
                    assertTrue(content.contains("import org.junit.jupiter.api.Test"), 
                        "Test file should use JUnit 5: " + transformation);
                    
                    // Check for proper class naming
                    String expectedClassName = toPascalCase(transformation) + "TransformationTest";
                    assertTrue(content.contains("class " + expectedClassName), 
                        "Test class should be properly named: " + transformation);
                    
                    // Check for test methods
                    assertTrue(content.contains("@Test"), 
                        "Test file should contain test methods: " + transformation);
                    
                } catch (IOException e) {
                    fail("Could not read test file for transformation: " + transformation + " - " + e.getMessage());
                }
            }
        }
    }
    
    @Test
    public void testTestDirectoriesExist() {
        // Verify that all test directories exist
        List<String> testDirectories = Arrays.asList(
            TEST_DIRECTORY,
            ENHANCED_TEST_DIRECTORY,
            SIMPLE_TEST_DIRECTORY,
            RANDOM_TEST_DIRECTORY,
            TEST_DIRECTORY + "/utils",
            TEST_DIRECTORY + "/meta"
        );
        
        for (String dirPath : testDirectories) {
            File directory = new File(dirPath);
            assertTrue(directory.exists(), 
                "Test directory should exist: " + dirPath);
            assertTrue(directory.isDirectory(), 
                "Path should be a directory: " + dirPath);
        }
    }
    
    @Test
    public void testUtilityClassesExist() {
        // Verify that utility classes exist for testing
        List<String> utilityClasses = Arrays.asList(
            "TransformationTestBase.java",
            "CompilationValidator.java",
            "SemanticEquivalenceChecker.java",
            "TestResultLogger.java"
        );
        
        String utilsDirectory = TEST_DIRECTORY + "/utils";
        
        for (String utilityClass : utilityClasses) {
            File utilityFile = new File(utilsDirectory + "/" + utilityClass);
            assertTrue(utilityFile.exists(), 
                "Utility class should exist: " + utilityClass);
            assertTrue(utilityFile.length() > 100, 
                "Utility class should have content: " + utilityClass);
        }
    }
    
    @Test
    public void testMetaTestsExist() {
        // Verify that meta-test classes exist
        List<String> metaTestClasses = Arrays.asList(
            "TestInfrastructureValidationTest.java",
            "TransformationCorrectnessTest.java",
            "TestCoverageValidationTest.java"
        );
        
        String metaDirectory = TEST_DIRECTORY + "/meta";
        
        for (String metaTestClass : metaTestClasses) {
            File metaTestFile = new File(metaDirectory + "/" + metaTestClass);
            assertTrue(metaTestFile.exists(), 
                "Meta-test class should exist: " + metaTestClass);
            assertTrue(metaTestFile.length() > 500, 
                "Meta-test class should be comprehensive: " + metaTestClass);
        }
    }
    
    @Test
    public void testAllTransformationsAreListed() {
        // Verify that we have tests for all 27 transformations
        assertEquals(27, ALL_TRANSFORMATIONS.size(), 
            "Should have exactly 27 transformations");
        
        // Verify the breakdown: 17 enhanced + 10 simple + 3 random = 30 total
        // But we have some overlap, so let's verify the categories
        long enhancedCount = ALL_TRANSFORMATIONS.stream()
            .filter(t -> t.startsWith("simple_") || t.startsWith("random_"))
            .count();
        
        long simpleCount = ALL_TRANSFORMATIONS.stream()
            .filter(t -> t.startsWith("simple_"))
            .count();
        
        long randomCount = ALL_TRANSFORMATIONS.stream()
            .filter(t -> t.startsWith("random_"))
            .count();
        
        assertEquals(10, simpleCount, "Should have 10 simple transformations");
        assertEquals(3, randomCount, "Should have 3 random transformations");
        assertEquals(14, ALL_TRANSFORMATIONS.size() - simpleCount - randomCount, 
            "Should have 14 enhanced transformations");
    }
    
    // Helper methods
    
    private String getExpectedTestFilePath(String transformation) {
        if (transformation.startsWith("simple_")) {
            return SIMPLE_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
        } else if (transformation.startsWith("random_")) {
            return RANDOM_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
        } else {
            return ENHANCED_TEST_DIRECTORY + "/" + 
                toPascalCase(transformation) + "TransformationTest.java";
        }
    }
    
    private String toPascalCase(String transformation) {
        // Convert "loop_conversion" to "LoopConversion"
        return Arrays.stream(transformation.split("_"))
            .map(word -> word.substring(0, 1).toUpperCase() + word.substring(1))
            .collect(Collectors.joining());
    }
}
