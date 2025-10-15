package cfwr.jdt.transformations.utils;

import cfwr.jdt.SemanticTransformer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.RegisterExtension;

import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Base class for transformation tests providing common infrastructure.
 * Includes SemanticTransformer setup, compilation validation, and test utilities.
 */
@ExtendWith(TestResultLogger.class)
public abstract class TransformationTestBase {
    
    protected SemanticTransformer transformer;
    protected CompilationValidator compilationValidator;
    protected SemanticEquivalenceChecker equivalenceChecker;
    protected TestResultLogger logger;
    
    // Fixed seed for reproducible tests
    private static final long FIXED_SEED = 42L;
    
    @BeforeEach
    public void setUp() {
        transformer = new SemanticTransformer(FIXED_SEED);
        compilationValidator = new CompilationValidator();
        equivalenceChecker = new SemanticEquivalenceChecker();
        logger = new TestResultLogger();
    }
    
    /**
     * Apply a single transformation to the given code.
     */
    protected String applyTransformation(String code, String transformation, String mode) {
        List<String> transformations = Arrays.asList(transformation);
        return transformer.transformCode(code, transformations, mode);
    }
    
    /**
     * Apply multiple transformations to the given code.
     */
    protected String applyTransformations(String code, List<String> transformations, String mode) {
        return transformer.transformCode(code, transformations, mode);
    }
    
    /**
     * Validate that the transformed code compiles successfully.
     */
    protected void assertCompiles(String code, String testName) {
        CompilationValidator.CompilationResult result = compilationValidator.compile(code);
        assert result.isSuccess() : String.format(
            "Code should compile for %s. Errors: %s", 
            testName, 
            result.getErrors().stream().collect(Collectors.joining("; "))
        );
    }
    
    /**
     * Validate that the transformed code does not compile (for error cases).
     */
    protected void assertDoesNotCompile(String code, String testName) {
        CompilationValidator.CompilationResult result = compilationValidator.compile(code);
        assert !result.isSuccess() : String.format(
            "Code should not compile for %s", testName
        );
    }
    
    /**
     * Check that transformation was actually applied (code changed).
     */
    protected void assertTransformationApplied(String original, String transformed, String testName) {
        assert !original.equals(transformed) : String.format(
            "Transformation should have been applied for %s", testName
        );
    }
    
    /**
     * Check that transformation was not applied (code unchanged).
     */
    protected void assertNoTransformation(String original, String transformed, String testName) {
        assert original.equals(transformed) : String.format(
            "No transformation should have been applied for %s", testName
        );
    }
    
    /**
     * Check semantic equivalence between original and transformed code.
     */
    protected void assertSemanticallyEquivalent(String original, String transformed, String testName) {
        boolean equivalent = equivalenceChecker.areEquivalent(original, transformed);
        assert equivalent : String.format(
            "Original and transformed code should be semantically equivalent for %s", testName
        );
    }
    
    /**
     * Create a complete Java class from a method body.
     */
    protected String createTestClass(String methodBody) {
        return String.format("""
            public class TestClass {
                %s
            }
            """, methodBody);
    }
    
    /**
     * Create a complete Java class with multiple methods.
     */
    protected String createTestClass(String... methodBodies) {
        StringBuilder sb = new StringBuilder("public class TestClass {\n");
        for (String method : methodBodies) {
            sb.append("    ").append(method).append("\n");
        }
        sb.append("}");
        return sb.toString();
    }
    
    /**
     * Create a complete Java class with imports and package declaration.
     */
    protected String createCompleteTestClass(String packageName, String[] imports, String classContent) {
        StringBuilder sb = new StringBuilder();
        if (packageName != null && !packageName.isEmpty()) {
            sb.append("package ").append(packageName).append(";\n\n");
        }
        if (imports != null) {
            for (String imp : imports) {
                sb.append("import ").append(imp).append(";\n");
            }
            sb.append("\n");
        }
        sb.append(classContent);
        return sb.toString();
    }
    
    /**
     * Log test execution details for debugging.
     */
    protected void logTestExecution(String testName, String original, String transformed, boolean success) {
        logger.logTestExecution(testName, original, transformed, success);
    }
    
    /**
     * Generate a unique test method name based on the transformation and case number.
     */
    protected String getTestMethodName(String transformation, int caseNumber, String description) {
        return String.format("test%s_Case%d_%s", 
            transformation.replace("_", ""), 
            caseNumber, 
            description.replace(" ", "_").replace("-", "_"));
    }
}
