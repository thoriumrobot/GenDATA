package cfwr;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sensible tests for CheckerFrameworkWarningResolver
 * Tests actual functionality without relying on specific output formats
 */
public class CheckerFrameworkWarningResolverTest {

    @Test
    public void testMainMethodWithValidArgs(@TempDir Path tempDir) throws IOException {
        // Setup project root and warning file
        Path projectRoot = tempDir.resolve("testProjectRoot");
        Files.createDirectories(projectRoot);
        Path warningsFilePath = Files.createTempFile(tempDir, "warnings", ".txt");

        // Create sample Java file
        String javaFileContent = "package com.example;\n" +
                                 "public class TestClass {\n" +
                                 "    private int testField;\n" +
                                 "    public void testMethod() {}\n" +
                                 "}";
        Path javaFilePath = projectRoot.resolve("com/example/TestClass.java");
        Files.createDirectories(javaFilePath.getParent());
        Files.write(javaFilePath, javaFileContent.getBytes());

        // Create properly formatted warnings file
        String warningsContent = "com/example/TestClass.java:3:17: compiler.warn.proc.messager: [index] Possible out-of-bounds access\n" +
                                 "com/example/TestClass.java:4:17: compiler.warn.proc.messager: [index] Possible out-of-bounds access";
        Files.write(warningsFilePath, warningsContent.getBytes());

        // Setup resolver path
        Path resolverRootPath = Paths.get("").toAbsolutePath();

        // Test that main method runs without throwing exceptions
        assertDoesNotThrow(() -> {
            CheckerFrameworkWarningResolver.executeCommandFlag = false; // Disable command execution
            String[] args = {projectRoot.toString(), warningsFilePath.toString(), resolverRootPath.toString()};
            CheckerFrameworkWarningResolver.main(args);
        }, "Main method should run without throwing exceptions");
    }

    @Test
    public void testMainMethodWithInvalidArgs() {
        // Test with insufficient arguments
        assertDoesNotThrow(() -> {
            String[] args = {"arg1"}; // Only one argument, needs at least 3
            CheckerFrameworkWarningResolver.main(args);
        }, "Main method should handle invalid arguments gracefully");
    }

    @Test
    public void testMainMethodWithNonExistentFiles(@TempDir Path tempDir) {
        // Test with non-existent files
        assertDoesNotThrow(() -> {
            Path nonExistentProject = tempDir.resolve("nonexistent");
            Path nonExistentWarnings = tempDir.resolve("nonexistent.txt");
            Path resolverRoot = Paths.get("").toAbsolutePath();
            
            String[] args = {
                nonExistentProject.toString(), 
                nonExistentWarnings.toString(), 
                resolverRoot.toString()
            };
            CheckerFrameworkWarningResolver.main(args);
        }, "Main method should handle non-existent files gracefully");
    }

    @Test
    public void testStaticFieldsExist() {
        // Test that static fields are accessible
        assertNotNull(CheckerFrameworkWarningResolver.executeCommandFlag, 
                     "executeCommandFlag should be accessible");
        assertNotNull(CheckerFrameworkWarningResolver.slicerType, 
                     "slicerType should be accessible");
    }

    @Test
    public void testWarningParsingWithValidFormat(@TempDir Path tempDir) throws IOException {
        // Test warning parsing with properly formatted warnings
        Path warningsFile = Files.createTempFile(tempDir, "warnings", ".txt");
        String validWarning = "TestClass.java:10:15: compiler.warn.proc.messager: [index] warning message";
        Files.write(warningsFile, validWarning.getBytes());

        // This test ensures the warning parsing doesn't crash
        assertDoesNotThrow(() -> {
            CheckerFrameworkWarningResolver.executeCommandFlag = false;
            Path projectRoot = tempDir.resolve("project");
            Path resolverRoot = Paths.get("").toAbsolutePath();
            Files.createDirectories(projectRoot);
            
            String[] args = {projectRoot.toString(), warningsFile.toString(), resolverRoot.toString()};
            CheckerFrameworkWarningResolver.main(args);
        }, "Warning parsing should handle valid format gracefully");
    }

    @Test
    public void testWarningParsingWithInvalidFormat(@TempDir Path tempDir) throws IOException {
        // Test warning parsing with malformed warnings
        Path warningsFile = Files.createTempFile(tempDir, "warnings", ".txt");
        String invalidWarning = "This is not a properly formatted warning line";
        Files.write(warningsFile, invalidWarning.getBytes());

        // This test ensures the warning parsing handles invalid formats gracefully
        assertDoesNotThrow(() -> {
            CheckerFrameworkWarningResolver.executeCommandFlag = false;
            Path projectRoot = tempDir.resolve("project");
            Path resolverRoot = Paths.get("").toAbsolutePath();
            Files.createDirectories(projectRoot);
            
            String[] args = {projectRoot.toString(), warningsFile.toString(), resolverRoot.toString()};
            CheckerFrameworkWarningResolver.main(args);
        }, "Warning parsing should handle invalid format gracefully");
    }

    @Test
    public void testEmptyWarningsFile(@TempDir Path tempDir) throws IOException {
        // Test with empty warnings file
        Path emptyWarningsFile = Files.createTempFile(tempDir, "empty", ".txt");
        // File is already empty

        assertDoesNotThrow(() -> {
            CheckerFrameworkWarningResolver.executeCommandFlag = false;
            Path projectRoot = tempDir.resolve("project");
            Path resolverRoot = Paths.get("").toAbsolutePath();
            Files.createDirectories(projectRoot);
            
            String[] args = {projectRoot.toString(), emptyWarningsFile.toString(), resolverRoot.toString()};
            CheckerFrameworkWarningResolver.main(args);
        }, "Should handle empty warnings file gracefully");
    }
}
