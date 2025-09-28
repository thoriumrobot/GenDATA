package cfwr;

import cfwr.CheckerFrameworkWarningResolver; // Added import statement

import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.utils.SourceRoot;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CheckerFrameworkWarningResolverTest {

    private final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();
    private PrintStream originalOut;

    @BeforeEach
    public void setUp() {
        originalOut = System.out;
        System.setOut(new PrintStream(outputStreamCaptor));
    }

    @AfterEach
    public void tearDown() {
        System.setOut(originalOut);
    }

    @Test
    public void testProcessWarnings(@TempDir Path tempDir) throws IOException {
        CheckerFrameworkWarningResolver.executeCommandFlag = false; // Disable command execution
        
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

        // Create sample warnings file
        String warningsContent = "com/example/TestClass.java:3:5: compiler.err.proc.messager: [index] some warning\n" +
                                 "com/example/TestClass.java:4:5: compiler.err.proc.messager: [index] some warning";
        Files.write(warningsFilePath, warningsContent.getBytes());

        // Setup resolver path to be the root directory of the CheckerFrameworkWarningResolver project
        Path resolverRootPath = Paths.get("").toAbsolutePath();

        // Run the CheckerFrameworkWarningResolver
        String[] args = {projectRoot.toString(), warningsFilePath.toString(), resolverRootPath.toString()+"/"};
        CheckerFrameworkWarningResolver.main(args);

        // Verify the output - the resolver now generates WALA slicer commands
        String output = outputStreamCaptor.toString().trim();
        // Check that the output contains WALA slicer commands
        assert(output.contains("java -jar"));
        assert(output.contains("wala-slicer-all.jar"));
        assert(output.contains("--targetMethod com.example.TestClass#testMethod()"));
    }
}
