package cfwr.jdt.transformations.utils;

import javax.tools.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Validates that transformed Java code compiles successfully.
 * Uses javax.tools.JavaCompiler to compile code snippets in memory.
 */
public class CompilationValidator {
    
    private final JavaCompiler compiler;
    private final StandardJavaFileManager fileManager;
    
    public CompilationValidator() {
        this.compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            throw new RuntimeException("Java compiler not available. Make sure to run with JDK, not JRE.");
        }
        this.fileManager = compiler.getStandardFileManager(null, null, null);
    }
    
    /**
     * Compile the given Java source code and return compilation result.
     */
    public CompilationResult compile(String sourceCode) {
        return compile(sourceCode, "TestClass.java");
    }
    
    /**
     * Compile the given Java source code with specified filename.
     */
    public CompilationResult compile(String sourceCode, String fileName) {
        try {
            // Create temporary directory for compilation
            Path tempDir = Files.createTempDirectory("transformation_test_");
            Path sourceFile = tempDir.resolve(fileName);
            
            // Write source code to file
            Files.write(sourceFile, sourceCode.getBytes());
            
            // Set up compilation options for Java 21
            List<String> options = Arrays.asList(
                "-source", "21",
                "-target", "21",
                "-cp", System.getProperty("java.class.path"),
                "-d", tempDir.toString()
            );
            
            // Create compilation task
            Iterable<? extends JavaFileObject> compilationUnits = 
                fileManager.getJavaFileObjects(sourceFile.toFile());
            
            StringWriter errorWriter = new StringWriter();
            JavaCompiler.CompilationTask task = compiler.getTask(
                errorWriter, 
                fileManager, 
                null, 
                options, 
                null, 
                compilationUnits
            );
            
            // Execute compilation
            boolean success = task.call();
            
            // Collect diagnostics - simplified for now
            List<String> errors = new ArrayList<>();
            List<String> warnings = new ArrayList<>();
            
            // If compilation failed, add the error output
            if (!success) {
                errors.add(errorWriter.toString());
            }
            
            // Clean up temporary directory
            deleteRecursively(tempDir);
            
            return new CompilationResult(success, errors, warnings, errorWriter.toString());
            
        } catch (Exception e) {
            return new CompilationResult(false, 
                Arrays.asList("Compilation failed: " + e.getMessage()), 
                Collections.emptyList(), 
                e.toString());
        }
    }
    
    /**
     * Validate that code compiles and return boolean result.
     */
    public boolean isValid(String sourceCode) {
        return compile(sourceCode).isSuccess();
    }
    
    /**
     * Get compilation errors as a formatted string.
     */
    public String getCompilationErrors(String sourceCode) {
        CompilationResult result = compile(sourceCode);
        return result.getErrors().stream().collect(Collectors.joining("\n"));
    }
    
    private void deleteRecursively(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
                for (Path entry : stream) {
                    deleteRecursively(entry);
                }
            }
        }
        Files.delete(path);
    }
    
    /**
     * Result of a compilation attempt.
     */
    public static class CompilationResult {
        private final boolean success;
        private final List<String> errors;
        private final List<String> warnings;
        private final String errorOutput;
        
        public CompilationResult(boolean success, List<String> errors, List<String> warnings, String errorOutput) {
            this.success = success;
            this.errors = errors != null ? new ArrayList<>(errors) : new ArrayList<>();
            this.warnings = warnings != null ? new ArrayList<>(warnings) : new ArrayList<>();
            this.errorOutput = errorOutput != null ? errorOutput : "";
        }
        
        public boolean isSuccess() {
            return success;
        }
        
        public List<String> getErrors() {
            return new ArrayList<>(errors);
        }
        
        public List<String> getWarnings() {
            return new ArrayList<>(warnings);
        }
        
        public String getErrorOutput() {
            return errorOutput;
        }
        
        public boolean hasErrors() {
            return !errors.isEmpty();
        }
        
        public boolean hasWarnings() {
            return !warnings.isEmpty();
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("CompilationResult{success=").append(success);
            if (!errors.isEmpty()) {
                sb.append(", errors=").append(errors);
            }
            if (!warnings.isEmpty()) {
                sb.append(", warnings=").append(warnings);
            }
            sb.append("}");
            return sb.toString();
        }
    }
}
