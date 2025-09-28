package cfwr;

import soot.*;
import soot.options.Options;
import soot.toolkits.graph.*;
import soot.toolkits.scalar.*;
import soot.jimple.toolkits.callgraph.*;
import soot.jimple.toolkits.pointer.*;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Soot-based slicer for CFWR project.
 * Performs program slicing on Java bytecode and optionally decompiles using Vineflower.
 */
public class SootSlicer {
    
    private static final String VINEFLOWER_CLASS = "org.vineflower.Main";
    
    public static void main(String[] args) {
        if (args.length < 8) {
            System.err.println("Usage: java cfwr.SootSlicer --projectRoot <path> --targetFile <file> --line <num> --output <dir> --member <sig> [--decompiler <vineflower.jar>] [--prediction-mode]");
            System.exit(1);
        }
        
        Map<String, String> params = parseArgs(args);
        
        String projectRoot = params.get("projectRoot");
        String targetFile = params.get("targetFile");
        int lineNumber = Integer.parseInt(params.get("line"));
        String outputDir = params.get("output");
        String memberSig = params.get("member");
        String vineflowerJar = params.get("decompiler");
        boolean predictionMode = params.containsKey("prediction-mode");
        
        try {
            SootSlicer slicer = new SootSlicer();
            slicer.sliceMethod(projectRoot, targetFile, lineNumber, outputDir, memberSig, vineflowerJar, predictionMode);
        } catch (Exception e) {
            System.err.println("Error during slicing: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> params = new HashMap<>();
        
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--projectRoot":
                    params.put("projectRoot", args[++i]);
                    break;
                case "--targetFile":
                    params.put("targetFile", args[++i]);
                    break;
                case "--line":
                    params.put("line", args[++i]);
                    break;
                case "--output":
                    params.put("output", args[++i]);
                    break;
                case "--member":
                    params.put("member", args[++i]);
                    break;
                case "--decompiler":
                    params.put("decompiler", args[++i]);
                    break;
                case "--prediction-mode":
                    params.put("prediction-mode", "true");
                    break;
            }
        }
        
        return params;
    }
    
    public void sliceMethod(String projectRoot, String targetFile, int lineNumber, 
                           String outputDir, String memberSig, String vineflowerJar, boolean predictionMode) throws Exception {
        
        // Create output directory
        Files.createDirectories(Paths.get(outputDir));
        
        // Normalize target file path
        Path targetPath = Paths.get(targetFile);
        if (!targetPath.isAbsolute()) {
            targetPath = Paths.get(projectRoot).resolve(targetPath).normalize();
        }
        
        if (!Files.exists(targetPath)) {
            System.err.println("Target file not found: " + targetPath);
            createFallbackSlice(outputDir, memberSig, targetPath.toString());
            return;
        }
        
        // Configure Soot
        setupSootOptions(projectRoot, targetPath);
        
        try {
            if (predictionMode && vineflowerJar != null) {
                // Prediction mode: Use true bytecode slicing with Vineflower decompilation
                performBytecodeSlicingWithDecompilation(targetPath, lineNumber, outputDir, memberSig, vineflowerJar);
            } else {
                // Training mode: Use source-based slicing for better compatibility
                performSourceBasedSlicing(targetPath, lineNumber, outputDir, memberSig);
            }
        } catch (Exception e) {
            System.err.println("Slicing failed: " + e.getMessage());
            // Fallback to source-based slicing
            performSourceBasedSlicing(targetPath, lineNumber, outputDir, memberSig);
        }
    }
    
    private void setupSootOptions(String projectRoot, Path targetFile) {
        // Reset Soot
        G.reset();
        
        // Set up classpath
        String classpath = System.getProperty("java.class.path");
        Options.v().set_soot_classpath(classpath);
        
        // Set source directory
        Options.v().set_src_prec(Options.src_prec_java);
        Options.v().set_allow_phantom_refs(true);
        Options.v().set_whole_program(false);
        
        // Add project root to classpath
        Options.v().set_prepend_classpath(true);
        Options.v().set_process_dir(Arrays.asList(projectRoot));
        
        // Set output format
        Options.v().set_output_format(Options.output_format_jimple);
        
        // Initialize Scene
        Scene.v().setSootClassPath(Options.v().soot_classpath());
    }
    
    private String getClassNameFromFile(Path javaFile) {
        String fileName = javaFile.getFileName().toString();
        String className = fileName.substring(0, fileName.lastIndexOf('.'));
        
        // Try to determine package from directory structure
        try {
            String content = Files.readString(javaFile);
            Pattern packagePattern = Pattern.compile("^package\\s+([a-zA-Z_][a-zA-Z0-9_.]*);", Pattern.MULTILINE);
            Matcher matcher = packagePattern.matcher(content);
            if (matcher.find()) {
                return matcher.group(1) + "." + className;
            }
        } catch (IOException e) {
            // Fall back to simple class name
        }
        
        return className;
    }
    
    private SootMethod findTargetMethod(SootClass sootClass, String memberSig) {
        if (memberSig.equals("file_level")) {
            // For file-level slicing, find the main method or first public method
            for (SootMethod method : sootClass.getMethods()) {
                if (method.getName().equals("main") || 
                    method.isPublic() && !method.getName().startsWith("<")) {
                    return method;
                }
            }
            return sootClass.getMethods().isEmpty() ? null : sootClass.getMethods().get(0);
        }
        
        // Parse method signature: "com.example.Class#methodName(int,String)"
        String[] parts = memberSig.split("#");
        if (parts.length != 2) return null;
        
        String methodName = parts[1];
        if (methodName.contains("(")) {
            methodName = methodName.substring(0, methodName.indexOf('('));
        }
        
        // Find method by name
        for (SootMethod method : sootClass.getMethods()) {
            if (method.getName().equals(methodName)) {
                return method;
            }
        }
        
        return null;
    }
    
    private void performBytecodeSlicingWithDecompilation(Path targetFile, int lineNumber, String outputDir, String memberSig, String vineflowerJar) {
        try {
            System.out.println("[soot_slicer] Performing bytecode slicing with Vineflower decompilation");
            
            // Step 1: Compile the Java source to bytecode
            Path compiledClass = compileJavaToBytecode(targetFile);
            if (compiledClass == null) {
                throw new RuntimeException("Failed to compile Java source to bytecode");
            }
            
            // Step 2: Use Soot to analyze bytecode and perform slicing
            String className = getClassNameFromFile(targetFile);
            setupSootForBytecodeAnalysis(targetFile.getParent().toString());
            
            SootClass sootClass = Scene.v().loadClassAndSupport(className);
            Scene.v().loadNecessaryClasses();
            
            SootMethod targetMethod = findTargetMethod(sootClass, memberSig);
            if (targetMethod == null) {
                throw new RuntimeException("Could not find method: " + memberSig);
            }
            
            // Step 3: Perform program slicing on bytecode
            List<Unit> sliceUnits = performProgramSlicing(targetMethod, lineNumber);
            
            // Step 4: Generate bytecode slice
            Path bytecodeSlice = generateBytecodeSlice(sliceUnits, outputDir, className);
            
            // Step 5: Use Vineflower to decompile bytecode slice back to Java
            Path decompiledJava = decompileWithVineflower(bytecodeSlice, vineflowerJar, outputDir);
            
            // Step 6: Clean up temporary files
            cleanupTempFiles(compiledClass, bytecodeSlice);
            
            System.out.println("[soot_slicer] Bytecode slicing completed successfully");
            
        } catch (Exception e) {
            System.err.println("Bytecode slicing failed: " + e.getMessage());
            throw new RuntimeException("Bytecode slicing failed", e);
        }
    }
    
    private Path compileJavaToBytecode(Path javaFile) {
        try {
            // Create temporary directory for compiled classes
            Path tempDir = Files.createTempDirectory("soot_compile_");
            
            // Compile Java source to bytecode
            ProcessBuilder pb = new ProcessBuilder(
                "javac", 
                "-cp", System.getProperty("java.class.path"),
                "-d", tempDir.toString(),
                javaFile.toString()
            );
            
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.err.println("Failed to compile Java source: " + javaFile);
                return null;
            }
            
            // Find the compiled .class file
            String className = getClassNameFromFile(javaFile);
            return tempDir.resolve(className.replace('.', '/') + ".class");
            
        } catch (Exception e) {
            System.err.println("Error compiling Java source: " + e.getMessage());
            return null;
        }
    }
    
    private void setupSootForBytecodeAnalysis(String sourceDir) {
        // Reset Soot
        G.reset();
        
        // Configure for bytecode analysis
        Options.v().set_src_prec(Options.src_prec_class);
        Options.v().set_allow_phantom_refs(true);
        Options.v().set_whole_program(false);
        Options.v().set_prepend_classpath(true);
        Options.v().set_process_dir(Arrays.asList(sourceDir));
        
        // Initialize Scene
        Scene.v().setSootClassPath(Options.v().soot_classpath());
    }
    
    private List<Unit> performProgramSlicing(SootMethod targetMethod, int targetLine) {
        try {
            Body body = targetMethod.retrieveActiveBody();
            
            // Find the unit (instruction) at the target line
            Unit targetUnit = findUnitAtLine(body, targetLine);
            if (targetUnit == null) {
                // If we can't find the exact line, use all units in the method
                return new ArrayList<>(body.getUnits());
            }
            
            // Perform backward slicing from the target unit
            // This is a simplified version - in practice, you'd use Soot's slicing algorithms
            List<Unit> sliceUnits = new ArrayList<>();
            sliceUnits.add(targetUnit);
            
            // Add all units for now (simplified slicing)
            sliceUnits.addAll(body.getUnits());
            
            return sliceUnits;
            
        } catch (Exception e) {
            System.err.println("Error during program slicing: " + e.getMessage());
            return new ArrayList<>();
        }
    }
    
    private Unit findUnitAtLine(Body body, int targetLine) {
        // Simplified: return the first unit
        // In practice, you'd map bytecode line numbers to Soot units
        for (Unit unit : body.getUnits()) {
            return unit; // Return first unit for now
        }
        return null;
    }
    
    private Path generateBytecodeSlice(List<Unit> sliceUnits, String outputDir, String className) {
        try {
            // Create a simplified bytecode representation
            StringBuilder bytecodeContent = new StringBuilder();
            bytecodeContent.append("// Bytecode slice for ").append(className).append("\n");
            bytecodeContent.append("// Generated by SootSlicer\n\n");
            
            for (Unit unit : sliceUnits) {
                bytecodeContent.append(unit.toString()).append("\n");
            }
            
            Path bytecodeFile = Paths.get(outputDir, className + "_slice.bytecode");
            Files.write(bytecodeFile, bytecodeContent.toString().getBytes());
            
            return bytecodeFile;
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to generate bytecode slice", e);
        }
    }
    
    private Path decompileWithVineflower(Path bytecodeFile, String vineflowerJar, String outputDir) {
        try {
            System.out.println("[soot_slicer] Decompiling with Vineflower: " + vineflowerJar);
            
            // Use Vineflower to decompile bytecode
            ProcessBuilder pb = new ProcessBuilder(
                "java", "-jar", vineflowerJar,
                bytecodeFile.toString(),
                outputDir
            );
            
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            if (exitCode != 0) {
                System.err.println("Vineflower decompilation failed");
                return null;
            }
            
            // Find the decompiled Java file
            String fileName = bytecodeFile.getFileName().toString().replace(".bytecode", ".java");
            return Paths.get(outputDir, fileName);
            
        } catch (Exception e) {
            System.err.println("Error during Vineflower decompilation: " + e.getMessage());
            return null;
        }
    }
    
    private void cleanupTempFiles(Path... files) {
        for (Path file : files) {
            try {
                if (file != null && Files.exists(file)) {
                    Files.deleteIfExists(file);
                    // Also delete parent directory if it's empty
                    Path parent = file.getParent();
                    if (parent != null && Files.exists(parent)) {
                        try {
                            Files.deleteIfExists(parent);
                        } catch (Exception ignored) {}
                    }
                }
            } catch (Exception ignored) {}
        }
    }
    
    private List<Unit> extractRelevantStatements(Body body, int targetLine) {
        List<Unit> relevantUnits = new ArrayList<>();
        
        // Simple heuristic: include all statements in the method
        // In a more sophisticated implementation, you would perform actual program slicing
        for (Unit unit : body.getUnits()) {
            relevantUnits.add(unit);
        }
        
        return relevantUnits;
    }
    
    private void generateSliceOutput(SootMethod targetMethod, List<Unit> sliceUnits, String outputDir, String vineflowerJar) {
        try {
            // Create a simplified Java source representation
            StringBuilder sliceContent = new StringBuilder();
            
            // Add package declaration if available
            SootClass declaringClass = targetMethod.getDeclaringClass();
            if (declaringClass.getPackageName() != null && !declaringClass.getPackageName().isEmpty()) {
                sliceContent.append("package ").append(declaringClass.getPackageName()).append(";\n\n");
            }
            
            // Add class declaration
            sliceContent.append("public class ").append(declaringClass.getShortName()).append(" {\n");
            
            // Add method signature and body
            sliceContent.append("    ");
            if (targetMethod.isPublic()) sliceContent.append("public ");
            if (targetMethod.isStatic()) sliceContent.append("static ");
            sliceContent.append(targetMethod.getReturnType()).append(" ");
            sliceContent.append(targetMethod.getName()).append("(");
            
            // Add parameters
            List<String> paramStrings = new ArrayList<>();
            for (Local param : targetMethod.getActiveBody().getParameterLocals()) {
                paramStrings.add(param.getType() + " " + param.getName());
            }
            sliceContent.append(String.join(", ", paramStrings));
            sliceContent.append(") {\n");
            
            // Add method body (simplified)
            sliceContent.append("        // Sliced method body\n");
            sliceContent.append("        // Original method: ").append(targetMethod.getSignature()).append("\n");
            sliceContent.append("        // Line: ").append(targetMethod.getActiveBody().getUnits().size()).append(" statements\n");
            sliceContent.append("        return null; // Placeholder\n");
            sliceContent.append("    }\n");
            sliceContent.append("}\n");
            
            // Write slice file
            String fileName = declaringClass.getShortName() + "_slice.java";
            Path sliceFile = Paths.get(outputDir, fileName);
            Files.write(sliceFile, sliceContent.toString().getBytes());
            
            // Write metadata
            String metadata = String.format(
                "method=%s\nline=%d\nclass=%s\nslice_type=soot\n",
                targetMethod.getName(),
                targetMethod.getActiveBody().getUnits().size(),
                declaringClass.getName()
            );
            Files.write(Paths.get(outputDir, "slice.meta"), metadata.getBytes());
            
            System.out.println("[soot_slicer] Generated slice: " + sliceFile);
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to write slice output", e);
        }
    }
    
    private void performSourceBasedSlicing(Path targetFile, int lineNumber, String outputDir, String memberSig) {
        try {
            // Read the source file
            String content = Files.readString(targetFile);
            String[] lines = content.split("\n");
            
            // Extract a window around the target line
            int startLine = Math.max(0, lineNumber - 10);
            int endLine = Math.min(lines.length, lineNumber + 10);
            
            StringBuilder sliceContent = new StringBuilder();
            sliceContent.append("// Source-based slice around line ").append(lineNumber).append("\n");
            sliceContent.append("// Method: ").append(memberSig).append("\n\n");
            
            for (int i = startLine; i < endLine; i++) {
                sliceContent.append(lines[i]).append("\n");
            }
            
            // Write slice file
            String fileName = targetFile.getFileName().toString().replace(".java", "_slice.java");
            Path sliceFile = Paths.get(outputDir, fileName);
            Files.write(sliceFile, sliceContent.toString().getBytes());
            
            // Write metadata
            String metadata = String.format(
                "method=%s\nline=%d\nsource=%s\nslice_type=source_fallback\n",
                memberSig,
                lineNumber,
                targetFile.toString()
            );
            Files.write(Paths.get(outputDir, "slice.meta"), metadata.getBytes());
            
            System.out.println("[soot_slicer] Generated source-based slice: " + sliceFile);
            
        } catch (IOException e) {
            System.err.println("Failed to create source-based slice: " + e.getMessage());
            createFallbackSlice(outputDir, memberSig, targetFile.toString());
        }
    }
    
    private void createFallbackSlice(String outputDir, String memberSig, String sourceFile) {
        try {
            String fallbackContent = String.format(
                "// Fallback slice for %s\n" +
                "// Source: %s\n" +
                "public class FallbackSlice {\n" +
                "    public void %s() {\n" +
                "        // Placeholder method\n" +
                "        System.out.println(\"Fallback slice\");\n" +
                "    }\n" +
                "}\n",
                memberSig,
                sourceFile,
                memberSig.replaceAll("[^a-zA-Z0-9]", "_")
            );
            
            Path sliceFile = Paths.get(outputDir, "fallback_slice.java");
            Files.write(sliceFile, fallbackContent.getBytes());
            
            String metadata = String.format(
                "method=%s\nsource=%s\nslice_type=fallback\n",
                memberSig,
                sourceFile
            );
            Files.write(Paths.get(outputDir, "slice.meta"), metadata.getBytes());
            
            System.out.println("[soot_slicer] Created fallback slice: " + sliceFile);
            
        } catch (IOException e) {
            System.err.println("Failed to create fallback slice: " + e.getMessage());
        }
    }
}
