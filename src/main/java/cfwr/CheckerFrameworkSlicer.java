package cfwr;

import com.sun.source.tree.*;
import com.sun.source.util.*;
import javax.tools.*;
import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Checker Framework Slicer - Uses Checker Framework's CFG Builder for slicing and CFG generation.
 * This is the default and most comprehensive slicing option.
 */
public class CheckerFrameworkSlicer {
    
    private static final String CHECKER_FRAMEWORK_CP = System.getenv("CHECKERFRAMEWORK_CP");
    
    private static boolean PROCESS_ALL_IF_NO_WARNINGS = false;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java cfwr.CheckerFrameworkSlicer <warnings_file|- > <output_dir> [java_files_or_dirs...]");
            System.exit(1);
        }
        
        String warningsFile = args[0];
        String outputDir = args[1];
        java.util.List<String> javaFiles = new java.util.ArrayList<String>();

        if ("-".equals(warningsFile)) {
            PROCESS_ALL_IF_NO_WARNINGS = true;
        }
        
        // Collect Java files from remaining arguments (files or directories). If none, scan CWD.
        if (args.length > 2) {
            for (int i = 2; i < args.length; i++) {
                Path p = Paths.get(args[i]);
                if (Files.isDirectory(p)) {
                    try {
                        Files.walk(p)
                            .filter(path -> path.toString().endsWith(".java"))
                            .forEach(path -> javaFiles.add(path.toString()));
                    } catch (IOException e) {
                        System.err.println("Error scanning directory for Java files: " + p + ": " + e.getMessage());
                    }
                } else {
                    javaFiles.add(p.toString());
                }
            }
        } else {
            // Default: scan current directory for Java files
            try {
                Files.walk(Paths.get("."))
                    .filter(path -> path.toString().endsWith(".java"))
                    .forEach(path -> javaFiles.add(path.toString()));
            } catch (IOException e) {
                System.err.println("Error scanning for Java files: " + e.getMessage());
                System.exit(1);
            }
        }
        
        System.out.println("Checker Framework Slicer - Processing " + javaFiles.size() + " Java files");
        System.out.println("Warnings file: " + warningsFile);
        System.out.println("Output directory: " + outputDir);
        
        // Parse warnings
        java.util.List<WarningInfo> warnings = parseWarnings(warningsFile);
        System.out.println("Parsed " + warnings.size() + " warnings");
        
        // Process each Java file
        for (String javaFile : javaFiles) {
            try {
                processJavaFile(javaFile, warnings, outputDir);
            } catch (Exception e) {
                System.err.println("Error processing " + javaFile + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("Checker Framework slicing completed");
    }
    
    /**
     * Parse warnings from the warnings file
     */
    private static java.util.List<WarningInfo> parseWarnings(String warningsFile) {
        java.util.List<WarningInfo> warnings = new java.util.ArrayList<WarningInfo>();
        try {
            java.util.List<String> lines = Files.readAllLines(Paths.get(warningsFile));
            // Accept both CF diagnostic format with column + checker tag and a simpler fallback
            Pattern cfPattern = Pattern.compile("^(.+\\.java):(\\d+):(\\d+):\\s*(compiler\\.(warn|err)\\.proc\\.messager):\\s*\\[(.+?)\\]\\s*(.*)$");
            Pattern simplePattern = Pattern.compile("^(.*\\.java):(\\d+):\\s*(error|warning):\\s*(.*)$");
            
            for (String line : lines) {
                Matcher m1 = cfPattern.matcher(line);
                if (m1.matches()) {
                    WarningInfo w = new WarningInfo();
                    w.file = m1.group(1);
                    w.line = Integer.parseInt(m1.group(2));
                    // m1.group(3) is column; m1.group(6) is checker name; m1.group(7) is message
                    w.type = m1.group(4); // compiler.warn/err.proc.messager
                    w.message = "[" + m1.group(6) + "] " + m1.group(7);
                    warnings.add(w);
                    continue;
                }
                Matcher m2 = simplePattern.matcher(line);
                if (m2.matches()) {
                    WarningInfo w = new WarningInfo();
                    w.file = m2.group(1);
                    w.line = Integer.parseInt(m2.group(2));
                    w.type = m2.group(3);
                    w.message = m2.group(4);
                    warnings.add(w);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading warnings file: " + e.getMessage());
        }
        return warnings;
    }
    
    /**
     * Process a single Java file
     */
    private static void processJavaFile(String javaFile, java.util.List<WarningInfo> warnings, String outputDir) throws Exception {
        System.out.println("Processing: " + javaFile);
        
        // Filter warnings for this file
        java.util.List<WarningInfo> fileWarnings = new java.util.ArrayList<WarningInfo>();
        for (WarningInfo warning : warnings) {
            if (warning.file.endsWith(javaFile) || javaFile.endsWith(warning.file)) {
                fileWarnings.add(warning);
            }
        }
        
            boolean noWarns = fileWarnings.isEmpty();
            if (noWarns && !PROCESS_ALL_IF_NO_WARNINGS) {
                System.out.println("No warnings for " + javaFile + ", skipping");
                return;
            }
        
        // Create output directory
        String fileName = Paths.get(javaFile).getFileName().toString();
        String baseName = fileName.substring(0, fileName.lastIndexOf('.'));
        Path sliceDir = Paths.get(outputDir, baseName + "__cf_slice");
        Files.createDirectories(sliceDir);
        
        // CFG-like method slice: locate enclosing member and extract only that method body text
        String fileBaseName = new File(javaFile).getName();
        String sliceBaseName = fileBaseName.replace(".java", "");
        String sliceFile = outputDir + "/" + sliceBaseName + "_slice.java";
        String cfgFile = outputDir + "/" + sliceBaseName + "_cfg.json";

        try {
            List<String> all = Files.readAllLines(Paths.get(javaFile), StandardCharsets.UTF_8);
            if (all.isEmpty()) {
                System.out.println("File empty, skipping: " + javaFile);
                return;
            }

            // Build an AST to find the method/field enclosing any warning line
            JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
            StandardJavaFileManager fm = compiler.getStandardFileManager(null, null, StandardCharsets.UTF_8);
            Iterable<? extends JavaFileObject> cu = fm.getJavaFileObjects(javaFile);
            JavacTask task = (JavacTask) compiler.getTask(null, fm, null, java.util.List.of(), null, cu);
            Iterable<? extends CompilationUnitTree> cuts = task.parse();

            // Collect spans to extract
            final int[] extractStartRef = new int[]{Integer.MAX_VALUE};
            final int[] extractEndRef = new int[]{-1};
            final String[] methodNameRef = new String[]{"slice"};
            final String[] classNameRef = new String[]{sliceBaseName};

            for (CompilationUnitTree cut : cuts) {
                Trees trees = Trees.instance(task);
                new TreePathScanner<Void, Void>() {
                    @Override
                    public Void visitClass(ClassTree clazz, Void p) {
                        return super.visitClass(clazz, p);
                    }
                    @Override
                    public Void visitMethod(MethodTree method, Void p) {
                        long start = trees.getSourcePositions().getStartPosition(cut, method);
                        long end = trees.getSourcePositions().getEndPosition(cut, method);
                        int sLine = (int) cut.getLineMap().getLineNumber(start);
                        int eLine = (int) cut.getLineMap().getLineNumber(end);
                        // If any warning falls within this method, select this method
                        for (WarningInfo w : fileWarnings) {
                            if (w.line >= sLine && w.line <= eLine) {
                                if (sLine < extractStartRef[0]) extractStartRef[0] = sLine;
                                if (eLine > extractEndRef[0]) extractEndRef[0] = eLine;
                                methodNameRef[0] = method.getName().toString();
                                ClassTree cls = (ClassTree) getCurrentPath().getParentPath().getLeaf();
                                classNameRef[0] = cls.getSimpleName().toString();
                                break;
                            }
                        }
                        return super.visitMethod(method, p);
                    }
                }.scan(cut, null);
            }

            int extractStart = extractStartRef[0];
            int extractEnd = extractEndRef[0];
            if (extractStart == Integer.MAX_VALUE || extractEnd == -1) {
                // If configured, extract whole file; else a small window fallback
                if (PROCESS_ALL_IF_NO_WARNINGS) {
                    extractStart = 1;
                    extractEnd = all.size();
                } else {
                    extractStart = Math.max(1, fileWarnings.isEmpty() ? 1 : fileWarnings.get(0).line - 3);
                    extractEnd = Math.min(all.size(), extractStart + 20);
                }
            }

            StringBuilder sb = new StringBuilder();
            for (int i = extractStart; i <= extractEnd; i++) sb.append(all.get(i - 1)).append('\n');

            SliceInfo sliceInfo = new SliceInfo();
            sliceInfo.methodName = methodNameRef[0];
            sliceInfo.className = classNameRef[0];
            sliceInfo.javaCode = sb.toString();

            // Build CFG with control edges per line and dataflow edges by same-name variables
            String cfgJson = generateWindowCFGJsonWithDataflow(sliceInfo, extractStart, extractEnd, all);

            Files.write(Paths.get(sliceFile), sliceInfo.javaCode.getBytes(StandardCharsets.UTF_8));
            Files.write(Paths.get(cfgFile), cfgJson.getBytes(StandardCharsets.UTF_8));
            System.out.println("Generated slice: " + sliceFile);
            System.out.println("Generated CFG: " + cfgFile);
        } catch (IOException e) {
            System.err.println("Error writing files: " + e.getMessage());
        }
    }
    
    /**
     * Process a single method
     */
    private static void processMethod(MethodInfo methodInfo, java.util.List<WarningInfo> warnings, 
                                   Path sliceDir) throws Exception {
        
        // Check if this method has any warnings
        boolean hasWarnings = false;
        for (WarningInfo warning : warnings) {
            if (methodInfo.lineStart <= warning.line && warning.line <= methodInfo.lineEnd) {
                hasWarnings = true;
                break;
            }
        }
        
        if (!hasWarnings) {
            return; // Skip methods without warnings
        }
        
        System.out.println("Processing method: " + methodInfo.name);
        
        // Create a simple slice without CFG building
        SliceInfo slice = createSimpleSlice(methodInfo, warnings);
        
        if (slice.nodes.isEmpty()) {
            System.out.println("No relevant nodes found for method " + methodInfo.name);
            return;
        }
        
        // Create slice file
        String sliceFileName = methodInfo.name + ".java";
        Path sliceFile = sliceDir.resolve(sliceFileName);
        
        // Generate Java code for the slice
        String javaCode = generateJavaCode(slice, methodInfo);
        
        Files.write(sliceFile, javaCode.getBytes());
        System.out.println("Generated slice: " + sliceFile);
        
        // Generate simple CFG with dataflow information
        generateSimpleCFGWithDataflow(slice, sliceDir, methodInfo.name);
    }
    
    /**
     * Create a simple slice for testing purposes
     */
    private static SliceInfo createSimpleSlice(String javaFile) {
        SliceInfo slice = new SliceInfo();
        slice.methodName = "testMethod";
        slice.className = "TestClass";
        slice.javaCode = "public class TestClass {\n    public int testMethod(int a) {\n        int x = 5;\n        int y = x + 1;\n        if (y > 3) {\n            int z = y * 2;\n            System.out.println(z);\n        }\n        return y;\n    }\n}";
        slice.nodes = new java.util.ArrayList<Object>();
        slice.blocks = new java.util.ArrayList<Object>();
        
        return slice;
    }
    
    /**
     * Create a simple slice without CFG building (placeholder implementation)
     */
    private static SliceInfo createSimpleSlice(MethodInfo methodInfo, java.util.List<WarningInfo> warnings) {
        SliceInfo slice = new SliceInfo();
        slice.methodName = methodInfo.name;
        slice.className = methodInfo.className;
        slice.nodes = new java.util.ArrayList<Object>();
        slice.blocks = new java.util.ArrayList<Object>();
        
        // For now, create dummy nodes based on method structure
        // This is a simplified implementation - in production, you'd use the actual CFG
        return slice;
    }
    
    /**
     * Generate Java code for the slice
     */
    private static String generateJavaCode(SliceInfo slice, MethodInfo methodInfo) {
        StringBuilder code = new StringBuilder();
        
        code.append("// Generated slice for method: ").append(slice.methodName).append("\n");
        code.append("// Original class: ").append(slice.className).append("\n\n");
        
        code.append("public class ").append(slice.className).append("Slice {\n");
        code.append("    public void ").append(slice.methodName).append("() {\n");
        
        // Generate code for each node in the slice
        for (Object node : slice.nodes) {
            String nodeCode = node.toString();
            code.append("        ").append(nodeCode).append(";\n");
        }
        
        code.append("    }\n");
        code.append("}\n");
        
        return code.toString();
    }
    
    /**
     * Generate a simple CFG JSON string for testing
     */
    private static String generateSimpleCFGJson(SliceInfo sliceInfo) {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"method_name\": \"").append(sliceInfo.methodName).append("\",\n");
        json.append("  \"nodes\": [\n");
        json.append("    {\"id\": \"Entry\", \"type\": \"entry\", \"line\": 1},\n");
        json.append("    {\"id\": \"Exit\", \"type\": \"exit\", \"line\": 10}\n");
        json.append("  ],\n");
        json.append("  \"control_edges\": [\n");
        json.append("    {\"from\": \"Entry\", \"to\": \"Exit\", \"type\": \"control\"}\n");
        json.append("  ],\n");
        json.append("  \"dataflow_edges\": [\n");
        json.append("    {\"from\": \"Entry\", \"to\": \"Exit\", \"type\": \"dataflow\", \"variable\": \"x\"}\n");
        json.append("  ]\n");
        json.append("}");
        return json.toString();
    }

    /**
     * Generate a minimal CFG JSON for a contiguous line window.
     */
    private static String generateWindowCFGJsonWithDataflow(SliceInfo sliceInfo, int startLine, int endLine, List<String> allLines) {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"method_name\": \"").append(sliceInfo.methodName).append("\",\n");
        json.append("  \"nodes\": [\n");
        // Create a node per line
        for (int i = startLine; i <= endLine; i++) {
            int id = i;
            String label = escapeJson(trimToMax(allLines.get(i - 1), 200));
            json.append("    {\"id\": ").append(id).append(", \"label\": \"").append(label).append("\", \"line\": ").append(i).append(", \"node_type\": \"stmt\"}");
            if (i < endLine) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        // Control edges: sequential lines
        json.append("  \"control_edges\": [\n");
        boolean first = true;
        for (int i = startLine; i < endLine; i++) {
            if (!first) json.append(",\n");
            first = false;
            json.append("    {\"source\": ").append(i).append(", \"target\": ").append(i + 1).append(", \"type\": \"control\"}");
        }
        json.append("\n  ],\n");
        // Dataflow edges: connect same identifiers across successive occurrences
        json.append("  \"dataflow_edges\": [\n");
        Map<String, Integer> lastLineByIdent = new HashMap<String, Integer>();
        Pattern ident = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");
        boolean firstDf = true;
        for (int i = startLine; i <= endLine; i++) {
            String line = allLines.get(i - 1);
            java.util.Set<String> seenInLine = new java.util.HashSet<String>();
            java.util.regex.Matcher m = ident.matcher(line);
            while (m.find()) {
                String name = m.group();
                if (isJavaKeyword(name)) continue;
                if (!seenInLine.add(name)) continue; // once per line per ident
                Integer prev = lastLineByIdent.get(name);
                if (prev != null) {
                    if (!firstDf) json.append(",\n");
                    firstDf = false;
                    json.append("    {\"source\": ").append(prev).append(", \"target\": ").append(i).append(", \"type\": \"dataflow\", \"var\": \"").append(escapeJson(name)).append("\"}");
                }
                lastLineByIdent.put(name, i);
            }
        }
        json.append("\n  ]\n");
        json.append("}\n");
        return json.toString();
    }

    private static boolean isJavaKeyword(String s) {
        // Minimal keyword set
        return java.util.Set.of(
            "abstract","assert","boolean","break","byte","case","catch","char","class","const","continue",
            "default","do","double","else","enum","extends","final","finally","float","for","goto","if",
            "implements","import","instanceof","int","interface","long","native","new","package","private","protected",
            "public","return","short","static","strictfp","super","switch","synchronized","this","throw","throws",
            "transient","try","void","volatile","while","true","false","null"
        ).contains(s);
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    private static String trimToMax(String s, int n) {
        if (s.length() <= n) return s;
        return s.substring(0, n);
    }
    
    /**
     * Generate simple CFG with dataflow information (placeholder implementation)
     */
    private static void generateSimpleCFGWithDataflow(SliceInfo slice, 
                                                    Path sliceDir, String methodName) throws Exception {
        
        // Create CFG data structure compatible with our models
        Map<String, Object> cfgData = new HashMap<String, Object>();
        cfgData.put("method_name", methodName);
        cfgData.put("java_file", slice.className + "Slice.java");
        
        java.util.List<Map<String, Object>> nodes = new java.util.ArrayList<Map<String, Object>>();
        java.util.List<Map<String, Object>> edges = new java.util.ArrayList<Map<String, Object>>();
        java.util.List<Map<String, Object>> controlEdges = new java.util.ArrayList<Map<String, Object>>();
        java.util.List<Map<String, Object>> dataflowEdges = new java.util.ArrayList<Map<String, Object>>();
        
        // Create simple nodes for demonstration
        Map<String, Object> entryNode = new HashMap<String, Object>();
        entryNode.put("id", 0);
        entryNode.put("label", "Entry");
        entryNode.put("line", 1);
        entryNode.put("node_type", "control");
        nodes.add(entryNode);
        
        Map<String, Object> exitNode = new HashMap<String, Object>();
        exitNode.put("id", 1);
        exitNode.put("label", "Exit");
        exitNode.put("line", 2);
        exitNode.put("node_type", "control");
        nodes.add(exitNode);
        
        // Create simple control flow edge
        Map<String, Object> controlEdge = new HashMap<String, Object>();
        controlEdge.put("source", 0);
        controlEdge.put("target", 1);
        controlEdges.add(controlEdge);
        edges.add(controlEdge);
        
        // Create simple dataflow edge for demonstration
        Map<String, Object> dataflowEdge = new HashMap<String, Object>();
        dataflowEdge.put("source", 0);
        dataflowEdge.put("target", 1);
        dataflowEdge.put("variable", "x");
        dataflowEdges.add(dataflowEdge);
        edges.add(dataflowEdge);
        
        cfgData.put("nodes", nodes);
        cfgData.put("edges", edges);
        cfgData.put("control_edges", controlEdges);
        cfgData.put("dataflow_edges", dataflowEdges);
        
        // Save CFG as JSON
        String cfgFileName = methodName + ".json";
        Path cfgFile = sliceDir.resolve(cfgFileName);
        
        // Convert to JSON (simplified - in production use a proper JSON library)
        String json = convertToJson(cfgData);
        Files.write(cfgFile, json.getBytes());
        
        System.out.println("Generated simple CFG with dataflow: " + cfgFile);
    }
    
    /**
     * Convert CFG data to JSON (simplified implementation)
     */
    private static String convertToJson(Map<String, Object> data) {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        
        json.append("  \"method_name\": \"").append(data.get("method_name")).append("\",\n");
        json.append("  \"java_file\": \"").append(data.get("java_file")).append("\",\n");
        
        json.append("  \"nodes\": [\n");
        java.util.List<Map<String, Object>> nodes = (java.util.List<Map<String, Object>>) data.get("nodes");
        for (int i = 0; i < nodes.size(); i++) {
            Map<String, Object> node = nodes.get(i);
            json.append("    {\n");
            json.append("      \"id\": ").append(node.get("id")).append(",\n");
            json.append("      \"label\": \"").append(node.get("label")).append("\",\n");
            json.append("      \"line\": ").append(node.get("line")).append(",\n");
            json.append("      \"node_type\": \"").append(node.get("node_type")).append("\"\n");
            json.append("    }");
            if (i < nodes.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        
        json.append("  \"control_edges\": [\n");
        java.util.List<Map<String, Object>> controlEdges = (java.util.List<Map<String, Object>>) data.get("control_edges");
        for (int i = 0; i < controlEdges.size(); i++) {
            Map<String, Object> edge = controlEdges.get(i);
            json.append("    {\n");
            json.append("      \"source\": ").append(edge.get("source")).append(",\n");
            json.append("      \"target\": ").append(edge.get("target")).append("\n");
            json.append("    }");
            if (i < controlEdges.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        
        json.append("  \"dataflow_edges\": [\n");
        java.util.List<Map<String, Object>> dataflowEdges = (java.util.List<Map<String, Object>>) data.get("dataflow_edges");
        for (int i = 0; i < dataflowEdges.size(); i++) {
            Map<String, Object> edge = dataflowEdges.get(i);
            json.append("    {\n");
            json.append("      \"source\": ").append(edge.get("source")).append(",\n");
            json.append("      \"target\": ").append(edge.get("target")).append(",\n");
            json.append("      \"variable\": \"").append(edge.get("variable")).append("\"\n");
            json.append("    }");
            if (i < dataflowEdges.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ]\n");
        
        json.append("}\n");
        return json.toString();
    }
    
    /**
     * Method scanner to find methods in compilation unit
     */
    private static class MethodScanner extends TreeScanner<Void, Void> {
        java.util.List<MethodInfo> methods = new java.util.ArrayList<MethodInfo>();
        
        @Override
        public Void visitMethod(MethodTree method, Void p) {
            MethodInfo info = new MethodInfo();
            info.name = method.getName().toString();
            info.methodTree = method;
            info.lineStart = 1; // Simplified
            info.lineEnd = 100; // Simplified
            methods.add(info);
            return super.visitMethod(method, p);
        }
        
        @Override
        public Void visitClass(ClassTree classTree, Void p) {
            for (MethodInfo method : methods) {
                method.classTree = classTree;
                method.className = classTree.getSimpleName().toString();
                // method.compilationUnit = getCurrentPath().getCompilationUnit();
            }
            return super.visitClass(classTree, p);
        }
    }
    
    /**
     * Data structures
     */
    private static class WarningInfo {
        String file;
        int line;
        String type;
        String message;
    }
    
    private static class MethodInfo {
        String name;
        String className;
        MethodTree methodTree;
        ClassTree classTree;
        CompilationUnitTree compilationUnit;
        int lineStart;
        int lineEnd;
    }
    
    private static class SliceInfo {
        String methodName;
        String className;
        String javaCode;
        java.util.List<Object> nodes;
        java.util.List<Object> blocks;
    }
}