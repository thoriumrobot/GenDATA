package cfwr;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.VariableDeclarator; // Corrected import

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * A utility program that processes a list of warnings from the Checker Framework
 * generated on a target Java program and outputs the field or method signatures
 * of the nearest enclosing field/method for each warning location.
 *
 * The purpose of the program is to take a list of warnings and then use WALA
 * to generate a slice for each warning in the input list.
 *
 * Usage:
 * ./gradlew run -PappArgs="<projectRoot> <warningsFilePath> <resolverRoot>"
 *
 * Arguments:
 * - projectRoot: Absolute path to the root directory of the target Java project.
 * - warningsFilePath: Absolute path to the file containing the Checker Framework warnings.
 * - resolverRoot: Absolute path to the root directory of this tool (CFWR).
 *
 * The warnings file should contain warnings in the standard format output by the Checker Framework.
 *
 * Example warning format:
 * /path/to/File.java:25:17: compiler.err.proc.messager: [index] Possible out-of-bounds access
 *
 * Example invocation:
 * ./gradlew run -PappArgs="/path/to/project /path/to/warnings.txt /path/to/CFWR"
 */
public class CheckerFrameworkWarningResolver {

    /**
     * A pattern that matches the format of the warnings produced by the Checker Framework.
     * It is intended to match lines like the following:
     * /path/to/File.java:25:17: compiler.err.proc.messager: [index] Possible out-of-bounds access
     */
    private static final Pattern WARNING_PATTERN = Pattern.compile("^(.+\\.java):(\\d+):(\\d+):\\s*(compiler\\.(warn|err)\\.proc\\.messager):\\s*\\[(.+?)\\]\\s*(.*)$");

    static String resolverPath;
    static boolean executeCommandFlag = true; // Flag to control command execution
    static String slicerType = "cf"; // Default slicer type now uses Checker Framework CFG Builder
    static String warningsFileGlobal;
    static int speciminLogCount = 0; // limit debug samples

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: java CheckerFrameworkWarningResolver <projectRoot> <warningsFilePath> <resolverRoot> [slicerType]");
            System.err.println("  slicerType: 'wala' (default) or 'specimin'");
            return;
        }

        String projectRoot = args[0];
        String warningsFilePath = args[1];
        warningsFileGlobal = warningsFilePath;
        resolverPath = args[2];
        
        // Optional fourth argument for slicer type
        if (args.length >= 4) {
            slicerType = args[3].toLowerCase();
            if (!slicerType.equals("wala") && !slicerType.equals("specimin") && !slicerType.equals("cf") && !slicerType.equals("soot")) {
                System.err.println("Error: slicerType must be 'cf' (default), 'wala', 'specimin', or 'soot', got: " + args[3]);
                return;
            }
        }

        try {
            JavaParser parser = new JavaParser();

            List<Warning> warnings = new ArrayList<>();

            try (BufferedReader br = Files.newBufferedReader(Paths.get(warningsFilePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    Matcher matcher = WARNING_PATTERN.matcher(line);
                    if (matcher.matches()) {
                        String fileName = matcher.group(1).trim();
                        int lineNumber = Integer.parseInt(matcher.group(2).trim());
                        int columnNumber = Integer.parseInt(matcher.group(3).trim());
                        String compilerMessageType = matcher.group(4).trim();
                        String checkerName = matcher.group(6).trim();
                        String message = matcher.group(7).trim();

                        Path filePath = Paths.get(fileName);
                        if (!filePath.isAbsolute()) {
                            filePath = Paths.get(projectRoot).resolve(filePath).normalize();
                        }

                        warnings.add(new Warning(filePath, lineNumber, columnNumber, compilerMessageType, checkerName, message));
                    } else {
                        System.err.println("Warning line does not match expected format: " + line);
                    }
                }
            }

            Set<Path> filesToParse = new HashSet<>();
            for (Warning warning : warnings) {
                filesToParse.add(warning.filePath);
            }

            Map<Path, CompilationUnit> compilationUnits = new HashMap<>();
            for (Path filePath : filesToParse) {
                try {
                    ParseResult<CompilationUnit> result = parser.parse(filePath);
                    if (result.isSuccessful() && result.getResult().isPresent()) {
                        compilationUnits.put(filePath, result.getResult().get());
                    } else {
                        System.err.println("Failed to parse file: " + filePath);
                    }
                } catch (IOException e) {
                    System.err.println("Error reading file: " + filePath);
                    e.printStackTrace();
                }
            }

            for (Warning warning : warnings) {
                processWarning(warning, compilationUnits, projectRoot);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void processWarning(Warning warning, Map<Path, CompilationUnit> compilationUnits, String projectRoot) {
        try {
            CompilationUnit compilationUnit = compilationUnits.get(warning.filePath);
            if (compilationUnit == null) {
                System.err.println("No compilation unit found for file: " + warning.filePath);
                return;
            }

            Position warningPosition = new Position(warning.lineNumber, warning.columnNumber);

            Optional<BodyDeclaration<?>> enclosingMember = findEnclosingMember(compilationUnit, warningPosition);

            if (enclosingMember.isPresent()) {
                List<String> command;
                String workingDirectory;
                
                if ("specimin".equals(slicerType)) {
                    command = buildSpeciminCommand(enclosingMember.get(), warning, projectRoot);
                    workingDirectory = Paths.get(resolverPath, "specimin").toString();
                } else if ("wala".equals(slicerType)) {
                    command = buildWalaSourceCommand(enclosingMember.get(), warning, projectRoot);
                    workingDirectory = Paths.get(resolverPath).toString();
                } else if ("soot".equals(slicerType)) {
                    command = buildSootCommand(enclosingMember.get(), warning, projectRoot);
                    workingDirectory = Paths.get(resolverPath).toString();
                } else { // "cf" Checker Framework CFG Builder (default)
                    command = buildCheckerFrameworkCommand(projectRoot);
                    // Run CF slicer in the project root so it can scan files when only warnings/output are given
                    workingDirectory = Paths.get(projectRoot).toString();
                }
                
                if (command != null) {
                    System.out.println("Using " + slicerType.toUpperCase() + " slicer: " + String.join(" ", command));
                    if (executeCommandFlag) {
                        executeCommand(command, workingDirectory);
                    }
                }
            } else {
                System.err.println("No enclosing member found for warning at " + warning.filePath + ":" + warning.lineNumber + ":" + warning.columnNumber);
                // File-level fallback for soot: produce a minimal slice to avoid blank outputs
                if ("soot".equals(slicerType)) {
                    try {
                        List<String> fallback = buildSootFileLevelFallback(warning, projectRoot);
                        if (fallback != null) {
                            System.out.println("[fallback] Using SOOT file-level slice: " + String.join(" ", fallback));
                            if (executeCommandFlag) {
                                executeCommand(fallback, Paths.get(resolverPath).toString());
                            }
                        }
                    } catch (Exception ef) {
                        System.err.println("Fallback failed: " + ef.getMessage());
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error processing warning: " + warning);
            e.printStackTrace();
        }
    }

    private static Optional<BodyDeclaration<?>> findEnclosingMember(CompilationUnit cu, Position position) {
        // Adjusted the list to use raw type to avoid type inference issues
        List<BodyDeclaration> bodyDeclarations = cu.findAll(BodyDeclaration.class);

        BodyDeclaration<?> closestMember = null;
        int smallestRange = Integer.MAX_VALUE;

        for (BodyDeclaration<?> member : bodyDeclarations) {
            if (member.getBegin().isPresent() && member.getEnd().isPresent()) {
                Range range = new Range(member.getBegin().get(), member.getEnd().get());

                if (range.contains(position)) {
                    int rangeSize = range.getLineCount();

                    if (rangeSize < smallestRange) {
                        smallestRange = rangeSize;
                        closestMember = member;
                    }
                }
            }
        }

        return Optional.ofNullable(closestMember);
    }

    private static List<String> buildSpeciminCommand(BodyDeclaration<?> member, Warning warning, String projectRoot) throws IOException {
        String baseSlicesDir = System.getenv().getOrDefault("SLICES_DIR_SOOT",
                System.getenv().getOrDefault("SLICES_DIR", "slices"));
        Path baseDirPath = Paths.get(baseSlicesDir).toAbsolutePath().normalize();
        Files.createDirectories(baseDirPath);

        String root = projectRoot;
        String targetFile = warning.filePath.toString();
        // Specimin expects targetFile paths relative to --root. If absolute and under root, relativize it.
        try {
            Path rootPath = Paths.get(root).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                targetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) { }
        String targetMethodOrField;

        String sliceNameComponent;
        if (member instanceof MethodDeclaration) {
            MethodDeclaration method = (MethodDeclaration) member;
            String qualifiedClassName = getQualifiedClassName(method);
            String methodSignature = getMethodSignature(method);
            targetMethodOrField = qualifiedClassName + "#" + methodSignature;
            sliceNameComponent = qualifiedClassName + "#" + methodSignature;
        } else if (member instanceof ConstructorDeclaration) {
            ConstructorDeclaration constructor = (ConstructorDeclaration) member;
            String qualifiedClassName = getQualifiedClassName(constructor);
            String methodSignature = getConstructorSignature(constructor);
            targetMethodOrField = qualifiedClassName + "#" + methodSignature;
            sliceNameComponent = qualifiedClassName + "#" + methodSignature;
        } else if (member instanceof FieldDeclaration) {
            FieldDeclaration field = (FieldDeclaration) member;
            VariableDeclarator variable = findVariableAtPosition(field, new Position(warning.lineNumber, warning.columnNumber));
            if (variable != null) {
                String qualifiedClassName = getQualifiedClassName(field);
                String fieldName = variable.getNameAsString();
                targetMethodOrField = qualifiedClassName + "#" + fieldName;
                sliceNameComponent = qualifiedClassName + "#" + fieldName;
            } else {
                System.err.println("No variable found at position in field declaration");
                return null;
            }
        } else {
            System.err.println("Unsupported member type: " + member.getClass().getSimpleName());
            return null;
        }

        String relativeTargetFile = targetFile;
        try {
            Path rootPath = Paths.get(root).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                relativeTargetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) { }

        String safeSliceDirName = sanitizeSliceName(relativeTargetFile + "__" + sliceNameComponent);
        Path outputPath = baseDirPath.resolve(safeSliceDirName);
        Files.createDirectories(outputPath);
        String outputDirectory = outputPath.toString();

        List<String> command = new ArrayList<>();
        command.add("./gradlew");
        command.add("run");

        // Collect possible jarPath directories for Specimin context
        // Priority: SPECIMIN_JARPATH (one or many dirs), then CHECKERFRAMEWORK_HOME/checker/dist, then dirs from CHECKERFRAMEWORK_CP
        LinkedHashSet<String> jarPathDirs = new LinkedHashSet<>();

        String speciminJarPath = System.getenv("SPECIMIN_JARPATH");
        if (speciminJarPath != null && !speciminJarPath.isBlank()) {
            for (String p : speciminJarPath.split(Pattern.quote(File.pathSeparator))) {
                if (p != null && !p.isBlank() && Files.isDirectory(Paths.get(p))) {
                    jarPathDirs.add(Paths.get(p).toAbsolutePath().normalize().toString());
                }
            }
        }

        String cfHome = System.getenv("CHECKERFRAMEWORK_HOME");
        if (cfHome != null && !cfHome.isBlank()) {
            Path dist = Paths.get(cfHome, "checker", "dist").toAbsolutePath().normalize();
            if (Files.isDirectory(dist)) jarPathDirs.add(dist.toString());
            Path libs = Paths.get(cfHome, "checker", "build", "libs").toAbsolutePath().normalize();
            if (Files.isDirectory(libs)) jarPathDirs.add(libs.toString());
        }

        String cfCp = System.getenv("CHECKERFRAMEWORK_CP");
        if (cfCp != null && !cfCp.isBlank()) {
            for (String cpEntry : cfCp.split(Pattern.quote(File.pathSeparator))) {
                if (cpEntry == null || cpEntry.isBlank()) continue;
                Path cpPath = Paths.get(cpEntry);
                if (Files.isRegularFile(cpPath) && cpEntry.endsWith(".jar")) {
                    Path parent = cpPath.getParent();
                    if (parent != null && Files.isDirectory(parent)) {
                        jarPathDirs.add(parent.toAbsolutePath().normalize().toString());
                    }
                } else if (Files.isDirectory(cpPath)) {
                    jarPathDirs.add(cpPath.toAbsolutePath().normalize().toString());
                }
            }
        }

        // Build arguments list for Specimin
        List<String> speciminArgs = new ArrayList<>();
        speciminArgs.add("--outputDirectory");
        speciminArgs.add(outputDirectory);
        speciminArgs.add("--root");
        speciminArgs.add(root);
        speciminArgs.add("--targetFile");
        speciminArgs.add(targetFile);
        
        if (member instanceof FieldDeclaration) {
            speciminArgs.add("--targetField");
            speciminArgs.add(targetMethodOrField);
        } else {
            speciminArgs.add("--targetMethod");
            speciminArgs.add(targetMethodOrField);
        }

        // Append any jarPath directories gathered above
        for (String dir : jarPathDirs) {
            speciminArgs.add("--jarPath");
            speciminArgs.add(dir);
        }

        // Join all arguments into a single string for --args, properly escaping
        StringBuilder argsBuilder = new StringBuilder();
        for (int i = 0; i < speciminArgs.size(); i++) {
            if (i > 0) argsBuilder.append(" ");
            String arg = speciminArgs.get(i);
            // Quote arguments that contain spaces or special characters
            if (arg.contains(" ") || arg.contains("(") || arg.contains(")") || arg.contains("#")) {
                argsBuilder.append("\"").append(arg).append("\"");
            } else {
                argsBuilder.append(arg);
            }
        }
        command.add("--args=" + argsBuilder.toString());

        // Debug: print a few example commands and key params
        try {
            if (++speciminLogCount <= 3) {
                String pretty = String.join(" ", command);
                System.out.println("[debug] Specimin example cmd (" + speciminLogCount + "): " + pretty);
                System.out.println("[debug]   root=" + root);
                System.out.println("[debug]   targetFile(rel?)=" + relativeTargetFile + ", raw=" + targetFile);
                System.out.println("[debug]   member=" + targetMethodOrField);
                if (!jarPathDirs.isEmpty()) {
                    System.out.println("[debug]   jarPath dirs=" + String.join(File.pathSeparator, jarPathDirs));
                }
            }
        } catch (Exception ignore) {}

        return command;
    }

    private static List<String> buildCheckerFrameworkCommand(String projectRoot) throws IOException {
        String warningsFile = warningsFileGlobal != null ? warningsFileGlobal : "index1.out";
        String outputDir = Paths.get(System.getenv().getOrDefault("SLICES_DIR", "slices")).toAbsolutePath().normalize().toString();

        List<String> cmd = new ArrayList<String>();
        cmd.add("java");
        cmd.add("-cp");
        // Use Gradle runtime classpath via the fat CF slicer jar if available; fall back to classes
        Path fatJar = Paths.get(resolverPath, "build", "libs", "CFWR-all.jar");
        if (Files.exists(fatJar)) {
            cmd.add(fatJar.toString());
            cmd.add("cfwr.CheckerFrameworkSlicer");
        } else {
            // Fallback to 'classes/java/main' and runtime classpath
            String rtCp = System.getProperty("java.class.path");
            Path classes = Paths.get(resolverPath, "build", "classes", "java", "main");
            cmd.add(rtCp + File.pathSeparator + classes.toString());
            cmd.add("cfwr.CheckerFrameworkSlicer");
        }

        cmd.add(warningsFile);
        cmd.add(outputDir);
        // Let the slicer scan for Java files relative to projectRoot
        cmd.add(Paths.get(projectRoot).toString());

        return cmd;
    }

    private static String sanitizeSliceName(String name) {
        return name.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static String getQualifiedClassName(Node node) {
        // Traverse up to find the enclosing ClassOrInterfaceDeclaration
        Optional<ClassOrInterfaceDeclaration> classDecl = node.findAncestor(ClassOrInterfaceDeclaration.class);
        if (classDecl.isPresent()) {
            String className = classDecl.get().getNameAsString();
            // Get package name
            Optional<CompilationUnit> cu = node.findCompilationUnit();
            String packageName = cu.flatMap(CompilationUnit::getPackageDeclaration)
                    .map(pd -> pd.getNameAsString())
                    .orElse("");
            if (!packageName.isEmpty()) {
                return packageName + "." + className;
            } else {
                return className;
            }
        } else {
            return ""; // Or handle anonymous classes if necessary
        }
    }

    private static String getMethodSignature(MethodDeclaration method) {
        StringBuilder signature = new StringBuilder();
        signature.append(method.getNameAsString());
        signature.append("(");
        List<String> params = new ArrayList<>();
        for (Parameter param : method.getParameters()) {
            // Strip annotations from parameter types for Specimin compatibility
            String typeString = param.getType().asString();
            // Remove annotations like @IndexFor("#1") from the type
            typeString = typeString.replaceAll("@\\w+(\\([^)]*\\))?\\s*", "");
            params.add(typeString.trim());
        }
        signature.append(String.join(",", params));
        signature.append(")");
        return signature.toString();
    }

    private static String getConstructorSignature(ConstructorDeclaration constructor) {
        StringBuilder signature = new StringBuilder();
        signature.append(constructor.getNameAsString());
        signature.append("(");
        List<String> params = new ArrayList<>();
        for (Parameter param : constructor.getParameters()) {
            // Strip annotations from parameter types for Specimin compatibility
            String typeString = param.getType().asString();
            // Remove annotations like @IndexFor("#1") from the type
            typeString = typeString.replaceAll("@\\w+(\\([^)]*\\))?\\s*", "");
            params.add(typeString.trim());
        }
        signature.append(String.join(",", params));
        signature.append(")");
        return signature.toString();
    }

    private static VariableDeclarator findVariableAtPosition(FieldDeclaration field, Position position) {
        for (VariableDeclarator variable : field.getVariables()) {
            if (variable.getBegin().isPresent() && variable.getEnd().isPresent()) {
                Range range = new Range(variable.getBegin().get(), variable.getEnd().get());
                if (range.contains(position)) {
                    return variable;
                }
            }
        }
        return null;
    }

    private static String getTempDir() throws IOException {
        Path tempDirectory = Files.createTempDirectory("cfwr_");
        return tempDirectory.toAbsolutePath().toString();
    }

    private static void executeCommand(List<String> command, String workingDirectory) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(new File(workingDirectory));
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            // Capture output
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                System.err.println("Command exited with code " + exitCode);
            } else {
                System.out.println("Command executed successfully: " + String.join(" ", command));
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error executing command: " + String.join(" ", command));
            e.printStackTrace();
        }
    }

    private static class Warning {
        Path filePath;
        int lineNumber;
        int columnNumber;
        String compilerMessageType; // 'compiler.warn.proc.messager' or 'compiler.err.proc.messager'
        String checkerName;
        String message;

        Warning(Path filePath, int lineNumber, int columnNumber, String compilerMessageType, String checkerName, String message) {
            this.filePath = filePath;
            this.lineNumber = lineNumber;
            this.columnNumber = columnNumber;
            this.compilerMessageType = compilerMessageType;
            this.checkerName = checkerName;
            this.message = message;
        }

        @Override
        public String toString() {
            return filePath + ":" + lineNumber + ":" + columnNumber + ": " + compilerMessageType + ": [" + checkerName + "] " + message;
        }
    }

    private static List<String> buildWalaSourceCommand(
        BodyDeclaration<?> member, Warning warning, String projectRoot) throws IOException {

        String baseSlicesDir = System.getenv().getOrDefault("SLICES_DIR_SOOT",
                System.getenv().getOrDefault("SLICES_DIR", "slices"));
        Path baseDirPath = Paths.get(baseSlicesDir).toAbsolutePath().normalize();
        Files.createDirectories(baseDirPath);

        // Make targetFile relative to projectRoot when possible
        String targetFile = warning.filePath.toString();
        try {
            Path rootPath = Paths.get(projectRoot).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                targetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) {}

        // Compute a descriptive slice dir name
        String sliceNameComponent;
        String targetMemberFlagName;   // "--targetMethod" or "--targetField"
        String targetMemberFlagValue;  // e.g., "com.foo.Bar#baz(int,String)" or "com.foo.Bar#QUX"

        if (member instanceof MethodDeclaration) {
            MethodDeclaration m = (MethodDeclaration) member;
            sliceNameComponent = getQualifiedClassName(m) + "#" + getMethodSignature(m);
            targetMemberFlagName  = "--targetMethod";
            targetMemberFlagValue = sliceNameComponent;
        } else if (member instanceof ConstructorDeclaration) {
            ConstructorDeclaration c = (ConstructorDeclaration) member;
            sliceNameComponent = getQualifiedClassName(c) + "#" + getConstructorSignature(c);
            targetMemberFlagName  = "--targetMethod"; // constructors are methods for our CLI
            targetMemberFlagValue = sliceNameComponent;
        } else if (member instanceof FieldDeclaration) {
            FieldDeclaration f = (FieldDeclaration) member;
            VariableDeclarator v = findVariableAtPosition(f, new Position(warning.lineNumber, warning.columnNumber));
            if (v == null) {
                System.err.println("No variable found at position in field declaration");
                return null;
            }
            sliceNameComponent = getQualifiedClassName(f) + "#" + v.getNameAsString();
            targetMemberFlagName  = "--targetField";
            targetMemberFlagValue = sliceNameComponent;
        } else {
            System.err.println("Unsupported member type: " + member.getClass().getSimpleName());
            return null;
        }

        String safeSliceDirName = sanitizeSliceName(targetFile + "__" + sliceNameComponent);
        Path outputPath = baseDirPath.resolve(safeSliceDirName);
        Files.createDirectories(outputPath);
        String outputDirectory = outputPath.toString();

        // Heuristic source roots (adjust if you know them)
        List<String> sourceRoots = new ArrayList<>();
        Path root = Paths.get(projectRoot);
        for (String p : new String[]{"src/main/java", "src/test/java", "src"}) {
            Path candidate = root.resolve(p);
            if (Files.isDirectory(candidate)) sourceRoots.add(candidate.toString());
        }
        if (sourceRoots.isEmpty()) sourceRoots.add(root.toString());
        String joinedRoots = String.join(File.pathSeparator, sourceRoots);

        // Path to the fat jar built by `./gradlew shadowJar`
        // (shadow config sets name to build/libs/wala-slicer-all.jar)
        Path fatJar = Paths.get(resolverPath, "build", "libs", "wala-slicer-all.jar");

        List<String> cmd = new ArrayList<>();
        cmd.add("java");
        cmd.add("-jar");
        cmd.add(fatJar.toString());
        cmd.add("--sourceRoots"); cmd.add(joinedRoots);
        cmd.add("--projectRoot"); cmd.add(projectRoot);
        cmd.add("--targetFile");  cmd.add(targetFile);               // use RELATIVE path
        cmd.add("--line");        cmd.add(Integer.toString(warning.lineNumber));
        cmd.add("--output");      cmd.add(outputDirectory);
        cmd.add(targetMemberFlagName); cmd.add(targetMemberFlagValue);

        return cmd;
    }

    private static List<String> buildSootCommand(
        BodyDeclaration<?> member, Warning warning, String projectRoot) throws IOException {

        String baseSlicesDir = System.getenv().getOrDefault("SLICES_DIR", "slices");
        Path baseDirPath = Paths.get(baseSlicesDir).toAbsolutePath().normalize();
        Files.createDirectories(baseDirPath);

        String targetFile = warning.filePath.toString();
        try {
            Path rootPath = Paths.get(projectRoot).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                targetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) {}

        String sliceNameComponent;
        if (member instanceof MethodDeclaration) {
            MethodDeclaration m = (MethodDeclaration) member;
            sliceNameComponent = getQualifiedClassName(m) + "#" + getMethodSignature(m);
        } else if (member instanceof ConstructorDeclaration) {
            ConstructorDeclaration c = (ConstructorDeclaration) member;
            sliceNameComponent = getQualifiedClassName(c) + "#" + getConstructorSignature(c);
        } else if (member instanceof FieldDeclaration) {
            FieldDeclaration f = (FieldDeclaration) member;
            VariableDeclarator v = findVariableAtPosition(f, new Position(warning.lineNumber, warning.columnNumber));
            if (v == null) {
                System.err.println("No variable found at position in field declaration");
                return null;
            }
            sliceNameComponent = getQualifiedClassName(f) + "#" + v.getNameAsString();
        } else {
            System.err.println("Unsupported member type: " + member.getClass().getSimpleName());
            return null;
        }

        String safeSliceDirName = sanitizeSliceName(targetFile + "__" + sliceNameComponent);
        Path outputPath = baseDirPath.resolve(safeSliceDirName);
        Files.createDirectories(outputPath);
        String outputDirectory = outputPath.toString();

        // Two ways to run: external CLI or java -jar
        String sootCli = System.getenv("SOOT_SLICE_CLI"); // e.g., /path/to/soot-slicer.sh
        String sootJar = System.getenv("SOOT_JAR");       // e.g., /path/to/soot-slicer-all.jar
        String vineflowerJar = System.getenv("VINEFLOWER_JAR"); // optional

        List<String> cmd = new ArrayList<>();
        if (sootCli != null && !sootCli.isBlank()) {
            cmd.add(sootCli);
        } else if (sootJar != null && !sootJar.isBlank()) {
            cmd.add("java");
            cmd.add("-jar");
            cmd.add(sootJar);
        } else {
            System.err.println("SOOT slicer not configured. Set SOOT_SLICE_CLI or SOOT_JAR.");
            return null;
        }

        // Generic arguments we expect a Soot-based slicer to take (placeholders):
        // --projectRoot, --targetFile, --line, --output, and optionally --decompiler <vineflower.jar>
        cmd.add("--projectRoot"); cmd.add(projectRoot);
        cmd.add("--targetFile");  cmd.add(targetFile);
        cmd.add("--line");        cmd.add(Integer.toString(warning.lineNumber));
        cmd.add("--output");      cmd.add(outputDirectory);
        cmd.add("--member");      cmd.add(sliceNameComponent);
        if (vineflowerJar != null && !vineflowerJar.isBlank()) {
            cmd.add("--decompiler");
            cmd.add(vineflowerJar);
        }
        
        // Add prediction mode for annotation placement
        String predictionMode = System.getenv("SOOT_PREDICTION_MODE");
        if (predictionMode != null && !predictionMode.isBlank()) {
            cmd.add("--prediction-mode");
        }

        return cmd;
    }

    private static List<String> buildSootFileLevelFallback(Warning warning, String projectRoot) throws IOException {
        String baseSlicesDir = System.getenv().getOrDefault("SLICES_DIR", "slices");
        Path baseDirPath = Paths.get(baseSlicesDir).toAbsolutePath().normalize();
        Files.createDirectories(baseDirPath);

        String targetFile = warning.filePath.toString();
        try {
            Path rootPath = Paths.get(projectRoot).toAbsolutePath().normalize();
            Path targetPath = Paths.get(targetFile).toAbsolutePath().normalize();
            if (targetPath.startsWith(rootPath)) {
                targetFile = rootPath.relativize(targetPath).toString();
            }
        } catch (Exception ignored) {}

        String safeSliceDirName = sanitizeSliceName(targetFile + "__file_level");
        Path outputPath = baseDirPath.resolve(safeSliceDirName);
        Files.createDirectories(outputPath);

        String sootCli = System.getenv("SOOT_SLICE_CLI");
        String sootJar = System.getenv("SOOT_JAR");
        String vineflowerJar = System.getenv("VINEFLOWER_JAR");

        List<String> cmd = new ArrayList<>();
        if (sootCli != null && !sootCli.isBlank()) {
            cmd.add(sootCli);
        } else if (sootJar != null && !sootJar.isBlank()) {
            cmd.add("java");
            cmd.add("-jar");
            cmd.add(sootJar);
        } else {
            System.err.println("SOOT slicer not configured. Set SOOT_SLICE_CLI or SOOT_JAR.");
            return null;
        }

        cmd.add("--projectRoot"); cmd.add(projectRoot);
        cmd.add("--targetFile");  cmd.add(targetFile);
        cmd.add("--line");        cmd.add(Integer.toString(warning.lineNumber));
        cmd.add("--output");      cmd.add(outputPath.toString());
        cmd.add("--member");      cmd.add("file_level");
        if (vineflowerJar != null && !vineflowerJar.isBlank()) {
            cmd.add("--decompiler");
            cmd.add(vineflowerJar);
        }

        return cmd;
    }

}
