package cfwr;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.Collections;

import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.classLoader.IBytecodeMethod;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.SourceDirectoryTreeModule;

import com.ibm.wala.classLoader.Language;
import com.ibm.wala.ipa.callgraph.*;
import com.ibm.wala.ipa.callgraph.AnalysisCacheImpl;
import com.ibm.wala.ipa.callgraph.impl.AllApplicationEntrypoints;
import com.ibm.wala.ipa.callgraph.impl.Util;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.cha.ClassHierarchyFactory;
import com.ibm.wala.ipa.cha.IClassHierarchy;

import com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope;

import com.ibm.wala.ipa.slicer.NormalStatement;
import com.ibm.wala.ipa.slicer.Slicer;
import com.ibm.wala.ipa.slicer.Statement;

import com.ibm.wala.shrike.shrikeCT.InvalidClassFileException;

import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.MethodReference;

import com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope;

/**
 * WalaSliceCLI (source-mode):
 *   Build a call graph from Java source roots, pick a seed instruction near a file:line inside
 *   the target method/field, take a backward thin slice, and write a textual slice.
 *
 * Required args:
 *   --sourceRoots "<dir1><pathsep><dir2>..."    (path-separator delimited, e.g. ':' on Linux, ';' on Windows)
 *   --projectRoot "<abs path to project root>"  (used to resolve files for output/materialization)
 *   --targetFile  "<relative or absolute path to .java file>"
 *   --line        "<1-based line>"
 *   --output      "<abs output directory>"
 *   exactly one of:
 *     --targetMethod "pkg.Cls#name(T1,T2,...)"
 *     --targetField  "pkg.Cls#FIELD"
 *
 * Example:
 *   java -jar build/libs/wala-slicer-all.jar \
 *     --sourceRoots "/proj/src/main/java:/proj/src/test/java" \
 *     --projectRoot "/proj" \
 *     --targetFile  "src/main/java/com/example/Foo.java" \
 *     --line 123 \
 *     --targetMethod "com.example.Foo#bar(int,String)" \
 *     --output "/tmp/slices/Foo__bar"
 */
public class WalaSliceCLI {

  public static void main(String[] argv) throws Exception {
    // Set WALA properties BEFORE any WALA classes are loaded
    System.setProperty("wala.source.loader", "com.ibm.wala.cast.java.translator.jdt.ecj.ECJSourceLoaderImpl");
    System.setProperty("wala.source.loader.impl", "com.ibm.wala.cast.java.translator.jdt.ecj.ECJSourceLoaderImpl");
    System.setProperty("com.ibm.wala.cast.java.source.loader", "com.ibm.wala.cast.java.translator.jdt.ecj.ECJSourceLoaderImpl");
    System.setProperty("wala.source.loader.class", "com.ibm.wala.cast.java.translator.jdt.ecj.ECJSourceLoaderImpl");
    System.setProperty("wala.source.loader.impl.class", "com.ibm.wala.cast.java.translator.jdt.ecj.ECJSourceLoaderImpl");
    System.setProperty("wala.classloader.factory", "com.ibm.wala.classLoader.ClassLoaderFactoryImpl");
    
    // Disable problematic loaders
    System.setProperty("wala.disable", "polyglot");
    System.setProperty("wala.source.loader.polyglot", "false");
    
    System.out.println("[DEBUG] Set WALA properties early");
    
    Map<String, String> a = parseArgs(argv);

    // --- required args
    Path projectRoot = Paths.get(req(a, "projectRoot")).toAbsolutePath().normalize();
    String sourceRootsStr = req(a, "sourceRoots");
    String targetFileArg  = req(a, "targetFile");
    Path targetFile = absolutizeUnderRoot(projectRoot, targetFileArg).normalize();
    int line = Integer.parseInt(req(a, "line"));
    Path outDir = Paths.get(req(a, "output")).toAbsolutePath().normalize();
    Files.createDirectories(outDir);

    String targetMethod = a.get("targetMethod");
    String targetField  = a.get("targetField");
    if ((targetMethod == null) == (targetField == null)) {
      die("Specify exactly ONE of --targetMethod or --targetField");
    }

    System.out.println("[DEBUG] Starting WALA slicer with improved source analysis");
    System.out.println("[DEBUG] Source roots: " + sourceRootsStr);
    System.out.println("[DEBUG] Project root: " + projectRoot);
    System.out.println("[DEBUG] Target file: " + targetFile);
    System.out.println("[DEBUG] Target method: " + targetMethod);
    System.out.println("[DEBUG] Output directory: " + outDir);
    
    try {
      // --- 1) Create proper Java source analysis scope
      System.out.println("[DEBUG] Creating Java source analysis scope...");
      com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope scope = 
        new com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope();
      
      // --- 2) Add Java runtime to scope
      System.out.println("[DEBUG] Adding Java runtime to scope...");
      String javaHome = System.getProperty("java.home");
      Path rtJarPath = Paths.get(javaHome, "lib", "rt.jar");
      Path modulesPath = Paths.get(javaHome, "lib", "modules");
      
      if (Files.exists(rtJarPath)) {
        System.out.println("[DEBUG] Adding rt.jar: " + rtJarPath);
        scope.addToScope(ClassLoaderReference.Primordial, rtJarPath.toFile());
      } else if (Files.exists(modulesPath)) {
        System.out.println("[DEBUG] Adding Java modules: " + modulesPath);
        // For Java 9+, try to add modules
        try {
          scope.addToScope(ClassLoaderReference.Primordial, modulesPath.toFile());
        } catch (Exception e) {
          System.out.println("[DEBUG] Failed to add modules, continuing without runtime");
        }
      } else {
        System.out.println("[DEBUG] No Java runtime found, continuing without runtime");
      }
      
      // --- 3) Add source directories to scope
      for (String rootDir : sourceRootsStr.split(Pattern.quote(File.pathSeparator))) {
        if (rootDir == null || rootDir.isBlank()) continue;
        Path p = Paths.get(rootDir).toAbsolutePath().normalize();
        if (Files.isDirectory(p)) {
          System.out.println("[DEBUG] Adding source root to scope: " + p);
          scope.addToScope(ClassLoaderReference.Application, new SourceDirectoryTreeModule(p.toFile()));
        } else {
          System.err.println("[warn] source root not found: " + p);
        }
      }
      
      // --- 4) Build class hierarchy
      System.out.println("[DEBUG] Building class hierarchy...");
      IClassHierarchy cha;
      try {
        cha = ClassHierarchyFactory.makeWithRoot(scope);
        System.out.println("[DEBUG] Class hierarchy built successfully with " + cha.getNumberOfClasses() + " classes");
      } catch (Exception e) {
        System.out.println("[DEBUG] Class hierarchy failed, trying without runtime: " + e.getMessage());
        // Try without runtime
        com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope simpleScope = 
          new com.ibm.wala.cast.java.ipa.callgraph.JavaSourceAnalysisScope();
        
        for (String rootDir : sourceRootsStr.split(Pattern.quote(File.pathSeparator))) {
          if (rootDir == null || rootDir.isBlank()) continue;
          Path p = Paths.get(rootDir).toAbsolutePath().normalize();
          if (Files.isDirectory(p)) {
            simpleScope.addToScope(ClassLoaderReference.Application, new SourceDirectoryTreeModule(p.toFile()));
          }
        }
        
        cha = ClassHierarchyFactory.makeWithRoot(simpleScope);
        System.out.println("[DEBUG] Simple class hierarchy built with " + cha.getNumberOfClasses() + " classes");
      }

      // --- 4) Create entry points
      Iterable<Entrypoint> entries = Util.makeMainEntrypoints(scope, cha);
      if (!entries.iterator().hasNext()) {
        System.out.println("[DEBUG] No main methods found, using AllApplicationEntrypoints");
        entries = new AllApplicationEntrypoints(scope, cha);
      }

      // --- 5) Build call graph
      AnalysisOptions options = new AnalysisOptions(scope, entries);
      AnalysisCache cache = new AnalysisCacheImpl();

      CallGraphBuilder<?> builder =
          Util.makeZeroOneCFABuilder(Language.JAVA, options, cache, cha, scope);

      System.out.println("[DEBUG] Building call graph...");
      CallGraph cg = builder.makeCallGraph(options, null);
      PointerAnalysis<?> pa = builder.getPointerAnalysis();
      System.out.println("[DEBUG] Call graph built successfully with " + cg.getNumberOfNodes() + " nodes");

      // --- 6) Resolve the target member to candidate CG nodes
      MemberCriterion crit = (targetMethod != null)
          ? MemberCriterion.method(targetMethod)
          : MemberCriterion.field(targetField);

      Set<CGNode> candidates = findCandidateNodes(cg, crit);
      System.out.println("[DEBUG] Found " + candidates.size() + " candidate nodes for " + crit.classDotMember);
      
      if (candidates.isEmpty()) {
        System.out.println("[DEBUG] No candidates found, creating fallback slice");
        createFallbackSlice(targetFile, line, outDir, crit.classDotMember);
        return;
      }

      // --- 7) Pick a seed instruction near the requested file:line
      Statement seed = findSeedStatement(candidates, targetFile, line);
      if (seed == null) {
        System.out.println("[DEBUG] No seed statement found, creating fallback slice");
        createFallbackSlice(targetFile, line, outDir, crit.classDotMember);
        return;
      }
      
      System.out.println("[DEBUG] Found seed statement: " + prettyStmt((NormalStatement) seed));

      // --- 8) Slice (backward thin)
      Slicer.DataDependenceOptions dataOpts = Slicer.DataDependenceOptions.FULL;
      Slicer.ControlDependenceOptions ctrlOpts = Slicer.ControlDependenceOptions.NONE;

      System.out.println("[DEBUG] Computing backward slice...");
      Collection<Statement> slice = Slicer.computeBackwardSlice(seed, cg, pa, dataOpts, ctrlOpts);
      System.out.println("[DEBUG] Slice computed with " + slice.size() + " statements");

      // --- 9) Materialize slice to source code
      Path targetRel = projectRoot.relativize(targetFile);
      SliceMaterialized sm = materializeSliceToText(slice, targetRel);
      
      if (sm.fileToLines.isEmpty()) {
        System.out.println("[DEBUG] Empty slice materialized, creating fallback slice");
        createFallbackSlice(targetFile, line, outDir, crit.classDotMember);
        return;
      }
      
      writeTrimmedFiles(sm, projectRoot, outDir);
      writeManifest(sm, outDir);

      System.out.println("WALA slice wrote " + sm.fileToLines.size() + " file(s) to " + outDir);
      
    } catch (Exception e) {
      System.err.println("[ERROR] WALA slicing failed: " + e.getMessage());
      e.printStackTrace();
      System.out.println("[DEBUG] Creating fallback slice due to error");
      createFallbackSlice(targetFile, line, outDir, 
          (targetMethod != null) ? targetMethod : targetField);
    }
  }

  // ========== helpers: args / paths ==========

  static Map<String,String> parseArgs(String[] argv) {
    Map<String,String> m = new LinkedHashMap<>();
    for (int i=0; i<argv.length; i++) {
      String k = argv[i];
      if (!k.startsWith("--")) continue;
      String key = k.substring(2);
      String val = "true";
      if (i+1 < argv.length && !argv[i+1].startsWith("--")) {
        val = argv[++i];
      }
      if ((val.startsWith("\"") && val.endsWith("\"")) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.substring(1, val.length()-1);
      }
      m.put(key, val);
    }
    return m;
  }

  static String req(Map<String,String> m, String k) {
    String v = m.get(k);
    if (v == null) die("Missing required arg: --" + k);
    return v;
  }

  static void die(String msg) { System.err.println(msg); System.exit(2); }

  static Path absolutizeUnderRoot(Path root, String p) {
    Path q = Paths.get(p);
    if (!q.isAbsolute()) q = root.resolve(q);
    return q.normalize();
  }

  // ========== member matching ==========

  /** "pkg.Clazz#method(T1,T2)" OR "pkg.Clazz#field" */
  static final class MemberCriterion {
    final String classDotMember;
    final boolean isMethod;
    private MemberCriterion(String s, boolean isMethod) { this.classDotMember = s; this.isMethod = isMethod; }
    static MemberCriterion method(String m) { return new MemberCriterion(m, true); }
    static MemberCriterion field(String f)  { return new MemberCriterion(f, false); }
  }

  static Set<CGNode> findCandidateNodes(CallGraph cg, MemberCriterion mc) {
    Set<CGNode> out = new HashSet<>();
    String[] parts = mc.classDotMember.split("#", 2);
    if (parts.length != 2) {
      System.out.println("[DEBUG] Invalid member criterion format: " + mc.classDotMember);
      return out;
    }
    String clazz  = parts[0];
    String member = parts[1];

    System.out.println("[DEBUG] Looking for class: " + clazz + ", member: " + member);

    for (CGNode n : cg) {
      MethodReference mr = n.getMethod().getReference();
      if (mr == null) continue;
      
      String cls = mr.getDeclaringClass().getName().toString()
          .replace('/', '.').replaceAll("^L", "").replaceAll(";$", "");
      
      System.out.println("[DEBUG] Checking node: " + cls + "." + mr.getName());

      if (!cls.equals(clazz)) continue;

      if (mc.isMethod) {
        String mname = mr.getName().toString();
        System.out.println("[DEBUG] Method name: " + mname + " (looking for: " + member + ")");
        
        // More flexible method matching
        if (member.startsWith(mname + "(") || member.equals(mname)) {
          out.add(n);
          System.out.println("[DEBUG] Added method node: " + mr);
        }
      } else {
        // field: accept nodes in that class; the seed selection (file+line) will narrow it
        out.add(n);
        System.out.println("[DEBUG] Added field node: " + mr);
      }
    }
    
    System.out.println("[DEBUG] Found " + out.size() + " candidate nodes");
    return out;
  }

  // ========== seed picking ==========

  static Statement findSeedStatement(Set<CGNode> cand, Path expectedFile, int line) {
    Statement best = null;
    int bestDist = Integer.MAX_VALUE;
    String expectedSimple = expectedFile.getFileName().toString();

    System.out.println("[DEBUG] Searching for seed statement in " + cand.size() + " candidate nodes");
    System.out.println("[DEBUG] Target file: " + expectedSimple + ", line: " + line);

    for (CGNode n : cand) {
      IR ir = n.getIR();
      if (ir == null) {
        System.out.println("[DEBUG] Node has no IR: " + n.getMethod().getReference());
        continue;
      }

      IMethod m = n.getMethod();
      System.out.println("[DEBUG] Checking method: " + m.getReference());
      
      // For source analysis, we don't need IBytecodeMethod
      SSAInstruction[] insns = ir.getInstructions();
      if (insns == null || insns.length == 0) {
        System.out.println("[DEBUG] Method has no instructions: " + m.getReference());
        continue;
      }

      // Heuristic: only consider instructions whose simple source filename matches our target file
      String guessed = guessSourceSimpleName(m);
      System.out.println("[DEBUG] Guessed source file: " + guessed + " (expected: " + expectedSimple + ")");
      
      if (guessed != null && !guessed.equals(expectedSimple)) {
        System.out.println("[DEBUG] Skipping method due to filename mismatch");
        continue;
      }

      // For source analysis, try to find the best instruction
      for (int i = 0; i < insns.length; i++) {
        if (insns[i] == null) continue;
        
        // Try to get line number information
        int srcLine = -1;
        try {
          if (m instanceof IBytecodeMethod) {
            IBytecodeMethod bm = (IBytecodeMethod) m;
            int bcIndex = bm.getBytecodeIndex(i);
            if (bcIndex >= 0) {
              srcLine = bm.getLineNumber(bcIndex);
            }
          }
        } catch (Exception e) {
          // Ignore bytecode-related errors in source mode
        }
        
        // If we can't get line number, use instruction index as heuristic
        if (srcLine < 0) {
          // Use a heuristic: assume instructions are roughly ordered by line
          srcLine = i + 1;
        }
        
        int d = Math.abs(srcLine - line);
        System.out.println("[DEBUG] Instruction " + i + " at line " + srcLine + " (distance: " + d + ")");
        
        if (d < bestDist) { 
          bestDist = d; 
          best = new NormalStatement(n, i);
          System.out.println("[DEBUG] New best seed: instruction " + i + " at line " + srcLine);
        }
      }
    }
    
    if (best != null) {
      System.out.println("[DEBUG] Selected seed statement with distance " + bestDist);
    } else {
      System.out.println("[DEBUG] No suitable seed statement found");
    }
    
    return best;
  }

  static String guessSourceSimpleName(IMethod m) {
    String cn = m.getDeclaringClass().getName().toString();
    cn = cn.replace('/', '.').replaceAll("^L", "").replaceAll(";$", "");
    String simple = cn.substring(cn.lastIndexOf('.') + 1);
    return simple + ".java";
  }

  // ========== materialization ==========

  static final class SliceMaterialized {
    final Map<Path, SortedSet<Integer>> fileToLines = new LinkedHashMap<>();
    final List<String> manifest = new ArrayList<>();
  }

  // CHANGED: only collect lines for the actual --targetFile (passed as targetRel)
  static SliceMaterialized materializeSliceToText(Collection<Statement> slice, Path targetRel) {
    SliceMaterialized sm = new SliceMaterialized();
    SortedSet<Integer> keep = new TreeSet<>();

    System.out.println("[DEBUG] Materializing " + slice.size() + " statements to text");

    for (Statement st : slice) {
      if (!(st instanceof NormalStatement)) continue;
      NormalStatement ns = (NormalStatement) st;
      CGNode n = ns.getNode();
      IR ir = n.getIR();
      if (ir == null) continue;

      IMethod m = n.getMethod();
      int idx = ns.getInstructionIndex();
      
      // Try to get line number information
      int srcLine = -1;
      try {
        if (m instanceof IBytecodeMethod) {
          IBytecodeMethod bm = (IBytecodeMethod) m;
          int bcIndex = bm.getBytecodeIndex(idx);
          if (bcIndex >= 0) {
            srcLine = bm.getLineNumber(bcIndex);
          }
        }
      } catch (Exception e) {
        // Ignore bytecode-related errors in source mode
      }
      
      // If we can't get line number, use instruction index as heuristic
      if (srcLine < 0) {
        // Use a heuristic: assume instructions are roughly ordered by line
        srcLine = idx + 1;
      }

      if (srcLine > 0) {
        keep.add(srcLine);
        System.out.println("[DEBUG] Added line " + srcLine + " from instruction " + idx);
      }
    }

    System.out.println("[DEBUG] Materialized " + keep.size() + " lines");

    if (!keep.isEmpty()) {
      sm.fileToLines.put(targetRel, keep);
      for (int l : keep) {
        sm.manifest.add(targetRel + ":" + l + "  // sliced");
      }
    }
    return sm;
  }

  // (kept for future multi-file variants; currently unused)
  static Path guessSourcePathRelative(IMethod m) {
    String cn = m.getDeclaringClass().getName().toString();
    cn = cn.replace('/', '/').replaceAll("^L", "").replaceAll(";$", "");
    String outer = cn.replaceAll("\\$.*$", "");
    return Paths.get(outer + ".java");
  }

  static String prettyStmt(NormalStatement ns) {
    return ns.getNode().getMethod().getReference().toString() + " @" + ns.getInstructionIndex();
  }

  static void writeTrimmedFiles(SliceMaterialized sm, Path projectRoot, Path outDir) throws IOException {
    for (Map.Entry<Path, SortedSet<Integer>> e : sm.fileToLines.entrySet()) {
      Path rel = e.getKey();
      Path src = projectRoot.resolve(rel).normalize();
      if (!Files.exists(src)) continue;

      List<String> all = Files.readAllLines(src, StandardCharsets.UTF_8);
      SortedSet<Integer> keep = e.getValue();

      // Keep only sliced lines (for CFG consumption; not guaranteed compilable)
      List<String> trimmed = new ArrayList<>();
      for (int i = 1; i <= all.size(); i++) {
        if (keep.contains(i)) trimmed.add(all.get(i - 1));
      }

      Path dest = outDir.resolve(rel);
      Files.createDirectories(dest.getParent());
      Files.write(dest, trimmed, StandardCharsets.UTF_8);
    }
  }

  static void writeManifest(SliceMaterialized sm, Path outDir) throws IOException {
    Path man = outDir.resolve("slice.manifest.txt");
    Files.createDirectories(outDir);
    Files.write(man, sm.manifest, StandardCharsets.UTF_8);
  }

  static void createFallbackSlice(Path targetFile, int line, Path outDir, String memberSig) {
    try {
      System.out.println("[DEBUG] Creating fallback slice for " + targetFile + ":" + line);
      
      // Read the original file
      if (!Files.exists(targetFile)) {
        System.err.println("[ERROR] Target file does not exist: " + targetFile);
        return;
      }
      
      List<String> allLines = Files.readAllLines(targetFile, StandardCharsets.UTF_8);
      
      // Create a fallback slice with a window around the target line
      int startLine = Math.max(0, line - 5);
      int endLine = Math.min(allLines.size(), line + 5);
      
      List<String> sliceLines = new ArrayList<>();
      sliceLines.add("// Fallback slice for " + memberSig);
      sliceLines.add("// Target line: " + line);
      sliceLines.add("// Generated by WALA slicer fallback");
      sliceLines.add("");
      
      for (int i = startLine; i < endLine; i++) {
        sliceLines.add(allLines.get(i));
      }
      
      // Write the fallback slice
      Path sliceFile = outDir.resolve(targetFile.getFileName().toString().replace(".java", "_slice.java"));
      Files.createDirectories(sliceFile.getParent());
      Files.write(sliceFile, sliceLines, StandardCharsets.UTF_8);
      
      // Write manifest
      List<String> manifest = new ArrayList<>();
      manifest.add("// Fallback slice manifest");
      manifest.add("target_file: " + targetFile);
      manifest.add("target_line: " + line);
      manifest.add("member: " + memberSig);
      manifest.add("slice_type: fallback");
      manifest.add("lines: " + startLine + "-" + endLine);
      
      Path manifestFile = outDir.resolve("slice.manifest.txt");
      Files.write(manifestFile, manifest, StandardCharsets.UTF_8);
      
      System.out.println("[DEBUG] Fallback slice created: " + sliceFile);
      System.out.println("[DEBUG] Fallback slice contains " + sliceLines.size() + " lines");
      
    } catch (IOException e) {
      System.err.println("[ERROR] Failed to create fallback slice: " + e.getMessage());
      e.printStackTrace();
    }
  }

  static void compileJavaFiles(String sourceRootsStr, Path projectRoot) throws IOException {
    // Simple Java compilation using javac - only compile test files, skip annotated JDK
    for (String rootDir : sourceRootsStr.split(Pattern.quote(File.pathSeparator))) {
      if (rootDir == null || rootDir.isBlank()) continue;
      Path sourceDir = Paths.get(rootDir).toAbsolutePath().normalize();
      if (!Files.isDirectory(sourceDir)) continue;
      
      // Find Java files, but exclude annotated-jdk directory
      List<Path> javaFiles = new ArrayList<>();
      Files.walk(sourceDir)
        .filter(p -> p.toString().endsWith(".java"))
        .filter(p -> !p.toString().contains("annotated-jdk")) // Skip annotated JDK files
        .forEach(javaFiles::add);
      
      System.out.println("[DEBUG] Found " + javaFiles.size() + " Java files to compile in " + sourceDir);
      if (javaFiles.isEmpty()) continue;
      
      // Compile Java files
      ProcessBuilder pb = new ProcessBuilder("javac");
      pb.command().add("-cp");
      // Add runtime classpath plus Checker Framework jars
      String rtCp = System.getProperty("java.class.path");
      String cfDist = "/home/ubuntu/checker-framework-3.42.0/checker/dist";
      String cfCp = String.join(File.pathSeparator,
        cfDist + "/checker.jar",
        cfDist + "/checker-qual.jar",
        cfDist + "/plume-util.jar",
        cfDist + "/javaparser-core-3.26.2.jar"
      );
      // Add javax.annotation-api from runtime classpath (it's now included in our build)
      String fullCp = rtCp + File.pathSeparator + cfCp;
      System.out.println("[DEBUG] Compilation classpath: " + fullCp);
      pb.command().add(fullCp);
      pb.command().add("-d");
      pb.command().add(sourceDir.toString()); // Output to same directory
      
      // Add all Java files
      for (Path javaFile : javaFiles) {
        pb.command().add(javaFile.toString());
      }
      
      try {
        Process proc = pb.start();
        int exitCode = proc.waitFor();
        if (exitCode != 0) {
          System.err.println("Warning: Java compilation failed with exit code " + exitCode);
          // Read error output
          try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(proc.getErrorStream()))) {
            reader.lines().forEach(System.err::println);
          }
        } else {
          System.out.println("Successfully compiled " + javaFiles.size() + " Java files in " + sourceDir);
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Compilation interrupted", e);
      }
    }
  }
}

