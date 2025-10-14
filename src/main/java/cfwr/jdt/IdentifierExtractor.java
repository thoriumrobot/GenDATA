package cfwr.jdt;

import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.formatter.DefaultCodeFormatterConstants;

import java.util.*;
import java.util.stream.Collectors;

/**
 * JDT-based identifier extractor.
 * Replaces regex-based identifier extraction with robust AST parsing.
 */
public class IdentifierExtractor {
    
    private ASTParser parser;
    
    public IdentifierExtractor() {
        this.parser = createParser();
    }
    
    private ASTParser createParser() {
        ASTParser parser = ASTParser.newParser(AST.JLS21);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);
        parser.setResolveBindings(false);
        parser.setBindingsRecovery(false);
        parser.setStatementsRecovery(true);
        
        Map<String, String> options = DefaultCodeFormatterConstants.getEclipseDefaultSettings();
        options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_21);
        parser.setCompilerOptions(options);
        
        return parser;
    }
    
    public Map<String, List<String>> extractIdentifiers(String javaCode) {
        Map<String, List<String>> identifiers = new HashMap<>();
        
        parser.setSource(javaCode.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        
        if (cu == null) {
            return identifiers; // Return empty map if parsing failed
        }
        
        // Initialize categories
        identifiers.put("variables", new ArrayList<>());
        identifiers.put("methods", new ArrayList<>());
        identifiers.put("types", new ArrayList<>());
        identifiers.put("packages", new ArrayList<>());
        identifiers.put("fields", new ArrayList<>());
        identifiers.put("parameters", new ArrayList<>());
        identifiers.put("local_variables", new ArrayList<>());
        
        // Extract package declaration
        if (cu.getPackage() != null) {
            identifiers.get("packages").add(cu.getPackage().getName().getFullyQualifiedName());
        }
        
        // Visit AST nodes to extract identifiers
        cu.accept(new IdentifierVisitor(identifiers));
        
        return identifiers;
    }
    
    private static class IdentifierVisitor extends ASTVisitor {
        private final Map<String, List<String>> identifiers;
        private final Set<String> javaKeywords;
        
        public IdentifierVisitor(Map<String, List<String>> identifiers) {
            this.identifiers = identifiers;
            this.javaKeywords = getJavaKeywords();
        }
        
        @Override
        public boolean visit(TypeDeclaration node) {
            String typeName = node.getName().getIdentifier();
            if (!isJavaKeyword(typeName)) {
                identifiers.get("types").add(typeName);
            }
            return true;
        }
        
        @Override
        public boolean visit(MethodDeclaration node) {
            String methodName = node.getName().getIdentifier();
            if (!isJavaKeyword(methodName)) {
                identifiers.get("methods").add(methodName);
            }
            
            // Extract parameters
            for (Object param : node.parameters()) {
                if (param instanceof SingleVariableDeclaration) {
                    SingleVariableDeclaration svd = (SingleVariableDeclaration) param;
                    String paramName = svd.getName().getIdentifier();
                    if (!isJavaKeyword(paramName)) {
                        identifiers.get("parameters").add(paramName);
                    }
                }
            }
            
            return true;
        }
        
        @Override
        public boolean visit(VariableDeclarationStatement node) {
            for (Object fragment : node.fragments()) {
                if (fragment instanceof VariableDeclarationFragment) {
                    VariableDeclarationFragment vdf = (VariableDeclarationFragment) fragment;
                    String varName = vdf.getName().getIdentifier();
                    if (!isJavaKeyword(varName)) {
                        identifiers.get("local_variables").add(varName);
                        identifiers.get("variables").add(varName);
                    }
                }
            }
            return true;
        }
        
        @Override
        public boolean visit(VariableDeclarationExpression node) {
            for (Object fragment : node.fragments()) {
                if (fragment instanceof VariableDeclarationFragment) {
                    VariableDeclarationFragment vdf = (VariableDeclarationFragment) fragment;
                    String varName = vdf.getName().getIdentifier();
                    if (!isJavaKeyword(varName)) {
                        identifiers.get("local_variables").add(varName);
                        identifiers.get("variables").add(varName);
                    }
                }
            }
            return true;
        }
        
        @Override
        public boolean visit(FieldDeclaration node) {
            for (Object fragment : node.fragments()) {
                if (fragment instanceof VariableDeclarationFragment) {
                    VariableDeclarationFragment vdf = (VariableDeclarationFragment) fragment;
                    String fieldName = vdf.getName().getIdentifier();
                    if (!isJavaKeyword(fieldName)) {
                        identifiers.get("fields").add(fieldName);
                        identifiers.get("variables").add(fieldName);
                    }
                }
            }
            return true;
        }
        
        @Override
        public boolean visit(SimpleName node) {
            String name = node.getIdentifier();
            
            // Check if this is a method call
            ASTNode parent = node.getParent();
            if (parent instanceof MethodInvocation) {
                MethodInvocation methodInvocation = (MethodInvocation) parent;
                if (methodInvocation.getName() == node && !isJavaKeyword(name)) {
                    identifiers.get("methods").add(name);
                }
            }
            
            // Check if this is a field access
            if (parent instanceof FieldAccess) {
                FieldAccess fieldAccess = (FieldAccess) parent;
                if (fieldAccess.getName() == node && !isJavaKeyword(name)) {
                    identifiers.get("fields").add(name);
                    identifiers.get("variables").add(name);
                }
            }
            
            // Check if this is a qualified name (package/type)
            if (parent instanceof QualifiedName) {
                QualifiedName qualifiedName = (QualifiedName) parent;
                if (qualifiedName.getName() == node && !isJavaKeyword(name)) {
                    identifiers.get("types").add(name);
                }
            }
            
            return true;
        }
        
        private boolean isJavaKeyword(String name) {
            return javaKeywords.contains(name);
        }
        
        private Set<String> getJavaKeywords() {
            return new HashSet<>(Arrays.asList(
                "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
                "class", "const", "continue", "default", "do", "double", "else", "enum",
                "extends", "final", "finally", "float", "for", "goto", "if", "implements",
                "import", "instanceof", "int", "interface", "long", "native", "new",
                "package", "private", "protected", "public", "return", "short", "static",
                "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
                "transient", "try", "void", "volatile", "while", "true", "false", "null"
            ));
        }
    }
}
