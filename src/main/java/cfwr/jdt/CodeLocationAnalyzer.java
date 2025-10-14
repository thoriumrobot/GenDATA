package cfwr.jdt;

import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.formatter.DefaultCodeFormatterConstants;
import org.eclipse.jdt.core.compiler.IProblem;

import java.util.*;
import java.util.stream.Collectors;

/**
 * JDT-based code location analyzer.
 * Replaces regex-based parsing in code_location_analyzer.py with robust AST parsing.
 */
public class CodeLocationAnalyzer {
    
    private ASTParser parser;
    
    public CodeLocationAnalyzer() {
        this.parser = createParser();
    }
    
    private ASTParser createParser() {
        ASTParser parser = ASTParser.newParser(AST.JLS21);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);
        parser.setResolveBindings(false);
        parser.setBindingsRecovery(false);
        parser.setStatementsRecovery(true);
        
        // Set compiler options for better parsing
        Map<String, String> options = DefaultCodeFormatterConstants.getEclipseDefaultSettings();
        options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_21);
        parser.setCompilerOptions(options);
        
        return parser;
    }
    
    public List<CodeLocation> analyzeCode(String javaCode) {
        List<CodeLocation> locations = new ArrayList<>();
        
        // Handle null or empty input
        if (javaCode == null || javaCode.trim().isEmpty()) {
            return locations;
        }
        
        parser.setSource(javaCode.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        
        if (cu == null) {
            return locations; // Return empty list if parsing failed
        }
        
        // Find class-level locations
        locations.addAll(findClassLevelLocations(cu));
        
        // Find method-level locations
        locations.addAll(findMethodLevelLocations(cu));
        
        // Find statement-level locations
        locations.addAll(findStatementLevelLocations(cu));
        
        // Find expression-level locations
        locations.addAll(findExpressionLevelLocations(cu));
        
        // Find block-level locations
        locations.addAll(findBlockLevelLocations(cu));
        
        return locations;
    }
    
    public boolean validateSyntax(String javaCode) {
        // Handle null or empty input
        if (javaCode == null || javaCode.trim().isEmpty()) {
            return false;
        }
        
        parser.setSource(javaCode.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        
        if (cu == null) {
            return false;
        }
        
        // Check for compilation problems
        IProblem[] problems = cu.getProblems();
        for (IProblem problem : problems) {
            if (problem.isError()) {
                return false;
            }
        }
        
        return true;
    }
    
    private List<CodeLocation> findClassLevelLocations(CompilationUnit cu) {
        List<CodeLocation> locations = new ArrayList<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(TypeDeclaration node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("class_name", node.getName().getIdentifier());
                context.put("modifiers", getModifiers(node));
                context.put("type", node.isInterface() ? "interface" : "class");
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("RANDOM_METHOD_INSERTION");
                applicableTransforms.add("METHOD_EXTRACTION");
                applicableTransforms.add("BUILDER_PATTERN");
                applicableTransforms.add("FUNCTIONAL_CONVERSION");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0, // Column positions would require more complex calculation
                    "CLASS_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
        });
        
        return locations;
    }
    
    private List<CodeLocation> findMethodLevelLocations(CompilationUnit cu) {
        List<CodeLocation> locations = new ArrayList<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodDeclaration node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("method_name", node.getName().getIdentifier());
                context.put("return_type", node.getReturnType2() != null ? 
                    node.getReturnType2().toString() : "void");
                context.put("modifiers", getModifiers(node));
                context.put("parameters", getParameters(node));
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("METHOD_EXTRACTION");
                applicableTransforms.add("VARIABLE_INLINING");
                applicableTransforms.add("GUARD_REVERSAL");
                applicableTransforms.add("TERNARY_IF_ELSE");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "METHOD_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
        });
        
        return locations;
    }
    
    private List<CodeLocation> findStatementLevelLocations(CompilationUnit cu) {
        List<CodeLocation> locations = new ArrayList<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(VariableDeclarationStatement node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("type", node.getType().toString());
                context.put("variables", node.fragments().stream()
                    .map(f -> f.toString())
                    .collect(Collectors.toList()));
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("VARIABLE_INLINING");
                applicableTransforms.add("SIMPLE_ASSIGNMENT");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "STATEMENT_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
            
            @Override
            public boolean visit(Assignment node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("operator", node.getOperator().toString());
                context.put("left_hand_side", node.getLeftHandSide().toString());
                context.put("right_hand_side", node.getRightHandSide().toString());
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("SIMPLE_ASSIGNMENT");
                applicableTransforms.add("IDENTITY_MATH");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "STATEMENT_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
            
            @Override
            public boolean visit(IfStatement node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("condition", node.getExpression().toString());
                context.put("has_else", node.getElseStatement() != null);
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("GUARD_REVERSAL");
                applicableTransforms.add("CONDITIONAL_RESTRUCTURING");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "STATEMENT_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
        });
        
        return locations;
    }
    
    private List<CodeLocation> findExpressionLevelLocations(CompilationUnit cu) {
        List<CodeLocation> locations = new ArrayList<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("operator", node.getOperator().toString());
                context.put("left_operand", node.getLeftOperand().toString());
                context.put("right_operand", node.getRightOperand().toString());
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("IDENTITY_MATH");
                applicableTransforms.add("ARITHMETIC_PROPERTIES");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "EXPRESSION_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
            
            @Override
            public boolean visit(ConditionalExpression node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("condition", node.getExpression().toString());
                context.put("then_expression", node.getThenExpression().toString());
                context.put("else_expression", node.getElseExpression().toString());
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("TERNARY_IF_ELSE");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "EXPRESSION_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
        });
        
        return locations;
    }
    
    private List<CodeLocation> findBlockLevelLocations(CompilationUnit cu) {
        List<CodeLocation> locations = new ArrayList<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(Block node) {
                int startLine = cu.getLineNumber(node.getStartPosition());
                int endLine = cu.getLineNumber(node.getStartPosition() + node.getLength());
                
                Map<String, Object> context = new HashMap<>();
                context.put("statement_count", node.statements().size());
                
                Set<String> applicableTransforms = new HashSet<>();
                applicableTransforms.add("BLOCK_RESTRUCTURING");
                
                String codeSnippet = extractCodeSnippet(cu, node.getStartPosition(), node.getLength());
                
                locations.add(new CodeLocation(
                    startLine, endLine,
                    0, 0,
                    "BLOCK_LEVEL",
                    context,
                    codeSnippet,
                    applicableTransforms
                ));
                
                return true;
            }
        });
        
        return locations;
    }
    
    private List<String> getModifiers(BodyDeclaration node) {
        List<String> modifiers = new ArrayList<>();
        for (Object modifier : node.modifiers()) {
            if (modifier instanceof Modifier) {
                modifiers.add(((Modifier) modifier).getKeyword().toString());
            }
        }
        return modifiers;
    }
    
    @SuppressWarnings("unchecked")
    private List<String> getParameters(MethodDeclaration node) {
        List<String> parameters = new ArrayList<>();
        for (Object param : node.parameters()) {
            if (param instanceof SingleVariableDeclaration) {
                parameters.add(param.toString());
            }
        }
        return parameters;
    }
    
    private String extractCodeSnippet(CompilationUnit cu, int startPosition, int length) {
        String source = cu.toString();
        if (startPosition >= 0 && startPosition + length <= source.length()) {
            return source.substring(startPosition, startPosition + length);
        }
        return "";
    }
}
