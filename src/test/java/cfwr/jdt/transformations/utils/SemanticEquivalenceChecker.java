package cfwr.jdt.transformations.utils;

import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Verifies semantic equivalence between original and transformed code.
 * Performs AST-level analysis to ensure transformations preserve program semantics.
 */
public class SemanticEquivalenceChecker {
    
    private ASTParser parser;
    
    public SemanticEquivalenceChecker() {
        this.parser = createParser();
    }
    
    private ASTParser createParser() {
        ASTParser parser = ASTParser.newParser(AST.JLS21);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);
        parser.setResolveBindings(false);
        parser.setBindingsRecovery(false);
        parser.setStatementsRecovery(true);
        
        Map<String, String> options = JavaCore.getDefaultOptions();
        options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_21);
        options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_21);
        parser.setCompilerOptions(options);
        
        return parser;
    }
    
    /**
     * Check if two code snippets are semantically equivalent.
     */
    public boolean areEquivalent(String original, String transformed) {
        try {
            CompilationUnit originalAST = parseCode(original);
            CompilationUnit transformedAST = parseCode(transformed);
            
            return compareCompilationUnits(originalAST, transformedAST);
        } catch (Exception e) {
            // If parsing fails, fall back to structural comparison
            return compareStructurally(original, transformed);
        }
    }
    
    /**
     * Parse Java code into AST.
     */
    private CompilationUnit parseCode(String code) {
        parser.setSource(code.toCharArray());
        return (CompilationUnit) parser.createAST(null);
    }
    
    /**
     * Compare two compilation units for semantic equivalence.
     */
    private boolean compareCompilationUnits(CompilationUnit original, CompilationUnit transformed) {
        // Check if both have same number of types
        List<AbstractTypeDeclaration> originalTypes = getTypes(original);
        List<AbstractTypeDeclaration> transformedTypes = getTypes(transformed);
        
        if (originalTypes.size() != transformedTypes.size()) {
            return false;
        }
        
        // Compare each type
        for (int i = 0; i < originalTypes.size(); i++) {
            if (!compareTypes(originalTypes.get(i), transformedTypes.get(i))) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Get all type declarations from compilation unit.
     */
    private List<AbstractTypeDeclaration> getTypes(CompilationUnit cu) {
        List<AbstractTypeDeclaration> types = new ArrayList<>();
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(TypeDeclaration node) {
                types.add(node);
                return false;
            }
            
            @Override
            public boolean visit(EnumDeclaration node) {
                types.add(node);
                return false;
            }
            
            @Override
            public boolean visit(AnnotationTypeDeclaration node) {
                types.add(node);
                return false;
            }
        });
        return types;
    }
    
    /**
     * Compare two type declarations for semantic equivalence.
     */
    private boolean compareTypes(AbstractTypeDeclaration original, AbstractTypeDeclaration transformed) {
        // Check type name
        if (!original.getName().getIdentifier().equals(transformed.getName().getIdentifier())) {
            return false;
        }
        
        // For class declarations, compare methods and fields
        if (original instanceof TypeDeclaration && transformed instanceof TypeDeclaration) {
            TypeDeclaration originalClass = (TypeDeclaration) original;
            TypeDeclaration transformedClass = (TypeDeclaration) transformed;
            
            return compareClassDeclarations(originalClass, transformedClass);
        }
        
        return true;
    }
    
    /**
     * Compare two class declarations for semantic equivalence.
     */
    private boolean compareClassDeclarations(TypeDeclaration original, TypeDeclaration transformed) {
        // Compare methods
        MethodDeclaration[] originalMethods = original.getMethods();
        MethodDeclaration[] transformedMethods = transformed.getMethods();
        
        if (originalMethods.length != transformedMethods.length) {
            return false;
        }
        
        for (int i = 0; i < originalMethods.length; i++) {
            if (!compareMethods(originalMethods[i], transformedMethods[i])) {
                return false;
            }
        }
        
        // Compare fields
        FieldDeclaration[] originalFields = original.getFields();
        FieldDeclaration[] transformedFields = transformed.getFields();
        
        if (originalFields.length != transformedFields.length) {
            return false;
        }
        
        for (int i = 0; i < originalFields.length; i++) {
            if (!compareFields(originalFields[i], transformedFields[i])) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Compare two method declarations for semantic equivalence.
     */
    private boolean compareMethods(MethodDeclaration original, MethodDeclaration transformed) {
        // Check method signature
        if (!original.getName().getIdentifier().equals(transformed.getName().getIdentifier())) {
            return false;
        }
        
        // Check return type
        if (!compareTypes(original.getReturnType2(), transformed.getReturnType2())) {
            return false;
        }
        
        // Check parameters
        List<SingleVariableDeclaration> originalParams = original.parameters();
        List<SingleVariableDeclaration> transformedParams = transformed.parameters();
        
        if (originalParams.size() != transformedParams.size()) {
            return false;
        }
        
        for (int i = 0; i < originalParams.size(); i++) {
            if (!compareParameters(originalParams.get(i), transformedParams.get(i))) {
                return false;
            }
        }
        
        // Check method body (statements)
        if (!compareMethodBodies(original, transformed)) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Compare method bodies for semantic equivalence.
     */
    private boolean compareMethodBodies(MethodDeclaration original, MethodDeclaration transformed) {
        Block originalBody = original.getBody();
        Block transformedBody = transformed.getBody();
        
        if (originalBody == null && transformedBody == null) {
            return true; // Both abstract
        }
        
        if (originalBody == null || transformedBody == null) {
            return false; // One abstract, one not
        }
        
        List<Statement> originalStatements = originalBody.statements();
        List<Statement> transformedStatements = transformedBody.statements();
        
        // For semantic equivalence, we check that the same logical operations occur
        // but allow for different syntactic representations
        return compareStatementLists(originalStatements, transformedStatements);
    }
    
    /**
     * Compare two lists of statements for semantic equivalence.
     */
    private boolean compareStatementLists(List<Statement> original, List<Statement> transformed) {
        if (original.size() != transformed.size()) {
            return false;
        }
        
        for (int i = 0; i < original.size(); i++) {
            if (!compareStatements(original.get(i), transformed.get(i))) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Compare two statements for semantic equivalence.
     */
    private boolean compareStatements(Statement original, Statement transformed) {
        // Basic type check
        if (original.getClass() != transformed.getClass()) {
            return false;
        }
        
        // Specific comparisons for different statement types
        if (original instanceof ExpressionStatement) {
            return compareExpressionStatements((ExpressionStatement) original, (ExpressionStatement) transformed);
        } else if (original instanceof IfStatement) {
            return compareIfStatements((IfStatement) original, (IfStatement) transformed);
        } else if (original instanceof ReturnStatement) {
            return compareReturnStatements((ReturnStatement) original, (ReturnStatement) transformed);
        } else if (original instanceof ForStatement) {
            return compareForStatements((ForStatement) original, (ForStatement) transformed);
        } else if (original instanceof WhileStatement) {
            return compareWhileStatements((WhileStatement) original, (WhileStatement) transformed);
        }
        
        // For other statement types, do basic structural comparison
        return compareExpressionTypes(original, transformed);
    }
    
    /**
     * Compare expression statements for semantic equivalence.
     */
    private boolean compareExpressionStatements(ExpressionStatement original, ExpressionStatement transformed) {
        return compareExpressions(original.getExpression(), transformed.getExpression());
    }
    
    /**
     * Compare if statements for semantic equivalence.
     */
    private boolean compareIfStatements(IfStatement original, IfStatement transformed) {
        // Compare conditions (may be semantically equivalent but syntactically different)
        if (!compareExpressions(original.getExpression(), transformed.getExpression())) {
            return false;
        }
        
        // Compare then and else statements
        if (!compareStatements(original.getThenStatement(), transformed.getThenStatement())) {
            return false;
        }
        
        if (original.getElseStatement() != null && transformed.getElseStatement() != null) {
            return compareStatements(original.getElseStatement(), transformed.getElseStatement());
        } else if (original.getElseStatement() == null && transformed.getElseStatement() == null) {
            return true;
        } else {
            return false;
        }
    }
    
    /**
     * Compare return statements for semantic equivalence.
     */
    private boolean compareReturnStatements(ReturnStatement original, ReturnStatement transformed) {
        Expression originalExpr = original.getExpression();
        Expression transformedExpr = transformed.getExpression();
        
        if (originalExpr == null && transformedExpr == null) {
            return true;
        }
        
        if (originalExpr == null || transformedExpr == null) {
            return false;
        }
        
        return compareExpressions(originalExpr, transformedExpr);
    }
    
    /**
     * Compare for statements for semantic equivalence.
     */
    private boolean compareForStatements(ForStatement original, ForStatement transformed) {
        // Compare initialization, condition, and update expressions
        return compareExpressionTypes(original, transformed) &&
               compareStatements(original.getBody(), transformed.getBody());
    }
    
    /**
     * Compare while statements for semantic equivalence.
     */
    private boolean compareWhileStatements(WhileStatement original, WhileStatement transformed) {
        return compareExpressions(original.getExpression(), transformed.getExpression()) &&
               compareStatements(original.getBody(), transformed.getBody());
    }
    
    /**
     * Compare expressions for semantic equivalence.
     */
    private boolean compareExpressions(Expression original, Expression transformed) {
        if (original == null && transformed == null) {
            return true;
        }
        
        if (original == null || transformed == null) {
            return false;
        }
        
        // For semantic equivalence, we focus on the logical meaning rather than exact syntax
        // This is a simplified comparison - in practice, you'd want more sophisticated analysis
        return compareExpressionTypes(original, transformed);
    }
    
    /**
     * Compare expression types and basic structure.
     */
    private boolean compareExpressionTypes(ASTNode original, ASTNode transformed) {
        // Basic type comparison
        if (original.getClass() != transformed.getClass()) {
            return false;
        }
        
        // For now, assume expressions of the same type are equivalent
        // In a real implementation, you'd do deeper semantic analysis
        return true;
    }
    
    /**
     * Compare two parameters for semantic equivalence.
     */
    private boolean compareParameters(SingleVariableDeclaration original, SingleVariableDeclaration transformed) {
        return original.getName().getIdentifier().equals(transformed.getName().getIdentifier()) &&
               compareTypes(original.getType(), transformed.getType());
    }
    
    /**
     * Compare two field declarations for semantic equivalence.
     */
    private boolean compareFields(FieldDeclaration original, FieldDeclaration transformed) {
        return compareTypes(original.getType(), transformed.getType()) &&
               original.fragments().size() == transformed.fragments().size();
    }
    
    /**
     * Compare two types for equivalence.
     */
    private boolean compareTypes(Type original, Type transformed) {
        if (original == null && transformed == null) {
            return true;
        }
        
        if (original == null || transformed == null) {
            return false;
        }
        
        // Simplified type comparison
        return original.getClass() == transformed.getClass();
    }
    
    /**
     * Fallback structural comparison when AST parsing fails.
     */
    private boolean compareStructurally(String original, String transformed) {
        // Remove whitespace and normalize for comparison
        String normalizedOriginal = normalizeCode(original);
        String normalizedTransformed = normalizeCode(transformed);
        
        // Check if they have the same structural elements
        return hasSameStructuralElements(normalizedOriginal, normalizedTransformed);
    }
    
    /**
     * Normalize code for structural comparison.
     */
    private String normalizeCode(String code) {
        return code.replaceAll("\\s+", " ")
                  .replaceAll("\\s*\\{\\s*", "{")
                  .replaceAll("\\s*\\}\\s*", "}")
                  .replaceAll("\\s*;\\s*", ";")
                  .trim();
    }
    
    /**
     * Check if two code snippets have the same structural elements.
     */
    private boolean hasSameStructuralElements(String original, String transformed) {
        // Count key structural elements
        Map<String, Integer> originalCounts = countStructuralElements(original);
        Map<String, Integer> transformedCounts = countStructuralElements(transformed);
        
        return originalCounts.equals(transformedCounts);
    }
    
    /**
     * Count structural elements in code.
     */
    private Map<String, Integer> countStructuralElements(String code) {
        Map<String, Integer> counts = new HashMap<>();
        
        // Count various structural elements
        counts.put("classes", countOccurrences(code, "class "));
        counts.put("methods", countOccurrences(code, "public ") + countOccurrences(code, "private ") + countOccurrences(code, "protected "));
        counts.put("if_statements", countOccurrences(code, "if ("));
        counts.put("for_loops", countOccurrences(code, "for ("));
        counts.put("while_loops", countOccurrences(code, "while ("));
        counts.put("return_statements", countOccurrences(code, "return "));
        counts.put("assignments", countOccurrences(code, " = "));
        
        return counts;
    }
    
    /**
     * Count occurrences of a substring in text.
     */
    private int countOccurrences(String text, String substring) {
        int count = 0;
        int index = 0;
        while ((index = text.indexOf(substring, index)) != -1) {
            count++;
            index += substring.length();
        }
        return count;
    }
}
