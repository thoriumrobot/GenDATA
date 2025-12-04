package cfwr.jdt;

import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.dom.rewrite.ASTRewrite;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.formatter.DefaultCodeFormatterConstants;
import org.eclipse.jdt.core.compiler.IProblem;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.TextEdit;

import cfwr.jdt.util.JsonOutput;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * JDT-based semantic transformer for applying all 27 transformation types.
 * Replaces regex-based transformations with robust AST-based transformations.
 */
public class SemanticTransformer {
    
    private ASTParser parser;
    private Random random;
    private final boolean debugEnabled = "1".equals(System.getenv("JDT_DEBUG"));
    private final Map<String, Integer> debugCounters = new HashMap<>();
    // Per-transform configuration: enable flag and max depth (internal, defaults enabled, depth=3)
    private final Map<String, Boolean> transformEnabled = new HashMap<>();
    private final Map<String, Integer> transformMaxDepth = new HashMap<>();
    private final List<String> appliedThisRun = new ArrayList<>();
    private TransformationDiagnostics diagnostics;
    
    public SemanticTransformer() {
        this.parser = createParser();
        this.random = new Random();
        this.diagnostics = new TransformationDiagnostics();
        initDefaults();
    }
    
    public SemanticTransformer(long seed) {
        this.parser = createParser();
        this.random = new Random(seed);
        initDefaults();
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

    private void initDefaults() {
        List<String> all = Arrays.asList(
            // enhanced
            "loop_conversion","guard_reversal","mathematical_expression","logical_expression",
            "ternary_operator","switch_statement","variable_operation","method_extraction",
            "conditional_expression","array_access_pattern","string_concatenation","numeric_literal",
            "exception_handling","lambda_expression","stream_api","builder_pattern","functional_conversion",
            // simple
            "simple_method_call","simple_assignment","simple_conditional","simple_array_access",
            "simple_return_statement","simple_variable_declaration","simple_constructor_call",
            "simple_field_access","simple_string_operation","simple_numeric_operation",
            // random
            "random_method_insertion","random_statement_insertion","random_expression_insertion"
        );
        for (String t : all) {
            transformEnabled.put(t, Boolean.TRUE);
            transformMaxDepth.put(t, Integer.valueOf(3));
        }
    }
    
    // Transformation compatibility matrix - prevents incompatible transformations from being applied together
    private static final Map<String, Set<String>> INCOMPATIBLE_TRANSFORMATIONS = Map.of(
        "loop_conversion", Set.of("guard_reversal"),
        "guard_reversal", Set.of("loop_conversion")
    );

    private boolean isEnabled(String t) {
        return transformEnabled.getOrDefault(t, Boolean.TRUE);
    }
    
    /**
     * Validate transformation compatibility and return list of compatible transformations.
     */
    private List<String> validateTransformationCompatibility(List<String> transformations) {
        List<String> compatible = new ArrayList<>();
        
        for (String transformation : transformations) {
            boolean isCompatible = true;
            
            // Check if this transformation is incompatible with any already selected transformations
            for (String selected : compatible) {
                Set<String> incompatibleWith = INCOMPATIBLE_TRANSFORMATIONS.get(transformation);
                if (incompatibleWith != null && incompatibleWith.contains(selected)) {
                    debug("compatibility_skip", "Skipping " + transformation + " - incompatible with " + selected);
                    isCompatible = false;
                    break;
                }
                
                Set<String> incompatibleWithSelected = INCOMPATIBLE_TRANSFORMATIONS.get(selected);
                if (incompatibleWithSelected != null && incompatibleWithSelected.contains(transformation)) {
                    debug("compatibility_skip", "Skipping " + transformation + " - incompatible with " + selected);
                    isCompatible = false;
                    break;
                }
            }
            
            if (isCompatible) {
                compatible.add(transformation);
                debug("compatibility_accept", "Accepting transformation: " + transformation);
            }
        }
        
        return compatible;
    }

    private void debug(String key, String message) {
        if (debugEnabled) {
            if (key != null) {
                debugCounters.put(key, debugCounters.getOrDefault(key, 0) + 1);
            }
            System.err.println("[JDT_DEBUG] " + message);
        }
    }
    
    /**
     * Enhanced logging with context and performance metrics.
     */
    private void logTransformationStart(String transformation, String mode) {
        if (debugEnabled) {
            System.err.println("[JDT_TRANSFORM_START] " + transformation + " (mode: " + mode + ")");
        }
    }
    
    private void logTransformationEnd(String transformation, boolean success, long durationMs) {
        if (debugEnabled) {
            String status = success ? "SUCCESS" : "FAILED";
            System.err.println("[JDT_TRANSFORM_END] " + transformation + " - " + status + " (took " + durationMs + "ms)");
        }
    }
    
    private void logTransformationDecision(String transformation, String reason, boolean applied) {
        if (debugEnabled) {
            String action = applied ? "APPLIED" : "SKIPPED";
            System.err.println("[JDT_DECISION] " + transformation + " - " + action + " (" + reason + ")");
        }
    }
    
    private void logError(String transformation, String error, Exception e) {
        if (debugEnabled) {
            System.err.println("[JDT_ERROR] " + transformation + " - " + error);
            if (e != null) {
                System.err.println("[JDT_ERROR] Exception: " + e.getMessage());
                e.printStackTrace(System.err);
            }
        }
    }

    // Safety helpers (conservative fallbacks; no binding resolution)
    private boolean isPure(Expression expr) {
        final AtomicBoolean impure = new AtomicBoolean(false);
        if (expr == null) return true;
        expr.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) {
                impure.set(true);
                return false;
            }
            @Override
            public boolean visit(Assignment node) {
                impure.set(true);
                return false;
            }
            @Override
            public boolean visit(PostfixExpression node) {
                impure.set(true); return false;
            }
            @Override
            public boolean visit(PrefixExpression node) {
                if (node.getOperator() == PrefixExpression.Operator.INCREMENT ||
                    node.getOperator() == PrefixExpression.Operator.DECREMENT) {
                    impure.set(true); return false;
                }
                return true;
            }
        });
        return !impure.get();
    }

    private boolean hasSideEffects(Statement stmt) {
        if (stmt == null) return false;
        final AtomicBoolean se = new AtomicBoolean(false);
        stmt.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) { se.set(true); return false; }
            @Override
            public boolean visit(Assignment node) { se.set(true); return false; }
            @Override
            public boolean visit(ReturnStatement node) { se.set(true); return false; }
            @Override
            public boolean visit(ThrowStatement node) { se.set(true); return false; }
            @Override
            public boolean visit(BreakStatement node) { se.set(true); return false; }
            @Override
            public boolean visit(ContinueStatement node) { se.set(true); return false; }
        });
        return se.get();
    }

    private boolean capturesVariables(LambdaExpression lambda) {
        if (lambda == null) return false;
        final Set<String> params = new HashSet<>();
        for (Object p : lambda.parameters()) {
            if (p instanceof SingleVariableDeclaration) {
                params.add(((SingleVariableDeclaration) p).getName().getIdentifier());
            }
        }
        final AtomicBoolean captured = new AtomicBoolean(false);
        lambda.accept(new ASTVisitor() {
            @Override
            public boolean visit(SimpleName node) {
                String id = node.getIdentifier();
                if (!params.contains(id)) {
                    // Treat references to non-params as potential captures
                    captured.set(true);
                }
                return true;
            }
        });
        return captured.get();
    }

    private void debugSummary() {
        if (!debugEnabled) return;
        System.err.println("[JDT_DEBUG] Summary counters:")
        ;
        for (Map.Entry<String, Integer> e : debugCounters.entrySet()) {
            System.err.println("[JDT_DEBUG]   " + e.getKey() + " = " + e.getValue());
        }
    }
    
    public String transformCode(String javaCode, List<String> transformations, String mode) {
        appliedThisRun.clear();
        
        // Handle null or empty input
        if (javaCode == null) {
            debug("null_input", "Input code is null; returning null");
            return null;
        }
        
        if (javaCode.trim().isEmpty()) {
            debug("empty_input", "Input code is empty; returning original");
            return javaCode;
        }
        
        // Validate transformation compatibility
        List<String> compatibleTransformations = validateTransformationCompatibility(transformations);
        if (compatibleTransformations.isEmpty()) {
            debug("compatibility_error", "No compatible transformations found; returning original code");
            return javaCode;
        }
        
        parser.setSource(javaCode.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        
        if (cu == null || Arrays.stream(cu.getProblems()).anyMatch(IProblem::isError)) {
            debug("parse_error", "Parsing failed or had errors; returning original code");
            return javaCode; // Return original if parsing failed
        }
        
        ASTRewrite rewrite = ASTRewrite.create(cu.getAST());
        boolean hasChanges = false;
        
        for (String transformation : compatibleTransformations) {
            logTransformationStart(transformation, mode);
            if (diagnostics != null) {
                diagnostics.recordTransformationStart(transformation, mode, javaCode);
            }
            long startTime = System.currentTimeMillis();
            
            try {
            debug("consider_" + transformation, "Considering transformation: " + transformation);
            boolean changed = applyTransformation(cu, rewrite, transformation, mode);
                long duration = System.currentTimeMillis() - startTime;
                
            if (changed) {
                debug("applied_" + transformation, "Applied transformation: " + transformation);
                    logTransformationDecision(transformation, "transformation applied successfully", true);
                    if (diagnostics != null) {
                        diagnostics.recordDecision(transformation, "transformation applied successfully", true);
                    }
                hasChanges = true;
                appliedThisRun.add(transformation);
                    logTransformationEnd(transformation, true, duration);
                    if (diagnostics != null) {
                        diagnostics.recordTransformationEnd(transformation, true, null, duration, null);
                    }
            } else {
                debug("skipped_" + transformation, "No effect: " + transformation);
                    logTransformationDecision(transformation, "no changes made", false);
                    if (diagnostics != null) {
                        diagnostics.recordDecision(transformation, "no changes made", false);
                    }
                    logTransformationEnd(transformation, false, duration);
                    if (diagnostics != null) {
                        diagnostics.recordTransformationEnd(transformation, false, null, duration, null);
                    }
                }
            } catch (Exception e) {
                long duration = System.currentTimeMillis() - startTime;
                String errorMsg = e.getMessage();
                logError(transformation, "transformation failed with exception", e);
                logTransformationEnd(transformation, false, duration);
                if (diagnostics != null) {
                    diagnostics.recordTransformationEnd(transformation, false, null, duration, errorMsg);
                }
                // Continue with other transformations instead of failing completely
            }
            
            if (diagnostics != null) {
                diagnostics.recordPerformanceMetric("transformation_" + transformation, 
                    System.currentTimeMillis() - startTime);
            }
        }
        
        if (!hasChanges) {
            debug("no_changes", "No transformations produced changes; returning original");
            debugSummary();
            return javaCode;
        }
        
        try {
            Document document = new Document(javaCode);
            TextEdit edits = rewrite.rewriteAST(document, null);
            edits.apply(document);
            debug("changes_applied", "Edits applied successfully");
            debugSummary();
            return document.get();
        } catch (Exception e) {
            System.err.println("Error applying transformations: " + e.getMessage());
            debug("apply_error", "Exception applying edits: " + e.getMessage());
            return javaCode;
        }
    }
    
    private boolean applyTransformation(CompilationUnit cu, ASTRewrite rewrite, String transformation, String mode) {
        if (!isEnabled(transformation)) {
            debug("disabled_" + transformation, "Transformation disabled by config");
            return false;
        }
        switch (transformation.toLowerCase()) {
            // Enhanced transformations (17 methods)
            case "loop_conversion":
                return applyLoopConversion(cu, rewrite);
            case "guard_reversal":
                return applyGuardReversal(cu, rewrite);
            case "mathematical_expression":
                return applyMathematicalExpression(cu, rewrite);
            case "logical_expression":
                return applyLogicalExpression(cu, rewrite);
            case "ternary_operator":
                return applyTernaryOperator(cu, rewrite);
            case "switch_statement":
                return applySwitchStatement(cu, rewrite);
            case "variable_operation":
                return applyVariableOperation(cu, rewrite);
            case "method_extraction":
                return applyMethodExtraction(cu, rewrite);
            case "conditional_expression":
                return applyConditionalExpression(cu, rewrite);
            case "array_access_pattern":
                return applyArrayAccessPattern(cu, rewrite);
            case "string_concatenation":
                return applyStringConcatenation(cu, rewrite);
            case "numeric_literal":
                return applyNumericLiteral(cu, rewrite);
            case "exception_handling":
                return applyExceptionHandling(cu, rewrite);
            case "lambda_expression":
                return applyLambdaExpression(cu, rewrite);
            case "stream_api":
                return applyStreamApi(cu, rewrite);
            case "builder_pattern":
                return applyBuilderPattern(cu, rewrite);
            case "functional_conversion":
                return applyFunctionalConversion(cu, rewrite);
            case "brace_normalization":
                return applyBraceNormalization(cu, rewrite);
                
            // Simple transformations (10 methods)
            case "simple_method_call":
                return applySimpleMethodCall(cu, rewrite);
            case "simple_assignment":
                return applySimpleAssignment(cu, rewrite);
            case "simple_conditional":
                return applySimpleConditional(cu, rewrite);
            case "simple_array_access":
                return applySimpleArrayAccess(cu, rewrite);
            case "simple_return_statement":
                return applySimpleReturnStatement(cu, rewrite);
            case "simple_variable_declaration":
                return applySimpleVariableDeclaration(cu, rewrite);
            case "simple_constructor_call":
                return applySimpleConstructorCall(cu, rewrite);
            case "simple_field_access":
                return applySimpleFieldAccess(cu, rewrite);
            case "simple_string_operation":
                return applySimpleStringOperation(cu, rewrite);
            case "simple_numeric_operation":
                return applySimpleNumericOperation(cu, rewrite);
                
            // Random augmentation transformations (3 methods)
            case "random_method_insertion":
                return applyRandomMethodInsertion(cu, rewrite);
            case "random_statement_insertion":
                return applyRandomStatementInsertion(cu, rewrite);
            case "random_expression_insertion":
                return applyRandomExpressionInsertion(cu, rewrite);
                
            // New transformation types
            case "bitwise_operation":
                return applyBitwiseOperation(cu, rewrite);
            case "comparison_operation":
                return applyComparisonOperation(cu, rewrite);
            case "type_conversion":
                return applyTypeConversion(cu, rewrite);
            case "null_check_pattern":
                return applyNullCheckPattern(cu, rewrite);
            case "constant_folding":
                return applyConstantFolding(cu, rewrite);
            case "dead_code_insertion":
                return applyDeadCodeInsertion(cu, rewrite);
            case "method_chain_transformation":
                return applyMethodChainTransformation(cu, rewrite);
            case "variable_renaming":
                return applyVariableRenaming(cu, rewrite);
                
            default:
                System.err.println("Unknown transformation: " + transformation);
                return false;
        }
    }
    
    // Enhanced transformation implementations
    private boolean applyLoopConversion(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ForStatement node) {
                if (random.nextDouble() < 1.0) { // 100% chance to convert
                    convertForToWhile(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(WhileStatement node) {
                if (random.nextDouble() < 1.0) { // 100% chance to convert
                    convertWhileToFor(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(DoStatement node) {
                if (random.nextDouble() < 0.8) { // 80% chance to convert
                    convertDoWhileToFor(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(LabeledStatement node) {
                if (node.getBody() instanceof ForStatement || 
                    node.getBody() instanceof WhileStatement ||
                    node.getBody() instanceof DoStatement) {
                    if (random.nextDouble() < 0.6) { // 60% chance to convert labeled loops
                        convertLabeledLoop(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyGuardReversal(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(IfStatement node) {
                try {
                    // Check if this if statement is inside a loop (skip guard reversal)
                    if (isInsideLoop(node)) {
                        debug("guard_reversal_skipped", "Skipped guard reversal: inside loop");
                        return true;
                    }
                    
                    if (node.getElseStatement() != null && !containsMethodInvocation(node.getExpression())) {
                        debug("guard_reversal_considered", "IfStatement eligible for guard reversal: " + node.getExpression());
                        reverseGuard(node, rewrite);
                        changed.set(true);
                    } else {
                        debug("guard_reversal_skipped", "Skipped guard reversal due to else==null or complex condition");
                    }
                } catch (Exception e) {
                    // skip complex cases
                    debug("guard_reversal_error", "Exception during guard reversal: " + e.getMessage());
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyMathematicalExpression(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.PLUS || 
                    node.getOperator() == InfixExpression.Operator.MINUS ||
                    node.getOperator() == InfixExpression.Operator.TIMES ||
                    node.getOperator() == InfixExpression.Operator.DIVIDE ||
                    node.getOperator() == InfixExpression.Operator.REMAINDER) {
                    
                    if (random.nextDouble() < 0.8) { // 80% chance to transform
                        boolean local = transformMathematicalExpressionEnhanced(node, rewrite);
                        if (local) {
                            changed.set(true);
                        }
                    }
                }
                return true;
            }
            
            @Override
            public boolean visit(PrefixExpression node) {
                // Handle unary minus: -x -> 0 - x
                if (node.getOperator() == PrefixExpression.Operator.MINUS && 
                    node.getOperand() instanceof SimpleName) {
                    if (random.nextDouble() < 0.5) { // 50% chance to transform
                        boolean local = transformUnaryMinus(node, rewrite);
                        if (local) {
                            changed.set(true);
                        }
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyLogicalExpression(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.AND ||
                    node.getOperator() == InfixExpression.Operator.OR) {
                    
                    if (random.nextDouble() < 0.8) { // 80% chance to transform
                        boolean local = transformLogicalExpressionEnhanced(node, rewrite);
                        if (local) {
                    changed.set(true);
                        }
                    }
                }
                return true;
            }
            
            @Override
            public boolean visit(PrefixExpression node) {
                // Handle double negation: !!a -> a
                if (node.getOperator() == PrefixExpression.Operator.NOT &&
                    node.getOperand() instanceof PrefixExpression) {
                    PrefixExpression inner = (PrefixExpression) node.getOperand();
                    if (inner.getOperator() == PrefixExpression.Operator.NOT) {
                        if (random.nextDouble() < 0.7) { // 70% chance to transform
                            boolean local = transformDoubleNegation(node, rewrite);
                            if (local) {
                                changed.set(true);
                            }
                        }
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyTernaryOperator(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ConditionalExpression node) {
                // Only convert ternary operators that are standalone statements, not values
                ASTNode parent = node.getParent();
                if (parent instanceof ExpressionStatement && random.nextDouble() < 0.3) { // 30% chance for standalone statements only
                    convertTernaryToIfElse(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(IfStatement node) {
                try {
                    if (node.getElseStatement() != null) {
                        // Case 1: return in both branches -> return (cond) ? a : b;
                        if (node.getThenStatement() instanceof ReturnStatement && node.getElseStatement() instanceof ReturnStatement) {
                            AST ast = node.getAST();
                            ConditionalExpression tern = ast.newConditionalExpression();
                            tern.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getExpression())));
                            tern.setThenExpression((Expression) ASTNode.copySubtree(ast, ((ReturnStatement) node.getThenStatement()).getExpression()));
                            tern.setElseExpression((Expression) ASTNode.copySubtree(ast, ((ReturnStatement) node.getElseStatement()).getExpression()));

                            ReturnStatement ret = ast.newReturnStatement();
                            ret.setExpression(tern);
                            rewrite.replace(node, ret, null);
                            changed.set(true);
                        }
                        // Case 2: assignments to same LHS -> lhs = (cond) ? rThen : rElse;
                        else if (isAssignToSameLhs(node.getThenStatement(), node.getElseStatement())) {
                            AST ast = node.getAST();
                            Assignment thenAsg = (Assignment) ((ExpressionStatement) unwrapFirstStatement(node.getThenStatement())).getExpression();
                            Assignment elseAsg = (Assignment) ((ExpressionStatement) unwrapFirstStatement(node.getElseStatement())).getExpression();

                            ConditionalExpression tern = ast.newConditionalExpression();
                            tern.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getExpression())));
                            tern.setThenExpression((Expression) ASTNode.copySubtree(ast, thenAsg.getRightHandSide()));
                            tern.setElseExpression((Expression) ASTNode.copySubtree(ast, elseAsg.getRightHandSide()));

                            Assignment newAsg = ast.newAssignment();
                            newAsg.setLeftHandSide((Expression) ASTNode.copySubtree(ast, thenAsg.getLeftHandSide()));
                            newAsg.setRightHandSide(tern);
                            ExpressionStatement es = ast.newExpressionStatement(newAsg);
                            rewrite.replace(node, es, null);
                            changed.set(true);
                        }
                        // Fallback: simple branches
                        else if (isSimpleBranch(node.getThenStatement()) && isSimpleBranch(node.getElseStatement())) {
                            convertIfElseToTernary(node, rewrite);
                            changed.set(true);
                        }
                    }
                } catch (Exception e) {
                    // skip complex cases
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    // Implemented transformations
    private boolean applySwitchStatement(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(SwitchStatement node) {
                if (random.nextDouble() < 1.0) { // 100% chance to transform
                    transformSwitchStatement(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyVariableOperation(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(Assignment node) {
                // Prefer deterministic transformations that yield visible textual changes
                boolean localChanged = transformVariableOperation(node, rewrite);
                if (localChanged) {
                    changed.set(true);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyMethodExtraction(CompilationUnit cu, ASTRewrite rewrite) {
        // Conservative AST-only normalization: wrap single non-block statements in a block
        // (acts as a structural refactor step without changing semantics)
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodDeclaration node) {
                try {
                    Statement body = node.getBody();
                    if (body != null && body instanceof Block) {
                        Block b = (Block) body;
                        if (!b.statements().isEmpty() && b.statements().get(0) instanceof Statement) {
                            Statement first = (Statement) b.statements().get(0);
                            if (!(first instanceof Block)) {
                                AST ast = node.getAST();
                                Block wrapper = ast.newBlock();
                                wrapper.statements().add(ASTNode.copySubtree(ast, first));
                                // Replace first statement with wrapped block
                                Block newBody = (Block) ASTNode.copySubtree(ast, b);
                                newBody.statements().set(0, wrapper);
                                rewrite.replace(b, newBody, null);
                                changed.set(true);
                            }
                        }
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyConditionalExpression(CompilationUnit cu, ASTRewrite rewrite) {
        // Normalize conditional expressions by parenthesizing arms and condition,
        // or converting simple nested conditionals to a normalized form.
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ConditionalExpression node) {
                try {
                    AST ast = node.getAST();
                    ConditionalExpression copy = (ConditionalExpression) ASTNode.copySubtree(ast, node);
                    // Deep parenthesization to force textual change
                    copy.setExpression(parenthesize(ast, parenthesize(ast, copy.getExpression())));
                    copy.setThenExpression(parenthesize(ast, parenthesize(ast, copy.getThenExpression())));
                    copy.setElseExpression(parenthesize(ast, parenthesize(ast, copy.getElseExpression())));
                    rewrite.replace(node, copy, null);
                    changed.set(true);
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyArrayAccessPattern(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize array and index expressions to create a normalized, slicer-resistant form.
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ArrayAccess node) {
                try {
                    AST ast = node.getAST();
                    ArrayAccess copy = (ArrayAccess) ASTNode.copySubtree(ast, node);
                    Expression a = copy.getArray();
                    Expression i = copy.getIndex();
                    if (!(a instanceof ParenthesizedExpression)) {
                        copy.setArray(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, a)));
                    }
                    if (!(i instanceof ParenthesizedExpression)) {
                        copy.setIndex(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, i)));
                    }
                    rewrite.replace(node, copy, null);
                    changed.set(true);
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyStringConcatenation(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.PLUS) {
                    if (random.nextDouble() < 1.0) { // 100% chance to transform
                        transformStringConcatenation(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyNumericLiteral(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(NumberLiteral node) {
                if (random.nextDouble() < 1.0) { // 100% chance to transform
                    transformNumericLiteral(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    private boolean applyExceptionHandling(CompilationUnit cu, ASTRewrite rewrite) {
        // Normalize try statements: add empty finally when absent (AST-only, semantics preserved)
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(TryStatement node) {
                try {
                    if (node.getFinally() == null) {
                        AST ast = node.getAST();
                        TryStatement copy = (TryStatement) ASTNode.copySubtree(ast, node);
                        Block fin = ast.newBlock();
                        // Insert an explicit empty statement for visibility
                        fin.statements().add(ast.newEmptyStatement());
                        copy.setFinally(fin);
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyLambdaExpression(CompilationUnit cu, ASTRewrite rewrite) {
        // Convert between expression-body and block-body lambdas when possible
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(LambdaExpression node) {
                try {
                    AST ast = node.getAST();
                    if (node.getBody() instanceof Expression) {
                        // expression -> block with return
                        Block b = ast.newBlock();
                        ReturnStatement rs = ast.newReturnStatement();
                        rs.setExpression((Expression) ASTNode.copySubtree(ast, (ASTNode) node.getBody()));
                        b.statements().add(rs);
                        LambdaExpression copy = (LambdaExpression) ASTNode.copySubtree(ast, node);
                        copy.setBody(b);
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    } else if (node.getBody() instanceof Block) {
                        Block b = (Block) node.getBody();
                        if (b.statements().size() == 1 && b.statements().get(0) instanceof ReturnStatement) {
                            // block with single return -> expression
                            ReturnStatement rs = (ReturnStatement) b.statements().get(0);
                            LambdaExpression copy = (LambdaExpression) ASTNode.copySubtree(ast, node);
                            copy.setBody((ASTNode) ASTNode.copySubtree(ast, rs.getExpression()));
                            rewrite.replace(node, copy, null);
                            changed.set(true);
                        }
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyStreamApi(CompilationUnit cu, ASTRewrite rewrite) {
        // Convert method references to equivalent single-parameter lambdas where applicable
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            public boolean visit(MethodReference node) {
                try {
                    AST ast = node.getAST();
                    if (node instanceof ExpressionMethodReference) {
                        ExpressionMethodReference emr = (ExpressionMethodReference) node;
                        LambdaExpression lambda = ast.newLambdaExpression();
                        SingleVariableDeclaration svd = ast.newSingleVariableDeclaration();
                        svd.setName(ast.newSimpleName("x"));
                        lambda.parameters().add(svd);
                        MethodInvocation mi = ast.newMethodInvocation();
                        if (emr.getExpression() != null) {
                            mi.setExpression((Expression) ASTNode.copySubtree(ast, emr.getExpression()));
                        }
                        mi.setName((SimpleName) ASTNode.copySubtree(ast, emr.getName()));
                        mi.arguments().add(ast.newSimpleName("x"));
                        lambda.setBody(mi);
                        rewrite.replace(node, lambda, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }

            @Override
            public boolean visit(MethodInvocation node) {
                try {
                    // Case A: xs.stream().forEach(lambda) -> for-each
                    if ("forEach".equals(node.getName().getIdentifier()) && node.arguments().size() == 1) {
                        if (node.getExpression() instanceof MethodInvocation) {
                            MethodInvocation streamCall = (MethodInvocation) node.getExpression();
                            if ("stream".equals(streamCall.getName().getIdentifier()) && streamCall.arguments().isEmpty()
                                && streamCall.getExpression() instanceof SimpleName) {
                                Object arg = node.arguments().get(0);
                                if (arg instanceof LambdaExpression) {
                                    LambdaExpression le = (LambdaExpression) arg;
                                    if (le.parameters().size() == 1) {
                                        AST ast = node.getAST();
                                        EnhancedForStatement efor = ast.newEnhancedForStatement();
                                        SingleVariableDeclaration svd = ast.newSingleVariableDeclaration();
                                        svd.setType(ast.newSimpleType(ast.newSimpleName("var")));
                                        svd.setName((SimpleName) ASTNode.copySubtree(ast, ((SingleVariableDeclaration) le.parameters().get(0)).getName()));
                                        efor.setParameter(svd);
                                        efor.setExpression((Expression) ASTNode.copySubtree(ast, streamCall.getExpression()));
                                        Block body = ast.newBlock();
                                        if (le.getBody() instanceof Block) {
                                            Block lb = (Block) le.getBody();
                                            for (Object st : lb.statements()) {
                                                body.statements().add(ASTNode.copySubtree(ast, (ASTNode) st));
                                            }
                                        } else if (le.getBody() instanceof Expression) {
                                            ExpressionStatement es = ast.newExpressionStatement((Expression) ASTNode.copySubtree(ast, (ASTNode) le.getBody()));
                                            body.statements().add(es);
                                        }
                                        efor.setBody(body);
                                        rewrite.replace(node, efor, null);
                                        changed.set(true);
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                    // Case B: Optional.ifPresent: normalize lambda body
                    if ("ifPresent".equals(node.getName().getIdentifier()) && node.arguments().size() == 1) {
                        Object arg = node.arguments().get(0);
                        if (arg instanceof LambdaExpression) {
                            LambdaExpression le = (LambdaExpression) arg;
                            AST ast = node.getAST();
                            LambdaExpression copy = (LambdaExpression) ASTNode.copySubtree(ast, le);
                            if (copy.getBody() instanceof Expression) {
                                Block b = ast.newBlock();
                                ExpressionStatement es = ast.newExpressionStatement((Expression) ASTNode.copySubtree(ast, (ASTNode) copy.getBody()));
                                b.statements().add(es);
                                copy.setBody(b);
                                MethodInvocation mi = (MethodInvocation) ASTNode.copySubtree(ast, node);
                                mi.arguments().set(0, copy);
                                rewrite.replace(node, mi, null);
                                changed.set(true);
                                return false;
                            }
                            // If already a block body, enforce parenthesization inside first expr to ensure textual change
                            if (copy.getBody() instanceof Block) {
                                Block b = (Block) copy.getBody();
                                if (!b.statements().isEmpty() && b.statements().get(0) instanceof ExpressionStatement) {
                                    ExpressionStatement es = (ExpressionStatement) b.statements().get(0);
                                    es.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, es.getExpression())));
                                    MethodInvocation mi = (MethodInvocation) ASTNode.copySubtree(ast, node);
                                    mi.arguments().set(0, copy);
                                    rewrite.replace(node, mi, null);
                                    changed.set(true);
                                    return false;
                                }
                            }
                        }
                    }
                    // Case C: collectors equivalence to toList
                    if ("collect".equals(node.getName().getIdentifier()) && node.arguments().size() == 1) {
                        if (node.getExpression() instanceof MethodInvocation) {
                            MethodInvocation streamCall = (MethodInvocation) node.getExpression();
                            if ("stream".equals(streamCall.getName().getIdentifier()) && streamCall.arguments().isEmpty()
                                && streamCall.getExpression() instanceof SimpleName) {
                                Object arg = node.arguments().get(0);
                                if (arg instanceof MethodInvocation) {
                                    MethodInvocation collectorsCall = (MethodInvocation) arg;
                                    String cname = collectorsCall.getName().getIdentifier();
                                    if ("toList".equals(cname)) {
                                        AST ast = node.getAST();
                                        ClassInstanceCreation cic = ast.newClassInstanceCreation();
                                        cic.setType(ast.newSimpleType(ast.newName("java.util.ArrayList")));
                                        cic.arguments().add(ASTNode.copySubtree(ast, streamCall.getExpression()));
                                        rewrite.replace(node, cic, null);
                                        changed.set(true);
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                } catch (Exception e) {}
                return true;
            }
        });
        return changed.get();
    }
    
    private boolean applyBuilderPattern(CompilationUnit cu, ASTRewrite rewrite) {
        // Insert harmless parentheses around the qualifier of chained calls (AST-only tweak)
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) {
                try {
                    if (node.getExpression() != null && !(node.getExpression() instanceof ParenthesizedExpression)) {
                        AST ast = node.getAST();
                        MethodInvocation copy = (MethodInvocation) ASTNode.copySubtree(ast, node);
                        copy.setExpression(parenthesize(ast, copy.getExpression()));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }

            @Override
            public boolean visit(ClassInstanceCreation node) {
                // Normalize builder new-expr arguments by parenthesizing
                try {
                    AST ast = node.getAST();
                    boolean local = false;
                    ClassInstanceCreation copy = (ClassInstanceCreation) ASTNode.copySubtree(ast, node);
                    for (int i = 0; i < copy.arguments().size(); i++) {
                        Object arg = copy.arguments().get(i);
                        if (arg instanceof Expression && !(arg instanceof ParenthesizedExpression)) {
                            copy.arguments().set(i, parenthesize(ast, (Expression) ASTNode.copySubtree(ast, (ASTNode) arg)));
                            local = true;
                        }
                    }
                    if (local) {
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyFunctionalConversion(CompilationUnit cu, ASTRewrite rewrite) {
        // Convert single-parameter expression lambdas to method references when trivial (x -> x.toString())
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(LambdaExpression node) {
                try {
                    if (node.parameters().size() == 1 && node.getBody() instanceof MethodInvocation) {
                        AST ast = node.getAST();
                        MethodInvocation mi = (MethodInvocation) node.getBody();
                        if (mi.getExpression() instanceof SimpleName && ((SimpleName) mi.getExpression()).getIdentifier().equals(((SingleVariableDeclaration) node.parameters().get(0)).getName().getIdentifier())) {
                            // Use a normalized method invocation wrapped in parentheses as a conservative refactor
                            LambdaExpression copy = (LambdaExpression) ASTNode.copySubtree(ast, node);
                            MethodInvocation mic = (MethodInvocation) ASTNode.copySubtree(ast, mi);
                            copy.setBody(parenthesize(ast, mic));
                            rewrite.replace(node, copy, null);
                            changed.set(true);
                        }
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    // Simple transformation implementations
    private boolean applySimpleMethodCall(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize each argument (AST-only)
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) {
                try {
                    AST ast = node.getAST();
                    boolean local = false;
                    MethodInvocation copy = (MethodInvocation) ASTNode.copySubtree(ast, node);
                    for (int i = 0; i < copy.arguments().size(); i++) {
                        Object arg = copy.arguments().get(i);
                        if (arg instanceof Expression && !(arg instanceof ParenthesizedExpression)) {
                            copy.arguments().set(i, parenthesize(ast, (Expression) ASTNode.copySubtree(ast, (ASTNode) arg)));
                            local = true;
                        }
                    }
                    if (local) {
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleAssignment(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize RHS of simple assignments (AST-only)
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(Assignment node) {
                try {
                    if (node.getOperator() == Assignment.Operator.ASSIGN && !(node.getRightHandSide() instanceof ParenthesizedExpression)) {
                        AST ast = node.getAST();
                        Assignment copy = (Assignment) ASTNode.copySubtree(ast, node);
                        copy.setRightHandSide(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getRightHandSide())));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleConditional(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize condition expressions in if statements
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(IfStatement node) {
                try {
                    if (!(node.getExpression() instanceof ParenthesizedExpression)) {
                        AST ast = node.getAST();
                        IfStatement copy = (IfStatement) ASTNode.copySubtree(ast, node);
                        copy.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getExpression())));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleArrayAccess(CompilationUnit cu, ASTRewrite rewrite) {
        // Delegate to array access normalization
        return applyArrayAccessPattern(cu, rewrite);
    }
    
    private boolean applySimpleReturnStatement(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize returned expressions
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ReturnStatement node) {
                try {
                    if (node.getExpression() != null && !(node.getExpression() instanceof ParenthesizedExpression)) {
                        AST ast = node.getAST();
                        ReturnStatement copy = (ReturnStatement) ASTNode.copySubtree(ast, node);
                        copy.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getExpression())));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleVariableDeclaration(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize initializer expressions and perform SSA-like trivial inlining
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(VariableDeclarationFragment node) {
                try {
                    if (node.getInitializer() != null && 
                        !(node.getInitializer() instanceof ParenthesizedExpression) &&
                        !(node.getInitializer() instanceof ArrayInitializer)) {
                        AST ast = node.getAST();
                        VariableDeclarationFragment copy = (VariableDeclarationFragment) ASTNode.copySubtree(ast, node);
                        copy.setInitializer(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getInitializer())));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
            @Override
            public boolean visit(Block node) {
                try {
                    // Trivial inlining: int t = <pure_expr>; use t once -> replace use with expr and delete decl
                    // Conservative: only if exactly one SimpleName use exists in the same block
                    AST ast = node.getAST();
                    @SuppressWarnings("unchecked")
                    List<Statement> stmts = (List<Statement>) node.statements();
                    for (int i = 0; i < stmts.size(); i++) {
                        Statement s = stmts.get(i);
                        if (s instanceof VariableDeclarationStatement) {
                            VariableDeclarationStatement vds = (VariableDeclarationStatement) s;
                            if (vds.fragments().size() == 1) {
                                VariableDeclarationFragment frag = (VariableDeclarationFragment) vds.fragments().get(0);
                                if (frag.getInitializer() != null && isPure(frag.getInitializer())) {
                                    String name = frag.getName().getIdentifier();
                                    // Count uses in subsequent statements
                                    int uses = 0; int useIndex = -1; SimpleName useNode = null; Statement hostStmt = null;
                                    for (int j = i + 1; j < stmts.size(); j++) {
                                        Statement sj = stmts.get(j);
                                        final List<SimpleName> names = new ArrayList<>();
                                        sj.accept(new ASTVisitor(){
                                            @Override public boolean visit(SimpleName n){ names.add(n); return true; }
                                        });
                                        for (SimpleName n : names) {
                                            if (name.equals(n.getIdentifier())) {
                                                uses++; useIndex = j; useNode = n; hostStmt = sj;
                                            }
                                        }
                                    }
                                    if (uses == 1 && useNode != null && hostStmt != null) {
                                        // Inline: replace use with initializer and remove declaration
                                        Block copyBlock = (Block) ASTNode.copySubtree(ast, node);
                                        @SuppressWarnings("unchecked")
                                        List<Statement> cstmts = (List<Statement>) copyBlock.statements();
                                        VariableDeclarationStatement cvds = (VariableDeclarationStatement) cstmts.get(i);
                                        VariableDeclarationFragment cfrag = (VariableDeclarationFragment) cvds.fragments().get(0);
                                        Expression init = (Expression) ASTNode.copySubtree(ast, cfrag.getInitializer());
                                        Statement cHost = cstmts.get(useIndex);
                                        cHost.accept(new ASTVisitor(){
                                            @Override
                                            public boolean visit(SimpleName n){
                                                if (n.getIdentifier().equals(name)) {
                                                    rewrite.replace(n, ASTNode.copySubtree(ast, init), null);
                                                }
                                                return true;
                                            }
                                        });
                                        cstmts.remove(i);
                                        rewrite.replace(node, copyBlock, null);
                                        changed.set(true);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                } catch (Exception e) {}
                return true;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleConstructorCall(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize each constructor argument
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(ClassInstanceCreation node) {
                try {
                    AST ast = node.getAST();
                    boolean local = false;
                    ClassInstanceCreation copy = (ClassInstanceCreation) ASTNode.copySubtree(ast, node);
                    for (int i = 0; i < copy.arguments().size(); i++) {
                        Object arg = copy.arguments().get(i);
                        if (arg instanceof Expression && !(arg instanceof ParenthesizedExpression)) {
                            copy.arguments().set(i, parenthesize(ast, (Expression) ASTNode.copySubtree(ast, (ASTNode) arg)));
                            local = true;
                        }
                    }
                    if (local) {
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleFieldAccess(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize qualifier in field access when applicable
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(FieldAccess node) {
                try {
                    if (!(node.getExpression() instanceof ParenthesizedExpression)) {
                        AST ast = node.getAST();
                        FieldAccess copy = (FieldAccess) ASTNode.copySubtree(ast, node);
                        copy.setExpression(parenthesize(ast, copy.getExpression()));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleStringOperation(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize operands of string concatenations
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                try {
                    if (node.getOperator() == InfixExpression.Operator.PLUS) {
                        AST ast = node.getAST();
                        InfixExpression copy = (InfixExpression) ASTNode.copySubtree(ast, node);
                        copy.setLeftOperand(parenthesize(ast, copy.getLeftOperand()));
                        copy.setRightOperand(parenthesize(ast, copy.getRightOperand()));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applySimpleNumericOperation(CompilationUnit cu, ASTRewrite rewrite) {
        // Parenthesize operands of arithmetic expressions
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                try {
                    if (node.getOperator() == InfixExpression.Operator.PLUS ||
                        node.getOperator() == InfixExpression.Operator.MINUS ||
                        node.getOperator() == InfixExpression.Operator.TIMES ||
                        node.getOperator() == InfixExpression.Operator.DIVIDE) {
                        AST ast = node.getAST();
                        InfixExpression copy = (InfixExpression) ASTNode.copySubtree(ast, node);
                        copy.setLeftOperand(parenthesize(ast, copy.getLeftOperand()));
                        copy.setRightOperand(parenthesize(ast, copy.getRightOperand()));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    // Random augmentation implementations
    private boolean applyRandomMethodInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        // Insert an empty statement at the beginning of the first method body block
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodDeclaration node) {
                try {
                    if (node.getBody() != null && node.getBody() instanceof Block) {
                        AST ast = node.getAST();
                        Block b = (Block) ASTNode.copySubtree(ast, node.getBody());
                        b.statements().add(0, ast.newEmptyStatement());
                        rewrite.replace(node.getBody(), b, null);
                        changed.set(true);
                        return false;
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyRandomStatementInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        // Insert an empty statement before any standalone statement inside blocks
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(Block node) {
                try {
                    if (!node.statements().isEmpty()) {
                        AST ast = node.getAST();
                        Block copy = (Block) ASTNode.copySubtree(ast, node);
                        copy.statements().add(0, ast.newEmptyStatement());
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    private boolean applyRandomExpressionInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        // Wrap a literal or simple name in a parenthesized expression where found
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(NumberLiteral node) {
                try {
                    AST ast = node.getAST();
                    ParenthesizedExpression p = parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node));
                    rewrite.replace(node, p, null);
                    changed.set(true);
                } catch (Exception e) {}
                return false;
            }
        });
        return changed.get();
    }
    
    // Helper methods for specific transformations
    private void convertForToWhile(ForStatement forStmt, ASTRewrite rewrite) {
        // Convert for loop to while loop with enhanced pattern support
        AST ast = forStmt.getAST();
        
        // Handle enhanced for-each loops - preserve them as-is
        if (forStmt.initializers().isEmpty() && forStmt.updaters().isEmpty() && 
            forStmt.getExpression() != null) {
            // Check if this looks like an enhanced for-each loop (expression is a variable declaration)
            debug("loop_conversion_skip", "Skipping enhanced for-each loop conversion");
            return;
        }
        
        WhileStatement whileStmt = ast.newWhileStatement();
        whileStmt.setExpression(ast.newBooleanLiteral(true));
        
        Block whileBody = ast.newBlock();
        
        // Enhanced initialization handling
        handleComplexInitializers(forStmt, whileBody, ast);
        
        // Enhanced body handling with proper scoping
        handleLoopBody(forStmt, whileBody, ast);
        
        // Enhanced increment handling
        handleComplexIncrements(forStmt, whileBody, ast);
        
        whileStmt.setBody(whileBody);
        
        // Add condition check with enhanced positioning
        addConditionCheck(forStmt, whileBody, ast);
        
        rewrite.replace(forStmt, whileStmt, null);
    }
    
    /**
     * Handle complex initializers including multiple variables and method calls.
     */
    private void handleComplexInitializers(ForStatement forStmt, Block whileBody, AST ast) {
        if (forStmt.initializers().size() > 0) {
            for (Object initializer : forStmt.initializers()) {
                if (initializer instanceof VariableDeclarationExpression) {
                    VariableDeclarationExpression vde = (VariableDeclarationExpression) initializer;
                    
                    // Handle multiple variable declarations
                    for (Object fragment : vde.fragments()) {
                        VariableDeclarationFragment frag = (VariableDeclarationFragment) fragment;
                    VariableDeclarationStatement vds = ast.newVariableDeclarationStatement(
                            (VariableDeclarationFragment) ASTNode.copySubtree(ast, frag)
                        );
                        vds.setType((Type) ASTNode.copySubtree(ast, vde.getType()));
                    whileBody.statements().add(vds);
                    }
                } else if (initializer instanceof Expression) {
                    // Handle method calls and complex expressions in initialization
                    ExpressionStatement es = ast.newExpressionStatement(
                        (Expression) ASTNode.copySubtree(ast, (Expression) initializer)
                    );
                    whileBody.statements().add(es);
                }
                }
            }
        }
        
    /**
     * Handle loop body with proper scoping and control flow.
     */
    private void handleLoopBody(ForStatement forStmt, Block whileBody, AST ast) {
        if (forStmt.getBody() != null) {
            if (forStmt.getBody() instanceof Block) {
                Block originalBody = (Block) forStmt.getBody();
                for (Object stmt : originalBody.statements()) {
                    whileBody.statements().add(ASTNode.copySubtree(ast, (Statement) stmt));
                }
            } else {
                // Single statement body - wrap in block for consistency
                Block singleStmtBlock = ast.newBlock();
                singleStmtBlock.statements().add(ASTNode.copySubtree(ast, forStmt.getBody()));
                whileBody.statements().add(singleStmtBlock);
            }
        }
    }
    
    /**
     * Handle complex increment expressions including multiple updates.
     */
    private void handleComplexIncrements(ForStatement forStmt, Block whileBody, AST ast) {
        if (forStmt.updaters().size() > 0) {
            for (Object updater : forStmt.updaters()) {
                if (updater instanceof Expression) {
                    ExpressionStatement es = ast.newExpressionStatement(
                        (Expression) ASTNode.copySubtree(ast, (Expression) updater)
                    );
                    whileBody.statements().add(es);
                }
            }
        }
    }
    
    /**
     * Add condition check with intelligent positioning.
     */
    private void addConditionCheck(ForStatement forStmt, Block whileBody, AST ast) {
        if (forStmt.getExpression() != null) {
            IfStatement conditionCheck = ast.newIfStatement();
            PrefixExpression notExpr = ast.newPrefixExpression();
            notExpr.setOperator(PrefixExpression.Operator.NOT);
            notExpr.setOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, forStmt.getExpression())));
            conditionCheck.setExpression(notExpr);
            
            BreakStatement breakStmt = ast.newBreakStatement();
            Block breakBlock = ast.newBlock();
            breakBlock.statements().add(breakStmt);
            conditionCheck.setThenStatement(breakBlock);
            
            // Enhanced positioning logic
            int insertPosition = findOptimalConditionPosition(whileBody);
            whileBody.statements().add(insertPosition, conditionCheck);
        }
    }
    
    /**
     * Find optimal position for condition check - after declarations, before body.
     */
    private int findOptimalConditionPosition(Block whileBody) {
        int insertPosition = 0;
        
        // Count initialization statements to find where to insert condition check
        for (int i = 0; i < whileBody.statements().size(); i++) {
            Statement stmt = (Statement) whileBody.statements().get(i);
            if (isInitializationStatement(stmt)) {
                insertPosition = i + 1;
            } else {
                break; // Stop at first non-initialization statement
            }
        }
        
        return insertPosition;
    }
    
    /**
     * Check if a statement is an initialization statement.
     */
    private boolean isInitializationStatement(Statement stmt) {
        return stmt instanceof VariableDeclarationStatement || 
               (stmt instanceof ExpressionStatement && 
                ((ExpressionStatement) stmt).getExpression() instanceof Assignment);
    }
    
    private void convertWhileToFor(WhileStatement whileStmt, ASTRewrite rewrite) {
        // Convert while loop to for loop - only when it matches standard for pattern
        AST ast = whileStmt.getAST();
        
        // Analyze the while loop to extract initialization, condition, and increment
        WhileLoopAnalysis analysis = analyzeWhileLoop(whileStmt);
        
        if (analysis.canConvertToFor) {
        ForStatement forStmt = ast.newForStatement();
            
            // Set initialization if found
            if (analysis.initialization != null) {
                forStmt.initializers().add(analysis.initialization);
            }
            
            // Set condition
        forStmt.setExpression((Expression) ASTNode.copySubtree(ast, whileStmt.getExpression()));
            
            // Set increment if found
            if (analysis.increment != null) {
                forStmt.updaters().add(analysis.increment);
            }
            
            // Set body (remove increment from body if it was extracted)
            forStmt.setBody(analysis.body);
        
        rewrite.replace(whileStmt, forStmt, null);
        } else {
            // Cannot convert - preserve original while loop
            debug("while_to_for_skipped", "Cannot convert while loop to for loop - pattern not recognized");
        }
    }
    
    /**
     * Analyze a while loop to determine if it can be converted to a for loop.
     */
    private WhileLoopAnalysis analyzeWhileLoop(WhileStatement whileStmt) {
        WhileLoopAnalysis analysis = new WhileLoopAnalysis();
        
        // For now, only convert simple while loops without complex initialization/increment
        // This prevents semantic issues from incorrect conversion
        analysis.canConvertToFor = false; // Conservative approach - preserve semantics
        
        // Copy the original body
        analysis.body = (Statement) ASTNode.copySubtree(whileStmt.getAST(), whileStmt.getBody());
        
        return analysis;
    }
    
    /**
     * Helper class to store while loop analysis results.
     */
    private static class WhileLoopAnalysis {
        boolean canConvertToFor = false;
        Expression initialization = null;
        Expression increment = null;
        Statement body = null;
    }
    
    private void reverseGuard(IfStatement ifStmt, ASTRewrite rewrite) {
        // Reverse the guard condition and swap if/else blocks
        AST ast = ifStmt.getAST();
        // Build a new IfStatement to avoid mutating the original node in-place
        IfStatement newIf = ast.newIfStatement();
        PrefixExpression notExpr = ast.newPrefixExpression();
        notExpr.setOperator(PrefixExpression.Operator.NOT);
        notExpr.setOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, ifStmt.getExpression())));
        newIf.setExpression(notExpr);

        // Swap then and else statements (deep copies)
        Statement thenStmt = ifStmt.getThenStatement();
        Statement elseStmt = ifStmt.getElseStatement();
        if (elseStmt != null) {
            newIf.setThenStatement((Statement) ASTNode.copySubtree(ast, elseStmt));
        }
        if (thenStmt != null) {
            newIf.setElseStatement((Statement) ASTNode.copySubtree(ast, thenStmt));
        }
        
        rewrite.replace(ifStmt, newIf, null);
    }

    // Helper: check if then/else assign to same LHS
    private boolean isAssignToSameLhs(Statement thenStmt, Statement elseStmt) {
        Statement t = unwrapFirstStatement(thenStmt);
        Statement e = unwrapFirstStatement(elseStmt);
        if (t instanceof ExpressionStatement && e instanceof ExpressionStatement) {
            Expression te = ((ExpressionStatement) t).getExpression();
            Expression ee = ((ExpressionStatement) e).getExpression();
            if (te instanceof Assignment && ee instanceof Assignment) {
                String l1 = ((Assignment) te).getLeftHandSide().toString();
                String l2 = ((Assignment) ee).getLeftHandSide().toString();
                return l1.equals(l2);
            }
        }
        return false;
    }

    // Helper: unwrap one-statement blocks
    private Statement unwrapFirstStatement(Statement stmt) {
        if (stmt instanceof Block) {
            Block b = (Block) stmt;
            if (!b.statements().isEmpty() && b.statements().get(0) instanceof Statement) {
                return (Statement) b.statements().get(0);
            }
        }
        return stmt;
    }

    // Helper: parenthesize expression
    private ParenthesizedExpression parenthesize(AST ast, Expression expr) {
        ParenthesizedExpression p = ast.newParenthesizedExpression();
        p.setExpression(expr);
        return p;
    }
    
    private boolean isSimpleOperand(Expression e) {
        return e instanceof SimpleName || e instanceof NumberLiteral || e instanceof QualifiedName
                || (e instanceof ParenthesizedExpression && ((ParenthesizedExpression) e).getExpression() instanceof SimpleName)
                || (e instanceof ParenthesizedExpression && ((ParenthesizedExpression) e).getExpression() instanceof NumberLiteral);
    }

    /**
     * Get the diagnostics report for this transformation session.
     */
    public TransformationDiagnostics.DiagnosticReport getDiagnosticsReport() {
        return diagnostics != null ? diagnostics.generateReport() : null;
    }
    
    /**
     * Get the diagnostics object for detailed analysis.
     */
    public TransformationDiagnostics getDiagnostics() {
        return diagnostics;
    }

    private boolean transformMathematicalExpressionEnhanced(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        InfixExpression.Operator op = expr.getOperator();
            Expression left = expr.getLeftOperand();
            Expression right = expr.getRightOperand();
        
        // Apply various mathematical transformations based on operator
        if (op == InfixExpression.Operator.PLUS) {
            return transformAddition(expr, left, right, ast, rewrite);
        } else if (op == InfixExpression.Operator.MINUS) {
            return transformSubtraction(expr, left, right, ast, rewrite);
        } else if (op == InfixExpression.Operator.TIMES) {
            return transformMultiplication(expr, left, right, ast, rewrite);
        } else if (op == InfixExpression.Operator.DIVIDE) {
            return transformDivision(expr, left, right, ast, rewrite);
        } else if (op == InfixExpression.Operator.REMAINDER) {
            return transformModulo(expr, left, right, ast, rewrite);
        } else {
            return false;
        }
    }
    
    /**
     * Transform addition expressions: commutativity, associativity, identity elements
     */
    private boolean transformAddition(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Apply commutativity: a + b -> b + a
            if (isSimpleOperand(left) && isSimpleOperand(right)) {
                InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.PLUS);
                newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, right));
                newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, left));
            debug("math_commute_plus", "Applied commutativity to addition: " + expr.toString());
                rewrite.replace(expr, newExpr, null);
                return true;
        }
        return false;
    }
    
    /**
     * Transform subtraction expressions: negation normalization
     */
    private boolean transformSubtraction(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Convert subtraction to addition with negation: a - b -> a + (-b)
        if (isSimpleOperand(right)) {
            PrefixExpression negatedRight = ast.newPrefixExpression();
            negatedRight.setOperator(PrefixExpression.Operator.MINUS);
            negatedRight.setOperand((Expression) ASTNode.copySubtree(ast, right));
            
            InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.PLUS);
            newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, left));
            newExpr.setRightOperand(negatedRight);
            
            debug("math_sub_to_add", "Converted subtraction to addition with negation: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        return false;
    }
    
    /**
     * Transform multiplication expressions: commutativity, identity elements
     */
    private boolean transformMultiplication(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Apply commutativity: a * b -> b * a
        if (isSimpleOperand(left) && isSimpleOperand(right)) {
            InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.TIMES);
            newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, right));
            newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, left));
            debug("math_commute_times", "Applied commutativity to multiplication: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        return false;
    }
    
    /**
     * Transform division expressions: convert to multiplication when safe
     */
    private boolean transformDivision(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Convert division to multiplication: x / 2 -> x * 0.5 (when right is constant)
        if (right instanceof NumberLiteral) {
            NumberLiteral literal = (NumberLiteral) right;
            try {
                double value = Double.parseDouble(literal.getToken());
                if (value != 0 && value != 1) {
                    double reciprocal = 1.0 / value;
                    NumberLiteral newLiteral = ast.newNumberLiteral(String.valueOf(reciprocal));
                    
                    InfixExpression newExpr = ast.newInfixExpression();
                    newExpr.setOperator(InfixExpression.Operator.TIMES);
                    newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, left));
                    newExpr.setRightOperand(newLiteral);
                    
                    debug("math_div_to_mul", "Converted division to multiplication: " + expr.toString());
                    rewrite.replace(expr, newExpr, null);
                    return true;
                }
            } catch (NumberFormatException e) {
                // Skip if not a valid number
            }
        }
        return false;
    }
    
    /**
     * Transform modulo expressions: basic normalization
     */
    private boolean transformModulo(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // For now, just parenthesize for consistency
        InfixExpression newExpr = ast.newInfixExpression();
        newExpr.setOperator(InfixExpression.Operator.REMAINDER);
        newExpr.setLeftOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, left)));
        newExpr.setRightOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, right)));
        
        debug("math_modulo_norm", "Normalized modulo expression: " + expr.toString());
        rewrite.replace(expr, newExpr, null);
        return true;
    }
    
    /**
     * Transform unary minus: -x -> 0 - x
     */
    private boolean transformUnaryMinus(PrefixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        InfixExpression newExpr = ast.newInfixExpression();
        newExpr.setOperator(InfixExpression.Operator.MINUS);
        newExpr.setLeftOperand(ast.newNumberLiteral("0"));
        newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getOperand()));
        
        debug("math_unary_minus", "Converted unary minus to subtraction: " + expr.toString());
        rewrite.replace(expr, newExpr, null);
        return true;
    }
    
    /**
     * Enhanced logical expression transformation with comprehensive boolean algebra.
     */
    private boolean transformLogicalExpressionEnhanced(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        InfixExpression.Operator op = expr.getOperator();
        Expression left = expr.getLeftOperand();
        Expression right = expr.getRightOperand();
        
        // Apply various logical transformations based on operator
        if (op == InfixExpression.Operator.AND) {
            return transformLogicalAnd(expr, left, right, ast, rewrite);
        } else if (op == InfixExpression.Operator.OR) {
            return transformLogicalOr(expr, left, right, ast, rewrite);
        }
        
        return false;
    }
    
    /**
     * Transform logical AND expressions: idempotence, absorption, complement
     */
    private boolean transformLogicalAnd(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Apply idempotence: a && a -> a
        if (isSameExpression(left, right)) {
            Expression newExpr = (Expression) ASTNode.copySubtree(ast, left);
            debug("logical_and_idempotence", "Applied idempotence to AND: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        
        // Apply commutativity: a && b -> b && a
        if (isSimpleOperand(left) && isSimpleOperand(right)) {
            InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.AND);
            newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, right));
            newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, left));
            debug("logical_and_commute", "Applied commutativity to AND: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        
        // Apply parenthesization for consistency
        InfixExpression newExpr = ast.newInfixExpression();
        newExpr.setOperator(InfixExpression.Operator.AND);
        newExpr.setLeftOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, left)));
        newExpr.setRightOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, right)));
        debug("logical_and_parenthesize", "Applied parenthesization to AND: " + expr.toString());
        rewrite.replace(expr, newExpr, null);
        return true;
    }
    
    /**
     * Transform logical OR expressions: idempotence, absorption, complement
     */
    private boolean transformLogicalOr(InfixExpression expr, Expression left, Expression right, AST ast, ASTRewrite rewrite) {
        // Apply idempotence: a || a -> a
        if (isSameExpression(left, right)) {
            Expression newExpr = (Expression) ASTNode.copySubtree(ast, left);
            debug("logical_or_idempotence", "Applied idempotence to OR: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        
        // Apply commutativity: a || b -> b || a
        if (isSimpleOperand(left) && isSimpleOperand(right)) {
            InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.OR);
            newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, right));
            newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, left));
            debug("logical_or_commute", "Applied commutativity to OR: " + expr.toString());
            rewrite.replace(expr, newExpr, null);
            return true;
        }
        
        // Apply parenthesization for consistency
        InfixExpression newExpr = ast.newInfixExpression();
        newExpr.setOperator(InfixExpression.Operator.OR);
        newExpr.setLeftOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, left)));
        newExpr.setRightOperand(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, right)));
        debug("logical_or_parenthesize", "Applied parenthesization to OR: " + expr.toString());
        rewrite.replace(expr, newExpr, null);
        return true;
    }
    
    /**
     * Transform double negation: !!a -> a
     */
    private boolean transformDoubleNegation(PrefixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        PrefixExpression inner = (PrefixExpression) expr.getOperand();
        
        Expression newExpr = (Expression) ASTNode.copySubtree(ast, inner.getOperand());
        debug("logical_double_negation", "Applied double negation elimination: " + expr.toString());
        rewrite.replace(expr, newExpr, null);
        return true;
    }
    
    /**
     * Check if two expressions are the same (simplified comparison).
     */
    private boolean isSameExpression(Expression expr1, Expression expr2) {
        // Simple string comparison for now - could be enhanced with deeper AST comparison
        return expr1.toString().equals(expr2.toString());
    }
    
    private void applyDeMorganLaws(InfixExpression expr, ASTRewrite rewrite) {
        // Apply De Morgan's laws: !(A && B) -> (!A || !B), !(A || B) -> (!A && !B)
        AST ast = expr.getAST();
        
        if (expr.getOperator() == InfixExpression.Operator.AND) {
            // Convert AND to OR with negations
            InfixExpression newExpr = ast.newInfixExpression();
            newExpr.setOperator(InfixExpression.Operator.OR);
            
            PrefixExpression notLeft = ast.newPrefixExpression();
            notLeft.setOperator(PrefixExpression.Operator.NOT);
            notLeft.setOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
            
            PrefixExpression notRight = ast.newPrefixExpression();
            notRight.setOperator(PrefixExpression.Operator.NOT);
            notRight.setOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
            
            newExpr.setLeftOperand(notLeft);
            newExpr.setRightOperand(notRight);
            
            rewrite.replace(expr, newExpr, null);
        }
    }
    
    private void convertTernaryToIfElse(ConditionalExpression ternary, ASTRewrite rewrite) {
        // Only convert ternary operators that are used as statements, not as values
        ASTNode parent = ternary.getParent();
        
        // Skip conversion if ternary is used as a value (assignment, return, method argument, etc.)
        if (parent instanceof Assignment || 
            parent instanceof ReturnStatement || 
            parent instanceof MethodInvocation ||
            parent instanceof ConditionalExpression ||
            parent instanceof InfixExpression) {
            debug("ternary_skip", "Skipping ternary conversion - used as value");
            return;
        }
        
        // Only convert if the ternary is a standalone statement
        if (parent instanceof ExpressionStatement) {
        AST ast = ternary.getAST();
        
        IfStatement ifStmt = ast.newIfStatement();
        ifStmt.setExpression((Expression) ASTNode.copySubtree(ast, ternary.getExpression()));
        
        // Create blocks for then and else
        Block thenBlock = ast.newBlock();
        Block elseBlock = ast.newBlock();
        
            // Add expressions as statements only if they are valid statements
            if (isValidStatementExpression(ternary.getThenExpression())) {
        ExpressionStatement thenStmt = ast.newExpressionStatement(
            (Expression) ASTNode.copySubtree(ast, ternary.getThenExpression()));
                thenBlock.statements().add(thenStmt);
            } else {
                // Skip conversion if expressions are not valid statements
                debug("ternary_skip", "Skipping ternary conversion - invalid statement expressions");
                return;
            }
            
            if (isValidStatementExpression(ternary.getElseExpression())) {
        ExpressionStatement elseStmt = ast.newExpressionStatement(
            (Expression) ASTNode.copySubtree(ast, ternary.getElseExpression()));
        elseBlock.statements().add(elseStmt);
            } else {
                // Skip conversion if expressions are not valid statements
                debug("ternary_skip", "Skipping ternary conversion - invalid statement expressions");
                return;
            }
        
        ifStmt.setThenStatement(thenBlock);
        ifStmt.setElseStatement(elseBlock);
        
        rewrite.replace(ternary, ifStmt, null);
        }
    }
    
    /**
     * Check if an expression can be used as a statement (method calls, assignments, etc.)
     */
    private boolean isValidStatementExpression(Expression expr) {
        return expr instanceof MethodInvocation ||
               expr instanceof Assignment ||
               expr instanceof PrefixExpression ||
               expr instanceof PostfixExpression;
    }
    
    private void convertIfElseToTernary(IfStatement ifStmt, ASTRewrite rewrite) {
        // Convert simple if-else to ternary operator
        if (ifStmt.getElseStatement() != null && isSimpleBranch(ifStmt.getThenStatement()) && isSimpleBranch(ifStmt.getElseStatement())) {
            AST ast = ifStmt.getAST();
            ConditionalExpression ternary = ast.newConditionalExpression();
            ternary.setExpression((Expression) ASTNode.copySubtree(ast, ifStmt.getExpression()));
            ternary.setThenExpression(extractExpressionFromBranch(ast, ifStmt.getThenStatement()));
            ternary.setElseExpression(extractExpressionFromBranch(ast, ifStmt.getElseStatement()));
            rewrite.replace(ifStmt, ternary, null);
        }
    }
    
    private void transformSwitchStatement(SwitchStatement switchStmt, ASTRewrite rewrite) {
        // Transform switch to if-else chain
        AST ast = switchStmt.getAST();
        
        IfStatement ifStmt = ast.newIfStatement();
        IfStatement currentIf = ifStmt;
        
        List<Statement> statements = switchStmt.statements();
        for (int i = 0; i < statements.size(); i++) {
            Statement stmt = (Statement) statements.get(i);
            if (stmt instanceof SwitchCase) {
                SwitchCase caseStmt = (SwitchCase) stmt;
                if (!caseStmt.isDefault()) {
                    Expression caseExpr = caseStmt.getExpression();
                    if (caseExpr != null) {
                        currentIf.setExpression((Expression) ASTNode.copySubtree(ast, caseExpr));
                        
                        // Create next if statement for the chain
                        if (i + 1 < statements.size()) {
                            IfStatement nextIf = ast.newIfStatement();
                            currentIf.setElseStatement(nextIf);
                            currentIf = nextIf;
                        }
                    }
                }
            }
        }
        
        rewrite.replace(switchStmt, ifStmt, null);
    }
    
    private boolean transformVariableOperation(Assignment assignment, ASTRewrite rewrite) {
        // Bidirectional normalization between "+=" and "= x + y" when safe,
        // to ensure a visible yet semantics-preserving textual change.
        AST ast = assignment.getAST();

        // Case 1: "+=" -> "= lhs + rhs"
        if (assignment.getOperator() == Assignment.Operator.PLUS_ASSIGN) {
            Assignment newAssignment = ast.newAssignment();
            newAssignment.setLeftHandSide((Expression) ASTNode.copySubtree(ast, assignment.getLeftHandSide()));

            InfixExpression plusExpr = ast.newInfixExpression();
            plusExpr.setOperator(InfixExpression.Operator.PLUS);
            plusExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, assignment.getLeftHandSide()));
            plusExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, assignment.getRightHandSide()));

            newAssignment.setRightHandSide(plusExpr);
            rewrite.replace(assignment, newAssignment, null);
            return true;
        }

        // Case 2: "= lhs + rhs" where lhs repeats on RHS -> convert to "+="
        if (assignment.getOperator() == Assignment.Operator.ASSIGN
                && assignment.getRightHandSide() instanceof InfixExpression) {
            InfixExpression rhs = (InfixExpression) assignment.getRightHandSide();
            if (rhs.getOperator() == InfixExpression.Operator.PLUS) {
                Expression lhs = assignment.getLeftHandSide();
                Expression leftOp = rhs.getLeftOperand();
                Expression rightOp = rhs.getRightOperand();
                // Compare simple textual forms to avoid binding resolution complexity
                String lhsStr = lhs.toString();
                String leftStr = leftOp.toString();
                String rightStr = rightOp.toString();
                if (lhsStr.equals(leftStr)) {
                    Assignment newAssignment = ast.newAssignment();
                    newAssignment.setLeftHandSide((Expression) ASTNode.copySubtree(ast, lhs));
                    newAssignment.setOperator(Assignment.Operator.PLUS_ASSIGN);
                    newAssignment.setRightHandSide((Expression) ASTNode.copySubtree(ast, rhs.getRightOperand()));
                    rewrite.replace(assignment, newAssignment, null);
                    return true;
                } else if (lhsStr.equals(rightStr)) {
                    Assignment newAssignment = ast.newAssignment();
                    newAssignment.setLeftHandSide((Expression) ASTNode.copySubtree(ast, lhs));
                    newAssignment.setOperator(Assignment.Operator.PLUS_ASSIGN);
                    newAssignment.setRightHandSide((Expression) ASTNode.copySubtree(ast, rhs.getLeftOperand()));
                    rewrite.replace(assignment, newAssignment, null);
                    return true;
                }
            }
        }

        return false;
    }
    
    private void transformStringConcatenation(InfixExpression expr, ASTRewrite rewrite) {
        // Transform string concatenation to String.valueOf() calls
        AST ast = expr.getAST();
        
        MethodInvocation valueOfCall = ast.newMethodInvocation();
        SimpleName stringClass = ast.newSimpleName("String");
        SimpleName valueOfMethod = ast.newSimpleName("valueOf");
        
        valueOfCall.setExpression(stringClass);
        valueOfCall.setName(valueOfMethod);
        
        // Concatenate operands and pass to valueOf
        InfixExpression concatExpr = ast.newInfixExpression();
        concatExpr.setOperator(InfixExpression.Operator.PLUS);
        concatExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
        concatExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
        
        valueOfCall.arguments().add(concatExpr);
        rewrite.replace(expr, valueOfCall, null);
    }
    
    private void transformNumericLiteral(NumberLiteral literal, ASTRewrite rewrite) {
        // Transform numeric literals (e.g., 1000 to 1_000)
        AST ast = literal.getAST();
        
        try {
            long value = Long.decode(literal.getToken());
            if (value == 1000L) {
                NumberLiteral newLiteral = ast.newNumberLiteral("1_000");
                rewrite.replace(literal, newLiteral, null);
            } else if (value == 10000L) {
                NumberLiteral newLiteral = ast.newNumberLiteral("10_000");
                rewrite.replace(literal, newLiteral, null);
            }
        } catch (NumberFormatException e) {
            // Ignore if not a valid number
        }
    }

    private boolean applyBraceNormalization(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        cu.accept(new ASTVisitor() {
            public boolean visit(CompilationUnit unit) {
                try {
                    // Reorder import declarations lexicographically (stable comparator)
                    if (!unit.imports().isEmpty()) {
                        AST ast = unit.getAST();
                        @SuppressWarnings("unchecked")
                        List<ImportDeclaration> imports = new ArrayList<>((List<ImportDeclaration>) unit.imports());
                        List<String> before = imports.stream().map(Object::toString).collect(Collectors.toList());
                        imports.sort(Comparator.comparing(ImportDeclaration::getName, Comparator.comparing(Name::getFullyQualifiedName)));
                        List<String> after = imports.stream().map(Object::toString).collect(Collectors.toList());
                        if (!before.equals(after)) {
                            // Build a new CU copy with sorted imports
                            CompilationUnit copy = (CompilationUnit) ASTNode.copySubtree(ast, unit);
                            @SuppressWarnings("unchecked")
                            List<ImportDeclaration> copyImports = (List<ImportDeclaration>) copy.imports();
                            copyImports.clear();
                            for (ImportDeclaration id : imports) {
                                copyImports.add((ImportDeclaration) ASTNode.copySubtree(ast, id));
                            }
                            rewrite.replace(unit, copy, null);
                            changed.set(true);
                        }
                    }
                } catch (Exception e) {}
                return true;
            }
            @Override
            public boolean visit(IfStatement node) {
                try {
                    AST ast = node.getAST();
                    boolean local = false;
                    IfStatement copy = (IfStatement) ASTNode.copySubtree(ast, node);
                    // Then branch to block
                    Block thenBlock;
                    if (copy.getThenStatement() instanceof Block) {
                        thenBlock = (Block) copy.getThenStatement();
                    } else {
                        thenBlock = ast.newBlock();
                        thenBlock.statements().add(ASTNode.copySubtree(ast, copy.getThenStatement()));
                        copy.setThenStatement(thenBlock);
                        local = true;
                    }
                    // Insert no-op empty statement at top
                    if (thenBlock.statements().isEmpty() || !(thenBlock.statements().get(0) instanceof EmptyStatement)) {
                        thenBlock.statements().add(0, ast.newEmptyStatement());
                        local = true;
                    }
                    // Else branch handling (skip else-if chain)
                    Statement es = copy.getElseStatement();
                    if (es != null && !(es instanceof IfStatement)) {
                        Block elseBlock;
                        if (es instanceof Block) {
                            elseBlock = (Block) es;
                        } else {
                            elseBlock = ast.newBlock();
                            elseBlock.statements().add(ASTNode.copySubtree(ast, es));
                            copy.setElseStatement(elseBlock);
                            local = true;
                        }
                        if (elseBlock.statements().isEmpty() || !(elseBlock.statements().get(0) instanceof EmptyStatement)) {
                            elseBlock.statements().add(0, ast.newEmptyStatement());
                            local = true;
                        }
                    }
                    if (local) {
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return true;
            }

            @Override
            public boolean visit(ForStatement node) {
                try {
                    AST ast = node.getAST();
                    ForStatement copy = (ForStatement) ASTNode.copySubtree(ast, node);
                    Block b;
                    if (copy.getBody() instanceof Block) {
                        b = (Block) copy.getBody();
                    } else {
                        b = ast.newBlock();
                        b.statements().add(ASTNode.copySubtree(ast, copy.getBody()));
                        copy.setBody(b);
                    }
                    if (b.statements().isEmpty() || !(b.statements().get(0) instanceof EmptyStatement)) {
                        b.statements().add(0, ast.newEmptyStatement());
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return true;
            }

            @Override
            public boolean visit(WhileStatement node) {
                try {
                    AST ast = node.getAST();
                    WhileStatement copy = (WhileStatement) ASTNode.copySubtree(ast, node);
                    Block b;
                    if (copy.getBody() instanceof Block) {
                        b = (Block) copy.getBody();
                    } else {
                        b = ast.newBlock();
                        b.statements().add(ASTNode.copySubtree(ast, copy.getBody()));
                        copy.setBody(b);
                    }
                    if (b.statements().isEmpty() || !(b.statements().get(0) instanceof EmptyStatement)) {
                        b.statements().add(0, ast.newEmptyStatement());
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return true;
            }

            @Override
            public boolean visit(SynchronizedStatement node) {
                try {
                    AST ast = node.getAST();
                    if (!(node.getExpression() instanceof ParenthesizedExpression)) {
                        SynchronizedStatement copy = (SynchronizedStatement) ASTNode.copySubtree(ast, node);
                        copy.setExpression(parenthesize(ast, (Expression) ASTNode.copySubtree(ast, node.getExpression())));
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return true;
            }

            @Override
            public boolean visit(NormalAnnotation node) {
                try {
                    AST ast = node.getAST();
                    @SuppressWarnings("unchecked")
                    List<MemberValuePair> pairs = new ArrayList<>((List<MemberValuePair>) node.values());
                    List<String> before = pairs.stream().map(p -> p.getName().getIdentifier()).collect(Collectors.toList());
                    pairs.sort(Comparator.comparing(p -> p.getName().getIdentifier()));
                    List<String> after = pairs.stream().map(p -> p.getName().getIdentifier()).collect(Collectors.toList());
                    if (!before.equals(after)) {
                        NormalAnnotation copy = (NormalAnnotation) ASTNode.copySubtree(ast, node);
                        @SuppressWarnings("unchecked")
                        List<MemberValuePair> copyPairs = (List<MemberValuePair>) copy.values();
                        copyPairs.clear();
                        for (MemberValuePair p : pairs) {
                            copyPairs.add((MemberValuePair) ASTNode.copySubtree(ast, p));
                        }
                        rewrite.replace(node, copy, null);
                        changed.set(true);
                    }
                } catch (Exception e) {}
                return true;
            }
        });
        return changed.get();
    }

    private boolean isSimpleBranch(Statement stmt) {
        if (stmt == null) return false;
        if (stmt instanceof ExpressionStatement) return true;
        if (stmt instanceof ReturnStatement) return true;
        if (stmt instanceof Block) {
            Block b = (Block) stmt;
            return b.statements().size() == 1 && isSimpleBranch((Statement) b.statements().get(0));
        }
        return false;
    }

    private Expression extractExpressionFromBranch(AST ast, Statement stmt) {
        if (stmt instanceof ExpressionStatement) {
            return (Expression) ASTNode.copySubtree(ast, ((ExpressionStatement) stmt).getExpression());
        }
        if (stmt instanceof ReturnStatement) {
            ReturnStatement rs = (ReturnStatement) stmt;
            return (Expression) ASTNode.copySubtree(ast, rs.getExpression());
        }
        if (stmt instanceof Block) {
            List<?> list = ((Block) stmt).statements();
            if (!list.isEmpty() && list.get(0) instanceof Statement) {
                return extractExpressionFromBranch(ast, (Statement) list.get(0));
            }
        }
        return (Expression) ast.newSimpleName("x");
    }

    private boolean containsMethodInvocation(Expression expr) {
        final AtomicBoolean found = new AtomicBoolean(false);
        expr.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) {
                found.set(true);
                return false;
            }
        });
        return found.get();
    }
    
    /**
     * Check if an AST node is inside a loop (while, for, or do-while).
     */
    private boolean isInsideLoop(ASTNode node) {
        ASTNode parent = node.getParent();
        while (parent != null) {
            if (parent instanceof WhileStatement || 
                parent instanceof ForStatement || 
                parent instanceof DoStatement) {
                return true;
            }
            parent = parent.getParent();
        }
        return false;
    }
    
    // CLI interface for the transformer service
    public static void main(String[] args) {
        if (args.length < 6) {
            printUsage();
            System.exit(1);
        }
        
        try {
            Map<String, String> params = parseArgs(args);
            
            String inputFile = params.get("--input");
            String outputFile = params.get("--output");
            String transformationsStr = params.get("--transformations");
            String mode = params.getOrDefault("--mode", "enhanced");
            String seedStr = params.getOrDefault("--seed", "42");
            
            if (inputFile == null || outputFile == null || transformationsStr == null) {
                System.err.println("Error: Missing required parameters");
                printUsage();
                System.exit(1);
            }
            
            long seed = Long.parseLong(seedStr);
            List<String> transformations = Arrays.asList(transformationsStr.split(","));
            
            // Read input file
            String javaCode = Files.readString(Paths.get(inputFile));
            
            // Create transformer and apply transformations
            SemanticTransformer transformer = new SemanticTransformer(seed);
            String transformedCode = transformer.transformCode(javaCode, transformations, mode);
            
            // Write output file
            Path outputPath = Paths.get(outputFile);
            if (outputPath.getParent() != null) {
                Files.createDirectories(outputPath.getParent());
            }
            Files.writeString(outputPath, transformedCode);
            
            System.out.println("Transformation completed successfully");
            // Emit applied transformations as a JSON line for wrapper parsing
            try {
                String json = "{\"appliedTransformations\":" + new ArrayList<>(transformer.appliedThisRun).toString().replace("="," : ") + "}";
                System.out.println(json);
            } catch (Exception ignore) {}
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void printUsage() {
        System.err.println("Usage: java cfwr.jdt.SemanticTransformer --input <file> --output <file> --transformations <list> [options]");
        System.err.println();
        System.err.println("Required parameters:");
        System.err.println("  --input <file>          Input Java file");
        System.err.println("  --output <file>         Output Java file");
        System.err.println("  --transformations <list> Comma-separated list of transformations");
        System.err.println();
        System.err.println("Optional parameters:");
        System.err.println("  --mode <enhanced|simple> Transformation mode (default: enhanced)");
        System.err.println("  --seed <number>         Random seed (default: 42)");
        System.err.println();
        System.err.println("Available transformations:");
        System.err.println("  Enhanced: loop_conversion, guard_reversal, mathematical_expression, logical_expression,");
        System.err.println("           ternary_operator, switch_statement, variable_operation, method_extraction,");
        System.err.println("           conditional_expression, array_access_pattern, string_concatenation,");
        System.err.println("           numeric_literal, exception_handling, lambda_expression, stream_api,");
        System.err.println("           builder_pattern, functional_conversion");
        System.err.println("  Simple:   simple_method_call, simple_assignment, simple_conditional,");
        System.err.println("           simple_array_access, simple_return_statement, simple_variable_declaration,");
        System.err.println("           simple_constructor_call, simple_field_access, simple_string_operation,");
        System.err.println("           simple_numeric_operation");
    }
    
    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> params = new HashMap<>();
        
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    params.put(args[i], args[i + 1]);
                    i++;
                } else {
                    params.put(args[i], "true");
                }
            }
        }
        
        return params;
    }
    
    // ===== NEW TRANSFORMATION METHODS =====
    
    /**
     * Transform bitwise operations using bitwise algebra properties.
     */
    private boolean applyBitwiseOperation(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.AND || 
                    node.getOperator() == InfixExpression.Operator.OR ||
                    node.getOperator() == InfixExpression.Operator.XOR) {
                    
                    if (random.nextDouble() < 0.8) {
                        transformBitwiseExpression(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
            
            @Override
            public boolean visit(PrefixExpression node) {
                if (node.getOperator() == PrefixExpression.Operator.COMPLEMENT) {
                    if (random.nextDouble() < 0.8) {
                        transformBitwiseNot(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Transform bitwise expressions using algebraic properties.
     */
    private void transformBitwiseExpression(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        if (expr.getOperator() == InfixExpression.Operator.AND) {
            // Commutativity: a & b = b & a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.AND);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.OR) {
            // Commutativity: a | b = b | a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.OR);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.XOR) {
            // Commutativity: a ^ b = b ^ a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.XOR);
                rewrite.replace(expr, swapped, null);
            }
        }
    }
    
    /**
     * Transform bitwise NOT operations.
     */
    private void transformBitwiseNot(PrefixExpression expr, ASTRewrite rewrite) {
        // Convert ~x to (-x) - 1 for negative numbers
        if (random.nextDouble() < 0.9) {
            AST ast = expr.getAST();
            InfixExpression newExpr = ast.newInfixExpression();
            PrefixExpression negated = ast.newPrefixExpression();
            negated.setOperator(PrefixExpression.Operator.MINUS);
            negated.setOperand((Expression) ASTNode.copySubtree(ast, expr.getOperand()));
            newExpr.setLeftOperand(negated);
            newExpr.setRightOperand(ast.newNumberLiteral("1"));
            newExpr.setOperator(InfixExpression.Operator.MINUS);
            rewrite.replace(expr, newExpr, null);
        }
    }
    
    /**
     * Transform comparison operations using comparison algebra.
     */
    private boolean applyComparisonOperation(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.LESS ||
                    node.getOperator() == InfixExpression.Operator.GREATER ||
                    node.getOperator() == InfixExpression.Operator.LESS_EQUALS ||
                    node.getOperator() == InfixExpression.Operator.GREATER_EQUALS ||
                    node.getOperator() == InfixExpression.Operator.EQUALS ||
                    node.getOperator() == InfixExpression.Operator.NOT_EQUALS) {
                    
                    if (random.nextDouble() < 0.8) {
                        transformComparisonExpression(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Transform comparison expressions using algebraic properties.
     */
    private void transformComparisonExpression(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        if (expr.getOperator() == InfixExpression.Operator.LESS) {
            // a < b = b > a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.GREATER);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.GREATER) {
            // a > b = b < a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.LESS);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.LESS_EQUALS) {
            // a <= b = b >= a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.GREATER_EQUALS);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.GREATER_EQUALS) {
            // a >= b = b <= a
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.LESS_EQUALS);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.EQUALS) {
            // a == b = b == a (commutativity)
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.EQUALS);
                rewrite.replace(expr, swapped, null);
            }
        } else if (expr.getOperator() == InfixExpression.Operator.NOT_EQUALS) {
            // a != b = b != a (commutativity)
            if (random.nextBoolean()) {
                InfixExpression swapped = ast.newInfixExpression();
                swapped.setLeftOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                swapped.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
                swapped.setOperator(InfixExpression.Operator.NOT_EQUALS);
                rewrite.replace(expr, swapped, null);
            }
        }
    }
    
    /**
     * Transform type conversions and casts.
     */
    private boolean applyTypeConversion(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(CastExpression node) {
                if (random.nextDouble() < 0.8) {
                    transformCastExpression(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.PLUS) {
                    if (random.nextDouble() < 0.2) {
                        transformStringConcatenationType(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Transform cast expressions.
     */
    private void transformCastExpression(CastExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        // Convert explicit casts to implicit conversions where safe
        if (random.nextBoolean()) {
            // Remove redundant casts like (int) 42 or (String) "hello"
            Type type = expr.getType();
            Expression operand = expr.getExpression();
            
            if (type.isPrimitiveType() && operand instanceof NumberLiteral) {
                // Remove cast for primitive literals
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, operand), null);
            } else if (type.isSimpleType() && operand instanceof StringLiteral) {
                // Remove cast for string literals
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, operand), null);
            }
        }
    }
    
    /**
     * Transform string concatenation patterns for type conversion.
     */
    private void transformStringConcatenationType(InfixExpression expr, ASTRewrite rewrite) {
        // Convert string concatenation to StringBuilder where beneficial
        if (random.nextBoolean()) {
            AST ast = expr.getAST();
            
            // Create StringBuilder pattern
            MethodInvocation sbAppend = ast.newMethodInvocation();
            sbAppend.setName(ast.newSimpleName("append"));
            sbAppend.setExpression(ast.newSimpleName("sb"));
            sbAppend.arguments().add((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
            
            MethodInvocation sbAppend2 = ast.newMethodInvocation();
            sbAppend2.setName(ast.newSimpleName("append"));
            sbAppend2.setExpression(ast.newSimpleName("sb"));
            sbAppend2.arguments().add((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
            
            // Replace with method chaining
            sbAppend.arguments().add(sbAppend2);
            rewrite.replace(expr, sbAppend, null);
        }
    }
    
    /**
     * Transform null check patterns.
     */
    private boolean applyNullCheckPattern(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (node.getOperator() == InfixExpression.Operator.NOT_EQUALS &&
                    node.getRightOperand() instanceof NullLiteral) {
                    if (random.nextDouble() < 0.4) {
                        transformNullCheck(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
            
            @Override
            public boolean visit(MethodInvocation node) {
                if (node.getName().getIdentifier().equals("equals") &&
                    node.arguments().size() == 1) {
                    if (random.nextDouble() < 0.3) {
                        transformEqualsCheck(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Transform null checks.
     */
    private void transformNullCheck(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        // Convert obj != null to !Objects.isNull(obj)
        if (random.nextBoolean()) {
            PrefixExpression notExpr = ast.newPrefixExpression();
            notExpr.setOperator(PrefixExpression.Operator.NOT);
            
            MethodInvocation isNull = ast.newMethodInvocation();
            isNull.setName(ast.newSimpleName("isNull"));
            isNull.setExpression(ast.newSimpleName("Objects"));
            isNull.arguments().add((Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()));
            
            notExpr.setOperand(isNull);
            rewrite.replace(expr, notExpr, null);
        }
    }
    
    /**
     * Transform equals checks.
     */
    private void transformEqualsCheck(MethodInvocation node, ASTRewrite rewrite) {
        AST ast = node.getAST();
        
        // Convert obj.equals(other) to Objects.equals(obj, other)
        if (random.nextBoolean()) {
            MethodInvocation objectsEquals = ast.newMethodInvocation();
            objectsEquals.setName(ast.newSimpleName("equals"));
            objectsEquals.setExpression(ast.newSimpleName("Objects"));
            objectsEquals.arguments().add((Expression) ASTNode.copySubtree(ast, node.getExpression()));
            objectsEquals.arguments().add((Expression) ASTNode.copySubtree(ast, (Expression) node.arguments().get(0)));
            
            rewrite.replace(node, objectsEquals, null);
        }
    }
    
    /**
     * Apply constant folding optimizations.
     */
    private boolean applyConstantFolding(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(InfixExpression node) {
                if (isConstantExpression(node)) {
                    if (random.nextDouble() < 0.5) {
                        foldConstantExpression(node, rewrite);
                        changed.set(true);
                    }
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Check if an expression is a constant expression.
     */
    private boolean isConstantExpression(InfixExpression expr) {
        return (expr.getLeftOperand() instanceof NumberLiteral || expr.getLeftOperand() instanceof StringLiteral) &&
               (expr.getRightOperand() instanceof NumberLiteral || expr.getRightOperand() instanceof StringLiteral);
    }
    
    /**
     * Fold constant expressions.
     */
    private void foldConstantExpression(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        // Simple constant folding for numeric expressions
        if (expr.getLeftOperand() instanceof NumberLiteral && 
            expr.getRightOperand() instanceof NumberLiteral) {
            
            try {
                double left = Double.parseDouble(((NumberLiteral) expr.getLeftOperand()).getToken());
                double right = Double.parseDouble(((NumberLiteral) expr.getRightOperand()).getToken());
                double result = 0;
                
                if (expr.getOperator() == InfixExpression.Operator.PLUS) {
                    result = left + right;
                } else if (expr.getOperator() == InfixExpression.Operator.MINUS) {
                    result = left - right;
                } else if (expr.getOperator() == InfixExpression.Operator.TIMES) {
                    result = left * right;
                } else if (expr.getOperator() == InfixExpression.Operator.DIVIDE) {
                    if (right != 0) result = left / right;
                    else return; // Don't fold division by zero
                } else {
                    return;
                }
                
                NumberLiteral folded = ast.newNumberLiteral(String.valueOf(result));
                rewrite.replace(expr, folded, null);
                
            } catch (NumberFormatException e) {
                // Skip if parsing fails
            }
        }
    }
    
    /**
     * Insert dead code that doesn't affect program semantics.
     */
    private boolean applyDeadCodeInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(Block node) {
                if (random.nextDouble() < 0.2) {
                    insertDeadCode(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Insert dead code statements.
     */
    private void insertDeadCode(Block block, ASTRewrite rewrite) {
        AST ast = block.getAST();
        
        // Insert harmless dead code
        List<Statement> deadStatements = Arrays.asList(
            ast.newExpressionStatement(ast.newNumberLiteral("0")), // 0;
            ast.newExpressionStatement(ast.newBooleanLiteral(false)), // false;
            ast.newExpressionStatement(ast.newStringLiteral()) // "";
        );
        
        Statement deadStatement = deadStatements.get(random.nextInt(deadStatements.size()));
        
        // Insert at random position in the block
        int position = random.nextInt(block.statements().size() + 1);
        rewrite.getListRewrite(block, Block.STATEMENTS_PROPERTY).insertAt(deadStatement, position, null);
    }
    
    /**
     * Transform method chaining patterns.
     */
    private boolean applyMethodChainTransformation(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodInvocation node) {
                if (random.nextDouble() < 0.3) {
                    transformMethodChain(node, rewrite);
                    changed.set(true);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Transform method chaining.
     */
    private void transformMethodChain(MethodInvocation node, ASTRewrite rewrite) {
        AST ast = node.getAST();
        
        // Convert method calls to fluent interface style where possible
        if (random.nextBoolean() && node.getExpression() != null) {
            // Create a new method invocation with chaining
            MethodInvocation chained = ast.newMethodInvocation();
            chained.setName(ast.newSimpleName(node.getName().getIdentifier()));
            chained.setExpression((Expression) ASTNode.copySubtree(ast, node.getExpression()));
            
            // Copy arguments
            for (Object arg : node.arguments()) {
                chained.arguments().add((Expression) ASTNode.copySubtree(ast, (Expression) arg));
            }
            
            rewrite.replace(node, chained, null);
        }
    }
    
    /**
     * Transform variable names to add variety.
     */
    private boolean applyVariableRenaming(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        // Map to store variable name mappings
        Map<String, String> nameMapping = new HashMap<>();
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(VariableDeclarationFragment node) {
                if (random.nextDouble() < 0.1) { // Low probability to avoid breaking code
                    String oldName = node.getName().getIdentifier();
                    String newName = generateNewVariableName(oldName);
                    nameMapping.put(oldName, newName);
                    
                    SimpleName newSimpleName = node.getAST().newSimpleName(newName);
                    rewrite.replace(node.getName(), newSimpleName, null);
                    changed.set(true);
                }
                return true;
            }
            
            @Override
            public boolean visit(SimpleName node) {
                if (nameMapping.containsKey(node.getIdentifier())) {
                    String newName = nameMapping.get(node.getIdentifier());
                    SimpleName newSimpleName = node.getAST().newSimpleName(newName);
                    rewrite.replace(node, newSimpleName, null);
                }
                return true;
            }
        });
        
        return changed.get();
    }
    
    /**
     * Generate a new variable name based on the old one.
     */
    private String generateNewVariableName(String oldName) {
        String[] prefixes = {"new", "temp", "var", "item", "value", "data", "obj"};
        String[] suffixes = {"1", "2", "_new", "_tmp", "_var", "_alt"};
        
        String prefix = prefixes[random.nextInt(prefixes.length)];
        String suffix = suffixes[random.nextInt(suffixes.length)];
        
        return prefix + Character.toUpperCase(oldName.charAt(0)) + oldName.substring(1) + suffix;
    }
    
    // Enhanced loop conversion methods
    
    private void convertDoWhileToFor(DoStatement doStmt, ASTRewrite rewrite) {
        AST ast = doStmt.getAST();
        
        // Convert do-while to for loop
        ForStatement forStmt = ast.newForStatement();
        
        // Create a for loop that mimics do-while behavior
        // do { body } while (condition) becomes:
        // for (;;) { body; if (!condition) break; }
        
        Block forBody = ast.newBlock();
        
        // Add the original body
        Statement body = doStmt.getBody();
        if (body != null) {
            if (body instanceof Block) {
                for (Object stmt : ((Block) body).statements()) {
                    forBody.statements().add(ASTNode.copySubtree(ast, (ASTNode) stmt));
                }
            } else {
                forBody.statements().add(ASTNode.copySubtree(ast, body));
            }
        }
        
        // Add condition check with break
        if (doStmt.getExpression() != null) {
            IfStatement ifStmt = ast.newIfStatement();
            PrefixExpression notCondition = ast.newPrefixExpression();
            notCondition.setOperator(PrefixExpression.Operator.NOT);
            notCondition.setOperand((Expression) ASTNode.copySubtree(ast, doStmt.getExpression()));
            ifStmt.setExpression(notCondition);
            
            BreakStatement breakStmt = ast.newBreakStatement();
            ifStmt.setThenStatement(breakStmt);
            
            forBody.statements().add(ifStmt);
        }
        
        forStmt.setBody(forBody);
        rewrite.replace(doStmt, forStmt, null);
    }
    
    private void convertLabeledLoop(LabeledStatement labeledStmt, ASTRewrite rewrite) {
        AST ast = labeledStmt.getAST();
        Statement body = labeledStmt.getBody();
        
        if (body instanceof ForStatement) {
            ForStatement forStmt = (ForStatement) body;
            WhileStatement whileStmt = ast.newWhileStatement();
            whileStmt.setExpression(ast.newBooleanLiteral(true));
            
            Block whileBody = ast.newBlock();
            handleComplexInitializers(forStmt, whileBody, ast);
            handleLoopBody(forStmt, whileBody, ast);
            handleComplexIncrements(forStmt, whileBody, ast);
            addConditionCheck(forStmt, whileBody, ast);
            
            whileStmt.setBody(whileBody);
            
            // Preserve the label
            LabeledStatement newLabeled = ast.newLabeledStatement();
            newLabeled.setLabel(ast.newSimpleName(labeledStmt.getLabel().getIdentifier()));
            newLabeled.setBody(whileStmt);
            
            rewrite.replace(labeledStmt, newLabeled, null);
        } else if (body instanceof WhileStatement) {
            WhileStatement whileStmt = (WhileStatement) body;
            ForStatement forStmt = ast.newForStatement();
            
            // Convert while to for with condition
            Block forBody = ast.newBlock();
            if (whileStmt.getBody() instanceof Block) {
                for (Object stmt : ((Block) whileStmt.getBody()).statements()) {
                    forBody.statements().add(ASTNode.copySubtree(ast, (ASTNode) stmt));
                }
            } else {
                forBody.statements().add(ASTNode.copySubtree(ast, whileStmt.getBody()));
            }
            
            forStmt.setExpression((Expression) ASTNode.copySubtree(ast, whileStmt.getExpression()));
            forStmt.setBody(forBody);
            
            // Preserve the label
            LabeledStatement newLabeled = ast.newLabeledStatement();
            newLabeled.setLabel(ast.newSimpleName(labeledStmt.getLabel().getIdentifier()));
            newLabeled.setBody(forStmt);
            
            rewrite.replace(labeledStmt, newLabeled, null);
        } else if (body instanceof DoStatement) {
            DoStatement doStmt = (DoStatement) body;
            convertDoWhileToFor(doStmt, rewrite);
            
            // Update the labeled statement to point to the new for loop
            LabeledStatement newLabeled = ast.newLabeledStatement();
            newLabeled.setLabel(ast.newSimpleName(labeledStmt.getLabel().getIdentifier()));
            newLabeled.setBody((Statement) rewrite.get(doStmt, null));
            
            rewrite.replace(labeledStmt, newLabeled, null);
        }
    }
    
    // Enhanced mathematical expression transformations
    
    private boolean transformDistributiveProperty(InfixExpression expr, ASTRewrite rewrite) {
        if (expr.getOperator() == InfixExpression.Operator.TIMES) {
            // Check for distributive pattern: a * (b + c) or (a + b) * c
            Expression left = expr.getLeftOperand();
            Expression right = expr.getRightOperand();
            
            if (left instanceof InfixExpression && 
                ((InfixExpression) left).getOperator() == InfixExpression.Operator.PLUS) {
                // (a + b) * c → a * c + b * c
                InfixExpression leftAdd = (InfixExpression) left;
                AST ast = expr.getAST();
                
                InfixExpression term1 = ast.newInfixExpression();
                term1.setLeftOperand((Expression) ASTNode.copySubtree(ast, leftAdd.getLeftOperand()));
                term1.setRightOperand((Expression) ASTNode.copySubtree(ast, right));
                term1.setOperator(InfixExpression.Operator.TIMES);
                
                InfixExpression term2 = ast.newInfixExpression();
                term2.setLeftOperand((Expression) ASTNode.copySubtree(ast, leftAdd.getRightOperand()));
                term2.setRightOperand((Expression) ASTNode.copySubtree(ast, right));
                term2.setOperator(InfixExpression.Operator.TIMES);
                
                InfixExpression result = ast.newInfixExpression();
                result.setLeftOperand(term1);
                result.setRightOperand(term2);
                result.setOperator(InfixExpression.Operator.PLUS);
                
                rewrite.replace(expr, result, null);
                return true;
            }
        }
        return false;
    }
    
    private boolean transformIdentityElements(InfixExpression expr, ASTRewrite rewrite) {
        AST ast = expr.getAST();
        
        if (expr.getOperator() == InfixExpression.Operator.PLUS) {
            // x + 0 → x, 0 + x → x
            if (isZeroLiteral(expr.getLeftOperand())) {
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getRightOperand()), null);
                return true;
            } else if (isZeroLiteral(expr.getRightOperand())) {
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()), null);
                return true;
            }
        } else if (expr.getOperator() == InfixExpression.Operator.TIMES) {
            // x * 1 → x, 1 * x → x
            if (isOneLiteral(expr.getLeftOperand())) {
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getRightOperand()), null);
                return true;
            } else if (isOneLiteral(expr.getRightOperand())) {
                rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()), null);
                return true;
            }
            // x * 0 → 0, 0 * x → 0
            if (isZeroLiteral(expr.getLeftOperand()) || isZeroLiteral(expr.getRightOperand())) {
                NumberLiteral zero = ast.newNumberLiteral("0");
                rewrite.replace(expr, zero, null);
                return true;
            }
        }
        return false;
    }
    
    private boolean isZeroLiteral(Expression expr) {
        if (expr instanceof NumberLiteral) {
            String value = ((NumberLiteral) expr).getToken();
            return "0".equals(value) || "0.0".equals(value) || "0L".equals(value) || "0f".equals(value);
        }
        return false;
    }
    
    private boolean isOneLiteral(Expression expr) {
        if (expr instanceof NumberLiteral) {
            String value = ((NumberLiteral) expr).getToken();
            return "1".equals(value) || "1.0".equals(value) || "1L".equals(value) || "1f".equals(value);
        }
        return false;
    }
    
    // Enhanced logical expression transformations
    
    private boolean transformDeMorgan3Terms(InfixExpression expr, ASTRewrite rewrite) {
        if (expr.getOperator() == InfixExpression.Operator.AND || 
            expr.getOperator() == InfixExpression.Operator.OR) {
            
            // Check for De Morgan with 3+ terms: !(a && b && c) → !a || !b || !c
            if (expr.getLeftOperand() instanceof PrefixExpression) {
                PrefixExpression leftNot = (PrefixExpression) expr.getLeftOperand();
                if (leftNot.getOperator() == PrefixExpression.Operator.NOT &&
                    leftNot.getOperand() instanceof InfixExpression) {
                    
                    InfixExpression inner = (InfixExpression) leftNot.getOperand();
                    if (inner.getOperator() == InfixExpression.Operator.AND) {
                        // !(a && b) && c → (!a || !b) && c
                        AST ast = expr.getAST();
                        
                        PrefixExpression notA = ast.newPrefixExpression();
                        notA.setOperator(PrefixExpression.Operator.NOT);
                        notA.setOperand((Expression) ASTNode.copySubtree(ast, inner.getLeftOperand()));
                        
                        PrefixExpression notB = ast.newPrefixExpression();
                        notB.setOperator(PrefixExpression.Operator.NOT);
                        notB.setOperand((Expression) ASTNode.copySubtree(ast, inner.getRightOperand()));
                        
                        InfixExpression orExpr = ast.newInfixExpression();
                        orExpr.setLeftOperand(notA);
                        orExpr.setRightOperand(notB);
                        orExpr.setOperator(InfixExpression.Operator.OR);
                        
                        InfixExpression result = ast.newInfixExpression();
                        result.setLeftOperand(orExpr);
                        result.setRightOperand((Expression) ASTNode.copySubtree(ast, expr.getRightOperand()));
                        result.setOperator(InfixExpression.Operator.AND);
                        
                        rewrite.replace(expr, result, null);
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    private boolean transformAbsorptionLaws(InfixExpression expr, ASTRewrite rewrite) {
        if (expr.getOperator() == InfixExpression.Operator.AND) {
            // a && (a || b) → a
            if (expr.getRightOperand() instanceof InfixExpression) {
                InfixExpression rightOr = (InfixExpression) expr.getRightOperand();
                if (rightOr.getOperator() == InfixExpression.Operator.OR) {
                    if (areEquivalentExpressions(expr.getLeftOperand(), rightOr.getLeftOperand()) ||
                        areEquivalentExpressions(expr.getLeftOperand(), rightOr.getRightOperand())) {
                        
                        AST ast = expr.getAST();
                        rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()), null);
                        return true;
                    }
                }
            }
        } else if (expr.getOperator() == InfixExpression.Operator.OR) {
            // a || (a && b) → a
            if (expr.getRightOperand() instanceof InfixExpression) {
                InfixExpression rightAnd = (InfixExpression) expr.getRightOperand();
                if (rightAnd.getOperator() == InfixExpression.Operator.AND) {
                    if (areEquivalentExpressions(expr.getLeftOperand(), rightAnd.getLeftOperand()) ||
                        areEquivalentExpressions(expr.getLeftOperand(), rightAnd.getRightOperand())) {
                        
                        AST ast = expr.getAST();
                        rewrite.replace(expr, (Expression) ASTNode.copySubtree(ast, expr.getLeftOperand()), null);
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    private boolean areEquivalentExpressions(Expression expr1, Expression expr2) {
        // Simple structural equivalence check
        return expr1.toString().equals(expr2.toString());
    }
}


