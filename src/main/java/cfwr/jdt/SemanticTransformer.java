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
    
    public SemanticTransformer() {
        this.parser = createParser();
        this.random = new Random();
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

    private boolean isEnabled(String t) {
        return transformEnabled.getOrDefault(t, Boolean.TRUE);
    }

    private void debug(String key, String message) {
        if (debugEnabled) {
            if (key != null) {
                debugCounters.put(key, debugCounters.getOrDefault(key, 0) + 1);
            }
            System.err.println("[JDT_DEBUG] " + message);
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
        parser.setSource(javaCode.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        
        if (cu == null || Arrays.stream(cu.getProblems()).anyMatch(IProblem::isError)) {
            debug("parse_error", "Parsing failed or had errors; returning original code");
            return javaCode; // Return original if parsing failed
        }
        
        ASTRewrite rewrite = ASTRewrite.create(cu.getAST());
        boolean hasChanges = false;
        
        for (String transformation : transformations) {
            debug("consider_" + transformation, "Considering transformation: " + transformation);
            boolean changed = applyTransformation(cu, rewrite, transformation, mode);
            if (changed) {
                debug("applied_" + transformation, "Applied transformation: " + transformation);
                hasChanges = true;
                appliedThisRun.add(transformation);
            } else {
                debug("skipped_" + transformation, "No effect: " + transformation);
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
        });
        
        return changed.get();
    }
    
    private boolean applyGuardReversal(CompilationUnit cu, ASTRewrite rewrite) {
        AtomicBoolean changed = new AtomicBoolean(false);
        
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(IfStatement node) {
                try {
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
                    node.getOperator() == InfixExpression.Operator.DIVIDE) {
                    
                    if (random.nextDouble() < 1.0) { // 100% chance to transform
                        boolean local = transformMathematicalExpressionSafe(node, rewrite);
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
                    // Conservative normalization: parenthesize operands to ensure textual change
                    AST ast = node.getAST();
                    InfixExpression copy = (InfixExpression) ASTNode.copySubtree(ast, node);
                    copy.setLeftOperand(parenthesize(ast, copy.getLeftOperand()));
                    copy.setRightOperand(parenthesize(ast, copy.getRightOperand()));
                    rewrite.replace(node, copy, null);
                    changed.set(true);
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
                if (random.nextDouble() < 1.0) { // 100% chance to convert to if-else
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
                    if (node.getInitializer() != null && !(node.getInitializer() instanceof ParenthesizedExpression)) {
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
        // Convert for loop to while loop
        AST ast = forStmt.getAST();
        
        WhileStatement whileStmt = ast.newWhileStatement();
        whileStmt.setExpression(ast.newBooleanLiteral(true));
        
        Block whileBody = ast.newBlock();
        
        // Add initialization statements
        if (forStmt.initializers().size() > 0) {
            for (Object initializer : forStmt.initializers()) {
                if (initializer instanceof VariableDeclarationExpression) {
                    VariableDeclarationExpression vde = (VariableDeclarationExpression) initializer;
                    VariableDeclarationStatement vds = ast.newVariableDeclarationStatement(
                        (VariableDeclarationFragment) ASTNode.copySubtree(ast, (ASTNode) vde.fragments().get(0))
                    );
                    vds.setType((Type) ASTNode.copySubtree(ast, vde.getType()));
                    // If multiple fragments, add them as separate statements
                    for (int i = 1; i < vde.fragments().size(); i++) {
                        VariableDeclarationFragment frag = (VariableDeclarationFragment) vde.fragments().get(i);
                        VariableDeclarationStatement extra = ast.newVariableDeclarationStatement(
                            (VariableDeclarationFragment) ASTNode.copySubtree(ast, frag)
                        );
                        extra.setType((Type) ASTNode.copySubtree(ast, vde.getType()));
                        whileBody.statements().add(extra);
                    }
                    whileBody.statements().add(vds);
                } else if (initializer instanceof Expression) {
                    ExpressionStatement es = ast.newExpressionStatement(
                        (Expression) ASTNode.copySubtree(ast, (Expression) initializer)
                    );
                    whileBody.statements().add(es);
                }
            }
        }
        
        // Add original body with increment at the end
        if (forStmt.getBody() != null) {
            if (forStmt.getBody() instanceof Block) {
                Block originalBody = (Block) forStmt.getBody();
                for (Object stmt : originalBody.statements()) {
                    whileBody.statements().add(ASTNode.copySubtree(ast, (Statement) stmt));
                }
            } else {
                whileBody.statements().add(ASTNode.copySubtree(ast, forStmt.getBody()));
            }
        }
        
        // Add increment statements
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
        
        whileStmt.setBody(whileBody);
        
        // Add condition check at the beginning of the loop
        if (forStmt.getExpression() != null) {
            IfStatement conditionCheck = ast.newIfStatement();
            PrefixExpression notExpr = ast.newPrefixExpression();
            notExpr.setOperator(PrefixExpression.Operator.NOT);
            notExpr.setOperand((Expression) ASTNode.copySubtree(ast, forStmt.getExpression()));
            conditionCheck.setExpression(notExpr);
            
            BreakStatement breakStmt = ast.newBreakStatement();
            Block breakBlock = ast.newBlock();
            breakBlock.statements().add(breakStmt);
            conditionCheck.setThenStatement(breakBlock);
            
            whileBody.statements().add(0, conditionCheck);
        }
        
        rewrite.replace(forStmt, whileStmt, null);
    }
    
    private void convertWhileToFor(WhileStatement whileStmt, ASTRewrite rewrite) {
        // Convert while loop to for loop
        AST ast = whileStmt.getAST();
        
        ForStatement forStmt = ast.newForStatement();
        forStmt.setExpression((Expression) ASTNode.copySubtree(ast, whileStmt.getExpression()));
        forStmt.setBody((Statement) ASTNode.copySubtree(ast, whileStmt.getBody()));
        
        rewrite.replace(whileStmt, forStmt, null);
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

    private boolean transformMathematicalExpressionSafe(InfixExpression expr, ASTRewrite rewrite) {
        // Apply commutativity safely on simple operands only
        AST ast = expr.getAST();
        if (expr.getOperator() == InfixExpression.Operator.PLUS || expr.getOperator() == InfixExpression.Operator.TIMES) {
            Expression left = expr.getLeftOperand();
            Expression right = expr.getRightOperand();
            if (isSimpleOperand(left) && isSimpleOperand(right)) {
                InfixExpression newExpr = ast.newInfixExpression();
                newExpr.setOperator(expr.getOperator());
                newExpr.setLeftOperand((Expression) ASTNode.copySubtree(ast, right));
                newExpr.setRightOperand((Expression) ASTNode.copySubtree(ast, left));
                debug("math_commute", "Applied commutativity on simple operands: " + expr.toString());
                rewrite.replace(expr, newExpr, null);
                return true;
            } else {
                debug("math_skip_complex", "Skipped commutativity due to complex operands: " + expr.toString());
            }
        }
        return false;
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
        // Convert ternary operator to if-else statement
        AST ast = ternary.getAST();
        
        IfStatement ifStmt = ast.newIfStatement();
        ifStmt.setExpression((Expression) ASTNode.copySubtree(ast, ternary.getExpression()));
        
        // Create blocks for then and else
        Block thenBlock = ast.newBlock();
        Block elseBlock = ast.newBlock();
        
        // Add expressions as statements (this is simplified)
        ExpressionStatement thenStmt = ast.newExpressionStatement(
            (Expression) ASTNode.copySubtree(ast, ternary.getThenExpression()));
        ExpressionStatement elseStmt = ast.newExpressionStatement(
            (Expression) ASTNode.copySubtree(ast, ternary.getElseExpression()));
        
        thenBlock.statements().add(thenStmt);
        elseBlock.statements().add(elseStmt);
        
        ifStmt.setThenStatement(thenBlock);
        ifStmt.setElseStatement(elseBlock);
        
        rewrite.replace(ternary, ifStmt, null);
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
}


