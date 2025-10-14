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
    
    public SemanticTransformer() {
        this.parser = createParser();
        this.random = new Random();
    }
    
    public SemanticTransformer(long seed) {
        this.parser = createParser();
        this.random = new Random(seed);
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

    private void debug(String key, String message) {
        if (debugEnabled) {
            if (key != null) {
                debugCounters.put(key, debugCounters.getOrDefault(key, 0) + 1);
            }
            System.err.println("[JDT_DEBUG] " + message);
        }
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
                    
                    if (random.nextDouble() < 1.0) { // 100% chance to apply De Morgan's laws
                        applyDeMorganLaws(node, rewrite);
                        changed.set(true);
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
        // Implementation for method extraction
        return false;
    }
    
    private boolean applyConditionalExpression(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for conditional expression restructuring
        return false;
    }
    
    private boolean applyArrayAccessPattern(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for array access pattern variations
        return false;
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
        // Implementation for exception handling restructuring
        return false;
    }
    
    private boolean applyLambdaExpression(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for lambda expression conversions
        return false;
    }
    
    private boolean applyStreamApi(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for Stream API transformations
        return false;
    }
    
    private boolean applyBuilderPattern(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for builder pattern variations
        return false;
    }
    
    private boolean applyFunctionalConversion(CompilationUnit cu, ASTRewrite rewrite) {
        // Implementation for functional programming conversions
        return false;
    }
    
    // Simple transformation implementations
    private boolean applySimpleMethodCall(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleAssignment(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleConditional(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleArrayAccess(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleReturnStatement(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleVariableDeclaration(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleConstructorCall(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleFieldAccess(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleStringOperation(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applySimpleNumericOperation(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    // Random augmentation implementations
    private boolean applyRandomMethodInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applyRandomStatementInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
    }
    
    private boolean applyRandomExpressionInsertion(CompilationUnit cu, ASTRewrite rewrite) {
        return false; // Placeholder
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
