package cfwr.ast;

import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.formatter.DefaultCodeFormatterConstants;
import org.eclipse.jface.text.Document;
import org.eclipse.text.edits.TextEdit;

import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

public class SemanticAugmenter {

    public static void main(String[] args) throws Exception {
        Map<String, String> kv = parseArgs(args);
        Path in = Path.of(require(kv, "--in"));
        Path out = Path.of(require(kv, "--out"));
        String mode = kv.getOrDefault("--mode", "enhanced").toLowerCase(Locale.ROOT);
        long seed = Long.parseLong(kv.getOrDefault("--seed", "42"));
        Set<String> disabled = parseCsv(kv.getOrDefault("--disable", ""));

        String source = Files.readString(in);

        ASTParser parser = ASTParser.newParser(AST.JLS21);
        parser.setKind(ASTParser.K_COMPILATION_UNIT);
        parser.setSource(source.toCharArray());
        parser.setResolveBindings(false);
        parser.setBindingsRecovery(false);
        parser.setStatementsRecovery(true);
        parser.setCompilerOptions(DefaultCodeFormatterConstants.getEclipseDefaultSettings());

        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        if (cu == null || Arrays.stream(cu.getProblems()).anyMatch(Problem::isError)) {
            Files.createDirectories(out.getParent());
            Files.writeString(out, source);
            return;
        }

        // Enabled transform families
        Set<String> enabled = new LinkedHashSet<>(Arrays.asList(
                "guard", "demorgan", "ternary", "array_index", "identity_math"
        ));
        if (mode.equals("enhanced")) {
            enabled.addAll(Arrays.asList("string_concat_alt", "numeric_literal_alt"));
        }
        enabled.removeAll(disabled);

        ASTRewrite rw = ASTRewrite.create(cu.getAST());

        if (enabled.contains("guard")) applyGuardReversal(cu, rw);
        if (enabled.contains("demorgan")) applyDeMorgan(cu, rw);
        if (enabled.contains("ternary")) applyTernaryIfElse(cu, rw);
        if (enabled.contains("array_index")) applyArrayIndexIdentity(cu, rw);
        if (enabled.contains("identity_math")) applyIdentityMath(cu, rw);
        if (enabled.contains("string_concat_alt")) applyStringConcatAlternatives(cu, rw);
        if (enabled.contains("numeric_literal_alt")) applyNumericLiteralAlternatives(cu, rw);

        Document doc = new Document(source);
        TextEdit edit = rw.rewriteAST(doc, null);
        edit.apply(doc);
        Files.createDirectories(out.getParent());
        Files.writeString(out, doc.get());
    }

    private static String require(Map<String, String> kv, String key) {
        String v = kv.get(key);
        if (v == null) throw new IllegalArgumentException("Missing arg: " + key);
        return v;
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    m.put(args[i], args[i + 1]);
                    i++;
                } else {
                    m.put(args[i], "true");
                }
            }
        }
        return m;
    }

    private static Set<String> parseCsv(String s) {
        if (s == null || s.isBlank()) return Collections.emptySet();
        return Arrays.stream(s.split(",")).map(String::trim).filter(t -> !t.isEmpty())
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    private static void applyGuardReversal(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(IfStatement node) {
                Statement thenS = node.getThenStatement();
                Statement elseS = node.getElseStatement();
                if (thenS == null || elseS == null) return true;
                AST ast = cu.getAST();
                Expression neg = negateCondition(ast, node.getExpression());
                IfStatement repl = ast.newIfStatement();
                repl.setExpression((Expression) ASTNode.copySubtree(ast, neg));
                repl.setThenStatement((Statement) ASTNode.copySubtree(ast, elseS));
                repl.setElseStatement((Statement) ASTNode.copySubtree(ast, thenS));
                rw.replace(node, repl, null);
                return true;
            }
        });
    }

    private static Expression negateCondition(AST ast, Expression expr) {
        if (expr instanceof ParenthesizedExpression pe) return negateCondition(ast, pe.getExpression());
        if (expr instanceof PrefixExpression px && px.getOperator() == PrefixExpression.Operator.NOT) {
            return (Expression) ASTNode.copySubtree(ast, px.getOperand());
        }
        if (expr instanceof InfixExpression ix) {
            InfixExpression repl = ast.newInfixExpression();
            repl.setLeftOperand((Expression) ASTNode.copySubtree(ast, ix.getLeftOperand()));
            repl.setRightOperand((Expression) ASTNode.copySubtree(ast, ix.getRightOperand()));
            repl.setOperator(switch (ix.getOperator().toString()) {
                case "==" -> InfixExpression.Operator.NOT_EQUALS;
                case "!=" -> InfixExpression.Operator.EQUALS;
                case "<" -> InfixExpression.Operator.GREATER_EQUALS;
                case ">" -> InfixExpression.Operator.LESS_EQUALS;
                case "<=" -> InfixExpression.Operator.GREATER;
                case ">=" -> InfixExpression.Operator.LESS;
                default -> null;
            });
            if (repl.getOperator() != null) return repl;
        }
        PrefixExpression not = ast.newPrefixExpression();
        not.setOperator(PrefixExpression.Operator.NOT);
        not.setOperand((Expression) ASTNode.copySubtree(ast, expr));
        return not;
    }

    private static void applyDeMorgan(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(PrefixExpression node) {
                if (node.getOperator() != PrefixExpression.Operator.NOT) return true;
                if (node.getOperand() instanceof ParenthesizedExpression pe && pe.getExpression() instanceof InfixExpression ix) {
                    if (ix.getOperator() == InfixExpression.Operator.CONDITIONAL_AND || ix.getOperator() == InfixExpression.Operator.CONDITIONAL_OR) {
                        AST ast = cu.getAST();
                        InfixExpression repl = ast.newInfixExpression();
                        repl.setOperator(ix.getOperator() == InfixExpression.Operator.CONDITIONAL_AND
                                ? InfixExpression.Operator.CONDITIONAL_OR
                                : InfixExpression.Operator.CONDITIONAL_AND);
                        PrefixExpression leftNot = ast.newPrefixExpression();
                        leftNot.setOperator(PrefixExpression.Operator.NOT);
                        leftNot.setOperand((Expression) ASTNode.copySubtree(ast, ix.getLeftOperand()));
                        PrefixExpression rightNot = ast.newPrefixExpression();
                        rightNot.setOperator(PrefixExpression.Operator.NOT);
                        rightNot.setOperand((Expression) ASTNode.copySubtree(ast, ix.getRightOperand()));
                        repl.setLeftOperand(leftNot);
                        repl.setRightOperand(rightNot);
                        rw.replace(node, repl, null);
                    }
                }
                return true;
            }
        });
    }

    private static void applyTernaryIfElse(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(IfStatement node) {
                Statement t = node.getThenStatement();
                Statement e = node.getElseStatement();
                if (t == null || e == null) return true;
                ExpressionStatement te = stmtAsExpr(t);
                ExpressionStatement ee = stmtAsExpr(e);
                if (te == null || ee == null) return true;
                Assignment ta = (te.getExpression() instanceof Assignment a) ? a : null;
                Assignment ea = (ee.getExpression() instanceof Assignment a) ? a : null;
                if (ta == null || ea == null) return true;
                if (!ta.getLeftHandSide().toString().equals(ea.getLeftHandSide().toString())) return true;

                AST ast = cu.getAST();
                ConditionalExpression cond = ast.newConditionalExpression();
                cond.setExpression((Expression) ASTNode.copySubtree(ast, node.getExpression()));
                cond.setThenExpression((Expression) ASTNode.copySubtree(ast, ta.getRightHandSide()));
                cond.setElseExpression((Expression) ASTNode.copySubtree(ast, ea.getRightHandSide()));

                Assignment asg = ast.newAssignment();
                asg.setLeftHandSide((Expression) ASTNode.copySubtree(ast, ta.getLeftHandSide()));
                asg.setRightHandSide(cond);
                rw.replace(node, ast.newExpressionStatement(asg), null);
                return true;
            }
        });
    }

    private static ExpressionStatement stmtAsExpr(Statement s) {
        if (s instanceof Block b && !b.statements().isEmpty() && b.statements().get(0) instanceof Statement st) {
            return (st instanceof ExpressionStatement es) ? es : null;
        }
        return (s instanceof ExpressionStatement es) ? es : null;
    }

    private static void applyArrayIndexIdentity(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(ArrayAccess node) {
                AST ast = cu.getAST();
                InfixExpression plus = ast.newInfixExpression();
                plus.setOperator(InfixExpression.Operator.PLUS);
                plus.setLeftOperand(ast.newNumberLiteral("0"));
                plus.setRightOperand((Expression) ASTNode.copySubtree(ast, node.getIndex()));
                ArrayAccess repl = ast.newArrayAccess();
                repl.setArray((Expression) ASTNode.copySubtree(ast, node.getArray()));
                repl.setIndex(plus);
                rw.replace(node, repl, null);
                return true;
            }
        });
    }

    private static void applyIdentityMath(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(InfixExpression node) {
                AST ast = cu.getAST();
                if (node.getOperator() == InfixExpression.Operator.PLUS && isZero(node.getRightOperand())) {
                    rw.replace(node, ASTNode.copySubtree(ast, node.getLeftOperand()), null);
                } else if (node.getOperator() == InfixExpression.Operator.PLUS && isZero(node.getLeftOperand())) {
                    rw.replace(node, ASTNode.copySubtree(ast, node.getRightOperand()), null);
                } else if (node.getOperator() == InfixExpression.Operator.TIMES && isOne(node.getRightOperand())) {
                    rw.replace(node, ASTNode.copySubtree(ast, node.getLeftOperand()), null);
                } else if (node.getOperator() == InfixExpression.Operator.TIMES && isOne(node.getLeftOperand())) {
                    rw.replace(node, ASTNode.copySubtree(ast, node.getRightOperand()), null);
                } else if (node.getOperator() == InfixExpression.Operator.MINUS && isZero(node.getRightOperand())) {
                    rw.replace(node, ASTNode.copySubtree(ast, node.getLeftOperand()), null);
                }
                return true;
            }
        });
    }

    private static boolean isZero(Expression e) { return e instanceof NumberLiteral nl && nl.getToken().equals("0"); }
    private static boolean isOne(Expression e) { return e instanceof NumberLiteral nl && nl.getToken().equals("1"); }

    private static void applyStringConcatAlternatives(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(MethodInvocation node) {
                if (!"valueOf".equals(node.getName().getIdentifier())) return true;
                if (!(node.getExpression() instanceof SimpleName sn) || !"String".equals(sn.getIdentifier())) return true;
                if (node.arguments().size() != 1) return true;
                AST ast = cu.getAST();
                InfixExpression plus = ast.newInfixExpression();
                plus.setOperator(InfixExpression.Operator.PLUS);
                StringLiteral empty = ast.newStringLiteral();
                empty.setLiteralValue("");
                plus.setLeftOperand(empty);
                plus.setRightOperand((Expression) ASTNode.copySubtree(ast, (ASTNode) node.arguments().get(0)));
                rw.replace(node, plus, null);
                return true;
            }
        });
    }

    private static void applyNumericLiteralAlternatives(CompilationUnit cu, ASTRewrite rw) {
        cu.accept(new ASTVisitor() {
            @Override public boolean visit(NumberLiteral node) {
                try {
                    long v = Long.decode(node.getToken());
                    if (v == 1000L) {
                        AST ast = cu.getAST();
                        NumberLiteral nl = ast.newNumberLiteral("1_000");
                        rw.replace(node, nl, null);
                    }
                } catch (NumberFormatException ignore) { }
                return true;
            }
        });
    }
}


