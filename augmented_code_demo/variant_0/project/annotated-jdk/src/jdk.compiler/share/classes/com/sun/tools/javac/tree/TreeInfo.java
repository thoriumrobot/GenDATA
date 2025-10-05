/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package com.sun.tools.javac.tree;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.source.tree.Tree;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.code.*;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.tree.JCTree.*;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCPolyExpression.*;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import com.sun.tools.javac.code.Symbol.VarSymbol;
    @Positive
import static com.sun.tools.javac.code.TypeTag.BOOLEAN;
    @Positive
import static com.sun.tools.javac.code.TypeTag.BOT;
    @Positive
import static com.sun.tools.javac.tree.JCTree.Tag.*;
    @Positive
import static com.sun.tools.javac.tree.JCTree.Tag.BLOCK;
    @Positive
import static com.sun.tools.javac.tree.JCTree.Tag.SYNCHRONIZED;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import java.util.function.ToIntFunction;
    @Positive
import static com.sun.tools.javac.tree.JCTree.JCOperatorExpression.OperandPos.LEFT;
    @Positive
import static com.sun.tools.javac.tree.JCTree.JCOperatorExpression.OperandPos.RIGHT;

    @Positive
public class TreeInfo {

    @Positive
    public static List<JCExpression> args(JCTree t);

    @Positive
    public static boolean isConstructor(JCTree tree);

    @Positive
    public static boolean isCanonicalConstructor(JCTree tree);

    @Positive
    public static boolean isCompactConstructor(JCTree tree);

    @Positive
    public static boolean isReceiverParam(JCTree tree);

    @Positive
    public static boolean hasConstructors(List<JCTree> trees);

    @Positive
    public static Name getConstructorInvocationName(List<? extends JCTree> trees, Names names);

    @Positive
    public static boolean isMultiCatch(JCCatch catchClause);

    @Positive
    public static boolean isSyntheticInit(JCTree stat);

    @Positive
    public static Name calledMethodName(JCTree tree);

    @Positive
    public static boolean isSelfCall(JCTree tree);

    @Positive
    public static boolean isThisQualifier(JCTree tree);

    @Positive
    public static boolean isIdentOrThisDotIdent(JCTree tree);

    @Positive
    public static boolean isSuperCall(JCTree tree);

    @Positive
    public static List<JCVariableDecl> recordFields(JCClassDecl tree);

    @Positive
    public static List<Type> recordFieldTypes(JCClassDecl tree);

    @Positive
    public static boolean isInitialConstructor(JCTree tree);

    @Positive
    public static JCMethodInvocation firstConstructorCall(JCTree tree);

    @Positive
    public static boolean isDiamond(JCTree tree);

    @Positive
    public static boolean isEnumInit(JCTree tree);

    @Positive
    public static void setPolyKind(JCTree tree, PolyKind pkind);

    @Positive
    public static void setVarargsElement(JCTree tree, Type varargsElement);

    @Positive
    public static boolean isExpressionStatement(JCExpression tree);

    @Positive
    public static boolean isStatement(JCTree tree);

    @Positive
    public static boolean isStaticSelector(JCTree base, Names names);

    @Positive
    public static boolean isNull(JCTree tree);

    @Positive
    public static boolean isInAnnotation(Env<?> env, JCTree tree);

    @Positive
    public static String getCommentText(Env<?> env, JCTree tree);

    @Positive
    public static DCTree.DCDocComment getCommentTree(Env<?> env, JCTree tree);

    @Positive
    public static int firstStatPos(JCTree tree);

    @Positive
    public static int endPos(JCTree tree);

    @Positive
    public static int getStartPos(JCTree tree);

    @Positive
    public static int getEndPos(JCTree tree, EndPosTable endPosTable);

    @Positive
    public static DiagnosticPosition diagEndPos(final JCTree tree);

    @Positive
    public enum PosKind {

    @Positive
        START_POS(TreeInfo::getStartPos), FIRST_STAT_POS(TreeInfo::firstStatPos), END_POS(TreeInfo::endPos);

    @Positive
        int toPos(JCTree tree);
    @Positive
    }

    @Positive
    public static int finalizerPos(JCTree tree, PosKind posKind);

    @Positive
    public static int positionFor(final Symbol sym, final JCTree tree);

    @Positive
    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final JCTree tree);

    @Positive
    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final JCTree tree, boolean returnNullIfNotFound);

    @Positive
    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final List<? extends JCTree> trees);

    @Positive
    private static class DeclScanner extends TreeScanner {

    @Positive
        public void scan(JCTree tree);

    @Positive
        public void visitTopLevel(JCCompilationUnit that);

    @Positive
        public void visitModuleDef(JCModuleDecl that);

    @Positive
        public void visitPackageDef(JCPackageDecl that);

    @Positive
        public void visitClassDef(JCClassDecl that);

    @Positive
        public void visitMethodDef(JCMethodDecl that);

    @Positive
        public void visitVarDef(JCVariableDecl that);

    @Positive
        public void visitTypeParameter(JCTypeParameter that);
    @Positive
    }

    @Positive
    public static JCTree declarationFor(final Symbol sym, final JCTree tree);

    @Positive
    public static Env<AttrContext> scopeFor(JCTree node, JCCompilationUnit unit);

    @Positive
    public static Env<AttrContext> scopeFor(List<JCTree> path);

    @Positive
    public static List<JCTree> pathFor(final JCTree node, final JCCompilationUnit unit);

    @Positive
    public static JCTree referencedStatement(JCLabeledStatement tree);

    @Positive
    public static JCExpression skipParens(JCExpression tree);

    @Positive
    public static JCTree skipParens(JCTree tree);

    @Positive
    public static List<Type> types(List<? extends JCTree> trees);

    @Positive
    public static Name name(JCTree tree);

    @Positive
    public static Name fullName(JCTree tree);

    @Positive
    public static Symbol symbolFor(JCTree node);

    @Positive
    public static boolean isDeclaration(JCTree node);

    @Positive
    public static Symbol symbol(JCTree tree);

    @Positive
    public static JCModifiers getModifiers(JCTree tree);

    @Positive
    public static boolean nonstaticSelect(JCTree tree);

    @Positive
    public static void setSymbol(JCTree tree, Symbol sym);

    @Positive
    public static long flags(JCTree tree);

    @Positive
    public static long firstFlag(long flags);

    @Positive
    public static String flagNames(long flags);

    @Positive
    public static final int notExpression, noPrec, assignPrec, assignopPrec, condPrec, orPrec, andPrec, bitorPrec, bitxorPrec, bitandPrec, eqPrec, ordPrec, shiftPrec, addPrec, mulPrec, prefixPrec, postfixPrec, precCount;

    @Positive
    public static int opPrec(JCTree.Tag op);

    @Positive
    static Tree.Kind tagToKind(JCTree.Tag tag);

    @Positive
    public static JCExpression typeIn(JCExpression tree);

    @Positive
    public static JCTree innermostType(JCTree type, boolean skipAnnos);

    @Positive
    private static class TypeAnnotationFinder extends TreeScanner {

    @Positive
        public boolean foundTypeAnno;

    @Positive
        @Override
    @Positive
        public void scan(JCTree tree);

    @Positive
        public void visitAnnotation(JCAnnotation tree);
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public static boolean containsTypeAnnotation(JCTree e);

    @Positive
    public static boolean isModuleInfo(JCCompilationUnit tree);

    @Positive
    public static JCModuleDecl getModule(JCCompilationUnit t);

    @Positive
    public static boolean isPackageInfo(JCCompilationUnit tree);

    @Positive
    public static boolean isErrorEnumSwitch(JCExpression selector, List<JCCase> cases);

    @Positive
    public static PatternPrimaryType primaryPatternType(JCPattern pat);

    @Positive
    public record PatternPrimaryType(Type type, boolean unconditional) {
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
