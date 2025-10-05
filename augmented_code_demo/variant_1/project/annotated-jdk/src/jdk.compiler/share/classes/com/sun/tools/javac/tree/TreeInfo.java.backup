/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.sun.tools.javac.tree;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.source.tree.Tree;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.code.*;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.tree.JCTree.*;
import com.sun.tools.javac.tree.JCTree.JCPolyExpression.*;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import static com.sun.tools.javac.code.TypeTag.BOOLEAN;
import static com.sun.tools.javac.code.TypeTag.BOT;
import static com.sun.tools.javac.tree.JCTree.Tag.*;
import static com.sun.tools.javac.tree.JCTree.Tag.BLOCK;
import static com.sun.tools.javac.tree.JCTree.Tag.SYNCHRONIZED;
import javax.tools.JavaFileObject;
import java.util.function.ToIntFunction;
import static com.sun.tools.javac.tree.JCTree.JCOperatorExpression.OperandPos.LEFT;
import static com.sun.tools.javac.tree.JCTree.JCOperatorExpression.OperandPos.RIGHT;

public class TreeInfo {

    public static List<JCExpression> args(JCTree t);

    public static boolean isConstructor(JCTree tree);

    public static boolean isCanonicalConstructor(JCTree tree);

    public static boolean isCompactConstructor(JCTree tree);

    public static boolean isReceiverParam(JCTree tree);

    public static boolean hasConstructors(List<JCTree> trees);

    public static Name getConstructorInvocationName(List<? extends JCTree> trees, Names names);

    public static boolean isMultiCatch(JCCatch catchClause);

    public static boolean isSyntheticInit(JCTree stat);

    public static Name calledMethodName(JCTree tree);

    public static boolean isSelfCall(JCTree tree);

    public static boolean isThisQualifier(JCTree tree);

    public static boolean isIdentOrThisDotIdent(JCTree tree);

    public static boolean isSuperCall(JCTree tree);

    public static List<JCVariableDecl> recordFields(JCClassDecl tree);

    public static List<Type> recordFieldTypes(JCClassDecl tree);

    public static boolean isInitialConstructor(JCTree tree);

    public static JCMethodInvocation firstConstructorCall(JCTree tree);

    public static boolean isDiamond(JCTree tree);

    public static boolean isEnumInit(JCTree tree);

    public static void setPolyKind(JCTree tree, PolyKind pkind);

    public static void setVarargsElement(JCTree tree, Type varargsElement);

    public static boolean isExpressionStatement(JCExpression tree);

    public static boolean isStatement(JCTree tree);

    public static boolean isStaticSelector(JCTree base, Names names);

    public static boolean isNull(JCTree tree);

    public static boolean isInAnnotation(Env<?> env, JCTree tree);

    public static String getCommentText(Env<?> env, JCTree tree);

    public static DCTree.DCDocComment getCommentTree(Env<?> env, JCTree tree);

    public static int firstStatPos(JCTree tree);

    public static int endPos(JCTree tree);

    public static int getStartPos(JCTree tree);

    public static int getEndPos(JCTree tree, EndPosTable endPosTable);

    public static DiagnosticPosition diagEndPos(final JCTree tree);

    public enum PosKind {

        START_POS(TreeInfo::getStartPos), FIRST_STAT_POS(TreeInfo::firstStatPos), END_POS(TreeInfo::endPos);

        int toPos(JCTree tree);
    }

    public static int finalizerPos(JCTree tree, PosKind posKind);

    public static int positionFor(final Symbol sym, final JCTree tree);

    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final JCTree tree);

    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final JCTree tree, boolean returnNullIfNotFound);

    public static DiagnosticPosition diagnosticPositionFor(final Symbol sym, final List<? extends JCTree> trees);

    private static class DeclScanner extends TreeScanner {

        public void scan(JCTree tree);

        public void visitTopLevel(JCCompilationUnit that);

        public void visitModuleDef(JCModuleDecl that);

        public void visitPackageDef(JCPackageDecl that);

        public void visitClassDef(JCClassDecl that);

        public void visitMethodDef(JCMethodDecl that);

        public void visitVarDef(JCVariableDecl that);

        public void visitTypeParameter(JCTypeParameter that);
    }

    public static JCTree declarationFor(final Symbol sym, final JCTree tree);

    public static Env<AttrContext> scopeFor(JCTree node, JCCompilationUnit unit);

    public static Env<AttrContext> scopeFor(List<JCTree> path);

    public static List<JCTree> pathFor(final JCTree node, final JCCompilationUnit unit);

    public static JCTree referencedStatement(JCLabeledStatement tree);

    public static JCExpression skipParens(JCExpression tree);

    public static JCTree skipParens(JCTree tree);

    public static List<Type> types(List<? extends JCTree> trees);

    public static Name name(JCTree tree);

    public static Name fullName(JCTree tree);

    public static Symbol symbolFor(JCTree node);

    public static boolean isDeclaration(JCTree node);

    public static Symbol symbol(JCTree tree);

    public static JCModifiers getModifiers(JCTree tree);

    public static boolean nonstaticSelect(JCTree tree);

    public static void setSymbol(JCTree tree, Symbol sym);

    public static long flags(JCTree tree);

    public static long firstFlag(long flags);

    public static String flagNames(long flags);

    public static final int notExpression, noPrec, assignPrec, assignopPrec, condPrec, orPrec, andPrec, bitorPrec, bitxorPrec, bitandPrec, eqPrec, ordPrec, shiftPrec, addPrec, mulPrec, prefixPrec, postfixPrec, precCount;

    public static int opPrec(JCTree.Tag op);

    static Tree.Kind tagToKind(JCTree.Tag tag);

    public static JCExpression typeIn(JCExpression tree);

    public static JCTree innermostType(JCTree type, boolean skipAnnos);

    private static class TypeAnnotationFinder extends TreeScanner {

        public boolean foundTypeAnno;

        @Override
        public void scan(JCTree tree);

        public void visitAnnotation(JCAnnotation tree);
    }

    @Pure
    public static boolean containsTypeAnnotation(JCTree e);

    public static boolean isModuleInfo(JCCompilationUnit tree);

    public static JCModuleDecl getModule(JCCompilationUnit t);

    public static boolean isPackageInfo(JCCompilationUnit tree);

    public static boolean isErrorEnumSwitch(JCExpression selector, List<JCCase> cases);

    public static PatternPrimaryType primaryPatternType(JCPattern pat);

    public record PatternPrimaryType(Type type, boolean unconditional) {
    }
}
