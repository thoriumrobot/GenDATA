/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.tool;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.code.Flags;
    @Positive
import com.sun.tools.javac.code.Symbol.*;
    @Positive
import com.sun.tools.javac.comp.MemberEnter;
    @Positive
import com.sun.tools.javac.tree.JCTree;
    @Positive
import com.sun.tools.javac.tree.JCTree.*;
    @Positive
import com.sun.tools.javac.tree.TreeInfo;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.List;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;

    @Positive
public class JavadocMemberEnter extends MemberEnter {

    @Positive
    public static JavadocMemberEnter instance0(Context context);

    @Positive
    public static void preRegister(Context context);

    @Positive
    protected JavadocMemberEnter(Context context) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void visitMethodDef(JCMethodDecl tree);

    @Positive
    @Override
    @Positive
    public void visitVarDef(JCVariableDecl tree);

    @Positive
    private static class MaybeConstantExpressionScanner extends JCTree.Visitor {

    @Positive
        @Pure
    @Positive
        public boolean containsNonConstantExpression(JCExpression tree);

    @Positive
        public void scan(JCTree tree);

    @Positive
        @Override
    @Positive
        public void visitTree(JCTree tree);

    @Positive
        @Override
    @Positive
        public void visitBinary(JCBinary tree);

    @Positive
        @Override
    @Positive
        public void visitConditional(JCConditional tree);

    @Positive
        @Override
    @Positive
        public void visitIdent(JCIdent tree);

    @Positive
        @Override
    @Positive
        public void visitLiteral(JCLiteral tree);

    @Positive
        @Override
    @Positive
        public void visitParens(JCParens tree);

    @Positive
        @Override
    @Positive
        public void visitSelect(JCTree.JCFieldAccess tree);

    @Positive
        @Override
    @Positive
        public void visitTypeCast(JCTypeCast tree);

    @Positive
        @Override
    @Positive
        public void visitTypeIdent(JCPrimitiveTypeTree tree);

    @Positive
        @Override
    @Positive
        public void visitUnary(JCUnary tree);
    @Positive
    }
    @Positive
}
