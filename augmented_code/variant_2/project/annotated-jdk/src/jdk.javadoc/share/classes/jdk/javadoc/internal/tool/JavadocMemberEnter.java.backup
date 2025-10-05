/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.tool;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Symbol.*;
import com.sun.tools.javac.comp.MemberEnter;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.*;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;

public class JavadocMemberEnter extends MemberEnter {

    public static JavadocMemberEnter instance0(Context context);

    public static void preRegister(Context context);

    protected JavadocMemberEnter(Context context) {
    }

    @Override
    public void visitMethodDef(JCMethodDecl tree);

    @Override
    public void visitVarDef(JCVariableDecl tree);

    private static class MaybeConstantExpressionScanner extends JCTree.Visitor {

        @Pure
        public boolean containsNonConstantExpression(JCExpression tree);

        public void scan(JCTree tree);

        @Override
        public void visitTree(JCTree tree);

        @Override
        public void visitBinary(JCBinary tree);

        @Override
        public void visitConditional(JCConditional tree);

        @Override
        public void visitIdent(JCIdent tree);

        @Override
        public void visitLiteral(JCLiteral tree);

        @Override
        public void visitParens(JCParens tree);

        @Override
        public void visitSelect(JCTree.JCFieldAccess tree);

        @Override
        public void visitTypeCast(JCTypeCast tree);

        @Override
        public void visitTypeIdent(JCPrimitiveTypeTree tree);

        @Override
        public void visitUnary(JCUnary tree);
    }
}
