/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
