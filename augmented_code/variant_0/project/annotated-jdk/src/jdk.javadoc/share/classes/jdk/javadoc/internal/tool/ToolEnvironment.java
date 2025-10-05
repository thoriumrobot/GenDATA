/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.*;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.JavaFileObject.Kind;
    @Positive
import com.sun.source.util.DocTrees;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.api.JavacTrees;
    @Positive
import com.sun.tools.javac.code.ClassFinder;
    @Positive
import com.sun.tools.javac.code.Flags;
    @Positive
import com.sun.tools.javac.code.Source;
    @Positive
import com.sun.tools.javac.code.Symbol;
    @Positive
import com.sun.tools.javac.code.Symbol.ClassSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.CompletionFailure;
    @Positive
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
    @Positive
import com.sun.tools.javac.code.Symtab;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Check;
    @Positive
import com.sun.tools.javac.comp.Enter;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.file.JavacFileManager;
    @Positive
import com.sun.tools.javac.model.JavacElements;
    @Positive
import com.sun.tools.javac.model.JavacTypes;
    @Positive
import com.sun.tools.javac.tree.JCTree;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCPackageDecl;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.Convert;
    @Positive
import com.sun.tools.javac.util.Name;
    @Positive
import com.sun.tools.javac.util.Names;

    @Positive
public class ToolEnvironment {

    @Positive
    protected static final Context.Key<ToolEnvironment> ToolEnvKey;

    @Positive
    public static ToolEnvironment instance(Context context);

    @Positive
    public final Symtab syms;

    @Positive
    public final Context context;

    @Positive
    public final Source source;

    @Positive
    public final Elements elements;

    @Positive
    public final JavacTypes typeutils;

    @Positive
    protected DocEnvImpl docEnv;

    @Positive
    public final DocTrees docTrees;

    @Positive
    public final Map<Element, TreePath> elementToTreePath;

    @Positive
    protected ToolEnvironment(Context context) {
    @Positive
    }

    @Positive
    public void initialize(ToolOptions options);

    @Positive
    public TypeElement loadClass(String name);

    @Positive
    @Pure
    @Positive
    boolean isSynthetic(Symbol sym);

    @Positive
    void setElementToTreePath(Element e, TreePath tree);

    @Positive
    public Kind getFileKind(TypeElement te);

    @Positive
    public void notice(String key);

    @Positive
    public void notice(String key, String a1);

    @Positive
    TreePath getTreePath(JCCompilationUnit tree);

    @Positive
    TreePath getTreePath(JCCompilationUnit toplevel, JCPackageDecl tree);

    @Positive
    TreePath getTreePath(JCCompilationUnit toplevel, JCClassDecl tree);

    @Positive
    TreePath getTreePath(JCCompilationUnit toplevel, JCClassDecl cdecl, JCTree tree);

    @Positive
    public com.sun.tools.javac.code.Types getTypes();

    @Positive
    public Env<AttrContext> getEnv(ClassSymbol tsym);

    @Positive
    @Pure
    @Positive
    public boolean isQuiet();
    @Positive
}
