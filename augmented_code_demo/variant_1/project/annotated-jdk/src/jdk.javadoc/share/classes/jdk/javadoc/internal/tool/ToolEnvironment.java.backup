/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.*;
import javax.lang.model.element.Element;
import javax.lang.model.element.TypeElement;
import javax.lang.model.util.Elements;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;
import com.sun.source.util.DocTrees;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.api.JavacTrees;
import com.sun.tools.javac.code.ClassFinder;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Source;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symbol.CompletionFailure;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Check;
import com.sun.tools.javac.comp.Enter;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.model.JavacElements;
import com.sun.tools.javac.model.JavacTypes;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.tree.JCTree.JCPackageDecl;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Convert;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;

public class ToolEnvironment {

    protected static final Context.Key<ToolEnvironment> ToolEnvKey;

    public static ToolEnvironment instance(Context context);

    public final Symtab syms;

    public final Context context;

    public final Source source;

    public final Elements elements;

    public final JavacTypes typeutils;

    protected DocEnvImpl docEnv;

    public final DocTrees docTrees;

    public final Map<Element, TreePath> elementToTreePath;

    protected ToolEnvironment(Context context) {
    }

    public void initialize(ToolOptions options);

    public TypeElement loadClass(String name);

    @Pure
    boolean isSynthetic(Symbol sym);

    void setElementToTreePath(Element e, TreePath tree);

    public Kind getFileKind(TypeElement te);

    public void notice(String key);

    public void notice(String key, String a1);

    TreePath getTreePath(JCCompilationUnit tree);

    TreePath getTreePath(JCCompilationUnit toplevel, JCPackageDecl tree);

    TreePath getTreePath(JCCompilationUnit toplevel, JCClassDecl tree);

    TreePath getTreePath(JCCompilationUnit toplevel, JCClassDecl cdecl, JCTree tree);

    public com.sun.tools.javac.code.Types getTypes();

    public Env<AttrContext> getEnv(ClassSymbol tsym);

    @Pure
    public boolean isQuiet();
}
