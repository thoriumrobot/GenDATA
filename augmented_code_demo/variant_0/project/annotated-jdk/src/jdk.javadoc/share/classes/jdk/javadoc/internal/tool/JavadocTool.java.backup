/*
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import com.sun.tools.javac.code.ClassFinder;
import com.sun.tools.javac.code.DeferredCompletionFailureHandler;
import com.sun.tools.javac.code.Symbol.Completer;
import com.sun.tools.javac.code.Symbol.CompletionFailure;
import com.sun.tools.javac.code.Symbol.PackageSymbol;
import com.sun.tools.javac.comp.Enter;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.util.Abort;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Position;
import jdk.javadoc.doclet.DocletEnvironment;
import static jdk.javadoc.internal.tool.Main.Result.*;

public class JavadocTool extends com.sun.tools.javac.main.JavaCompiler {

    protected JavadocTool(Context context) {
    }

    @Override
    protected boolean keepComments();

    public static JavadocTool make0(Context context);

    public DocletEnvironment getEnvironment(ToolOptions toolOptions, List<String> javaNames, Iterable<? extends JavaFileObject> fileObjects) throws ToolException;

    @Pure
    boolean isValidPackageName(String s);

    @Pure
    public static boolean isValidClassName(String s);

    List<JCClassDecl> listClasses(List<JCCompilationUnit> trees);
}
