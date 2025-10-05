/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.file.Files;
    @Positive
import java.nio.file.InvalidPathException;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.tools.javac.code.ClassFinder;
    @Positive
import com.sun.tools.javac.code.DeferredCompletionFailureHandler;
    @Positive
import com.sun.tools.javac.code.Symbol.Completer;
    @Positive
import com.sun.tools.javac.code.Symbol.CompletionFailure;
    @Positive
import com.sun.tools.javac.code.Symbol.PackageSymbol;
    @Positive
import com.sun.tools.javac.comp.Enter;
    @Positive
import com.sun.tools.javac.tree.JCTree;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
    @Positive
import com.sun.tools.javac.util.Abort;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.ListBuffer;
    @Positive
import com.sun.tools.javac.util.Position;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;
    @Positive
import static jdk.javadoc.internal.tool.Main.Result.*;

    @Positive
public class JavadocTool extends com.sun.tools.javac.main.JavaCompiler {

    @Positive
    protected JavadocTool(Context context) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    protected boolean keepComments();

    @Positive
    public static JavadocTool make0(Context context);

    @Positive
    public DocletEnvironment getEnvironment(ToolOptions toolOptions, List<String> javaNames, Iterable<? extends JavaFileObject> fileObjects) throws ToolException;

    @Positive
    @Pure
    @Positive
    boolean isValidPackageName(String s);

    @Positive
    @Pure
    @Positive
    public static boolean isValidClassName(String s);

    @Positive
    List<JCClassDecl> listClasses(List<JCCompilationUnit> trees);
    @Positive
}
