/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2001, 2021, Oracle and/or its affiliates. All rights reserved.
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
