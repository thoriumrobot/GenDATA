/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.TreeSet;
    @Positive
import javax.lang.model.element.AnnotationMirror;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.lang.model.util.Types;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.code.Attribute;
    @Positive
import com.sun.tools.javac.code.Flags;
    @Positive
import com.sun.tools.javac.code.Scope;
    @Positive
import com.sun.tools.javac.code.Symbol;
    @Positive
import com.sun.tools.javac.code.Symbol.ClassSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.MethodSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.PackageSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.VarSymbol;
    @Positive
import com.sun.tools.javac.code.TypeTag;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.model.JavacElements;
    @Positive
import com.sun.tools.javac.util.Names;
    @Positive
import com.sun.tools.javac.util.Options;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import jdk.javadoc.internal.tool.ToolEnvironment;
    @Positive
import jdk.javadoc.internal.tool.DocEnvImpl;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
    @Positive
import static javax.lang.model.element.ElementKind.*;

    @Positive
public class WorkArounds {

    @Positive
    public final BaseConfiguration configuration;

    @Positive
    public final ToolEnvironment toolEnv;

    @Positive
    public final Utils utils;

    @Positive
    public final Elements elementUtils;

    @Positive
    public final Types typeUtils;

    @Positive
    public final com.sun.tools.javac.code.Types javacTypes;

    @Positive
    public WorkArounds(BaseConfiguration configuration) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean isDeprecated0(Element e);

    @Positive
    @Pure
    @Positive
    public boolean isSynthesized(AnnotationMirror aDesc);

    @Positive
    public Map<Element, TreePath> getElementToTreePath();

    @Positive
    FileObject getJavaFileObject(PackageElement packageElement);

    @Positive
    public TypeElement searchClass(TypeElement klass, String className);

    @Positive
    public TypeMirror overriddenType(ExecutableElement method);

    @Positive
    public boolean overrides(ExecutableElement e1, ExecutableElement e2, TypeElement cls);

    @Positive
    public Location getLocationForModule(ModuleElement mdle);

    @Positive
    public SortedSet<VariableElement> getSerializableFields(TypeElement typeElem);

    @Positive
    public SortedSet<ExecutableElement> getSerializationMethods(TypeElement typeElem);

    @Positive
    public boolean definesSerializableFields(TypeElement typeElem);

    @Positive
    static class NewSerializedForm {

    @Positive
        public ExecutableElement findMethod(TypeElement te, String methodName, List<String> paramTypes);
    @Positive
    }

    @Positive
    public PackageElement getAbbreviatedPackageElement(PackageElement pkg);

    @Positive
    public boolean isPreviewAPI(Element el);

    @Positive
    public boolean isReflectivePreviewAPI(Element el);

    @Positive
    public boolean accessInternalAPI();
    @Positive
}
