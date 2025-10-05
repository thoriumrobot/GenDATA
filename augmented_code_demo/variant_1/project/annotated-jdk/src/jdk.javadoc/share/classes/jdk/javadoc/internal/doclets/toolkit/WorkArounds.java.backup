/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit;

import org.checkerframework.dataflow.qual.Pure;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import javax.tools.FileObject;
import javax.tools.JavaFileManager.Location;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.code.Attribute;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Scope;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symbol.PackageSymbol;
import com.sun.tools.javac.code.Symbol.VarSymbol;
import com.sun.tools.javac.code.TypeTag;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.model.JavacElements;
import com.sun.tools.javac.util.Names;
import com.sun.tools.javac.util.Options;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import jdk.javadoc.internal.tool.ToolEnvironment;
import jdk.javadoc.internal.tool.DocEnvImpl;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
import static javax.lang.model.element.ElementKind.*;

public class WorkArounds {

    public final BaseConfiguration configuration;

    public final ToolEnvironment toolEnv;

    public final Utils utils;

    public final Elements elementUtils;

    public final Types typeUtils;

    public final com.sun.tools.javac.code.Types javacTypes;

    public WorkArounds(BaseConfiguration configuration) {
    }

    @Pure
    public boolean isDeprecated0(Element e);

    @Pure
    public boolean isSynthesized(AnnotationMirror aDesc);

    public Map<Element, TreePath> getElementToTreePath();

    FileObject getJavaFileObject(PackageElement packageElement);

    public TypeElement searchClass(TypeElement klass, String className);

    public TypeMirror overriddenType(ExecutableElement method);

    public boolean overrides(ExecutableElement e1, ExecutableElement e2, TypeElement cls);

    public Location getLocationForModule(ModuleElement mdle);

    public SortedSet<VariableElement> getSerializableFields(TypeElement typeElem);

    public SortedSet<ExecutableElement> getSerializationMethods(TypeElement typeElem);

    public boolean definesSerializableFields(TypeElement typeElem);

    static class NewSerializedForm {

        public ExecutableElement findMethod(TypeElement te, String methodName, List<String> paramTypes);
    }

    public PackageElement getAbbreviatedPackageElement(PackageElement pkg);

    public boolean isPreviewAPI(Element el);

    public boolean isReflectivePreviewAPI(Element el);

    public boolean accessInternalAPI();
}
