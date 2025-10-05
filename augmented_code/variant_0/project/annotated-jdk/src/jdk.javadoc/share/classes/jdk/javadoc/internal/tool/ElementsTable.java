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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import java.io.IOException;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumMap;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.ModuleElement.ExportsDirective;
    @Positive
import javax.lang.model.element.ModuleElement.RequiresDirective;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.util.ElementFilter;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardLocation;
    @Positive
import com.sun.tools.javac.code.Kinds.Kind;
    @Positive
import com.sun.tools.javac.code.Source;
    @Positive
import com.sun.tools.javac.code.Source.Feature;
    @Positive
import com.sun.tools.javac.code.Symbol;
    @Positive
import com.sun.tools.javac.code.Symbol.ClassSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.CompletionFailure;
    @Positive
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
    @Positive
import com.sun.tools.javac.code.Symbol.PackageSymbol;
    @Positive
import com.sun.tools.javac.code.Symtab;
    @Positive
import com.sun.tools.javac.comp.Modules;
    @Positive
import com.sun.tools.javac.main.JavaCompiler;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCModuleDecl;
    @Positive
import com.sun.tools.javac.tree.TreeInfo;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.ListBuffer;
    @Positive
import com.sun.tools.javac.util.Name;
    @Positive
import com.sun.tools.javac.util.Names;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment.ModuleMode;
    @Positive
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
    @Positive
import static javax.lang.model.util.Elements.Origin.*;
    @Positive
import static javax.tools.JavaFileObject.Kind.*;
    @Positive
import static jdk.javadoc.internal.tool.Main.Result.*;
    @Positive
import static jdk.javadoc.internal.tool.JavadocTool.isValidClassName;

    @Positive
public class ElementsTable {

    @Positive
    public ModuleMode getModuleMode();

    @Positive
    public Set<? extends Element> getSpecifiedElements();

    @Positive
    public Set<? extends Element> getIncludedElements();

    @Positive
    @Pure
    @Positive
    public boolean isIncluded(Element e);

    @Positive
    void analyze() throws ToolException;

    @Positive
    ElementsTable classTrees(com.sun.tools.javac.util.List<JCCompilationUnit> classTrees);

    @Positive
    void sanityCheckSourcePathModules(List<String> moduleNames) throws ToolException;

    @Positive
    ElementsTable scanSpecifiedItems() throws ToolException;

    @Positive
    ElementsTable setClassArgList(List<String> classList);

    @Positive
    ElementsTable setClassDeclList(List<JCClassDecl> classesDecList);

    @Positive
    ElementsTable packages(Collection<String> packageNames);

    @Positive
    Iterable<ModulePackage> getPackagesToParse() throws IOException;

    @Positive
    List<JavaFileObject> getFilesToParse() throws ToolException;

    @Positive
    @Pure
    @Positive
    public boolean isSelected(Element e);

    @Positive
    private class IncludedVisitor extends SimpleElementVisitor14<Boolean, Void> {

    @Positive
        public IncludedVisitor() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Boolean visitModule(ModuleElement e, Void p);

    @Positive
        @Override
    @Positive
        public Boolean visitPackage(PackageElement e, Void p);

    @Positive
        @Override
    @Positive
        public Boolean visitType(TypeElement e, Void p);

    @Positive
        @Override
    @Positive
        public Boolean defaultAction(Element e, Void p);

    @Positive
        @Override
    @Positive
        public Boolean visitUnknown(Element e, Void p);
    @Positive
    }

    @Positive
    class Entry {

    @Positive
        @Pure
    @Positive
        boolean isExcluded();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static class ModulePackage {

    @Positive
        public final String moduleName;

    @Positive
        public final String packageName;

    @Positive
        boolean hasModule();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static class ModifierFilter {

    @Positive
        static EnumSet<AccessKind> getFilterSet(AccessKind accessValue);

    @Positive
        public AccessKind getAccessValue(ElementKind kind);

    @Positive
        public boolean checkModifier(Element e);
    @Positive
    }
    @Positive
}
