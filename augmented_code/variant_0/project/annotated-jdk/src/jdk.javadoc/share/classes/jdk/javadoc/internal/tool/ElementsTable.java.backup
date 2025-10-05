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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.ModuleElement.ExportsDirective;
import javax.lang.model.element.ModuleElement.RequiresDirective;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.util.ElementFilter;
import javax.lang.model.util.SimpleElementVisitor14;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.StandardLocation;
import com.sun.tools.javac.code.Kinds.Kind;
import com.sun.tools.javac.code.Source;
import com.sun.tools.javac.code.Source.Feature;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symbol.CompletionFailure;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symbol.PackageSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.comp.Modules;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.tree.JCTree.JCModuleDecl;
import com.sun.tools.javac.tree.TreeInfo;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;
import jdk.javadoc.doclet.DocletEnvironment;
import jdk.javadoc.doclet.DocletEnvironment.ModuleMode;
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
import static javax.lang.model.util.Elements.Origin.*;
import static javax.tools.JavaFileObject.Kind.*;
import static jdk.javadoc.internal.tool.Main.Result.*;
import static jdk.javadoc.internal.tool.JavadocTool.isValidClassName;

public class ElementsTable {

    public ModuleMode getModuleMode();

    public Set<? extends Element> getSpecifiedElements();

    public Set<? extends Element> getIncludedElements();

    @Pure
    public boolean isIncluded(Element e);

    void analyze() throws ToolException;

    ElementsTable classTrees(com.sun.tools.javac.util.List<JCCompilationUnit> classTrees);

    void sanityCheckSourcePathModules(List<String> moduleNames) throws ToolException;

    ElementsTable scanSpecifiedItems() throws ToolException;

    ElementsTable setClassArgList(List<String> classList);

    ElementsTable setClassDeclList(List<JCClassDecl> classesDecList);

    ElementsTable packages(Collection<String> packageNames);

    Iterable<ModulePackage> getPackagesToParse() throws IOException;

    List<JavaFileObject> getFilesToParse() throws ToolException;

    @Pure
    public boolean isSelected(Element e);

    private class IncludedVisitor extends SimpleElementVisitor14<Boolean, Void> {

        public IncludedVisitor() {
        }

        @Override
        public Boolean visitModule(ModuleElement e, Void p);

        @Override
        public Boolean visitPackage(PackageElement e, Void p);

        @Override
        public Boolean visitType(TypeElement e, Void p);

        @Override
        public Boolean defaultAction(Element e, Void p);

        @Override
        public Boolean visitUnknown(Element e, Void p);
    }

    class Entry {

        @Pure
        boolean isExcluded();

        @Override
        public String toString();
    }

    static class ModulePackage {

        public final String moduleName;

        public final String packageName;

        boolean hasModule();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        @Override
        public String toString();
    }

    static class ModifierFilter {

        static EnumSet<AccessKind> getFilterSet(AccessKind accessValue);

        public AccessKind getAccessValue(ElementKind kind);

        public boolean checkModifier(Element e);
    }
}
