/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Function;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.util.Elements;
import javax.lang.model.util.SimpleElementVisitor14;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.util.DocTreePath;
import com.sun.source.util.TreePath;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;
import jdk.javadoc.doclet.Doclet;
import jdk.javadoc.doclet.DocletEnvironment;
import jdk.javadoc.doclet.Reporter;
import jdk.javadoc.doclet.StandardDoclet;
import jdk.javadoc.doclet.Taglet;
import jdk.javadoc.internal.doclets.toolkit.builders.BuilderFactory;
import jdk.javadoc.internal.doclets.toolkit.taglets.TagletManager;
import jdk.javadoc.internal.doclets.toolkit.util.Comparators;
import jdk.javadoc.internal.doclets.toolkit.util.DocFile;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileFactory;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
import jdk.javadoc.internal.doclets.toolkit.util.Extern;
import jdk.javadoc.internal.doclets.toolkit.util.Group;
import jdk.javadoc.internal.doclets.toolkit.util.MetaKeywords;
import jdk.javadoc.internal.doclets.toolkit.util.SimpleDocletException;
import jdk.javadoc.internal.doclets.toolkit.util.TypeElementCatalog;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import jdk.javadoc.internal.doclets.toolkit.util.Utils.Pair;
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberCache;
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
import jdk.javadoc.internal.doclint.DocLint;

public abstract class BaseConfiguration {

    public final Doclet doclet;

    protected BuilderFactory builderFactory;

    public TagletManager tagletManager;

    public MetaKeywords metakeywords;

    public DocletEnvironment docEnv;

    public Utils utils;

    public WorkArounds workArounds;

    public String sourcepath;

    public boolean showModules;

    public TypeElementCatalog typeElementCatalog;

    public final Group group;

    public Extern extern;

    public final Reporter reporter;

    public final Locale locale;

    public abstract Messages getMessages();

    public abstract Resources getDocResources();

    public abstract Runtime.Version getDocletVersion();

    public abstract boolean finishOptionSettings();

    public CommentUtils cmtUtils;

    public SortedSet<PackageElement> packages;

    public OverviewElement overviewElement;

    public DocFileFactory docFileFactory;

    public SortedMap<ModuleElement, Set<PackageElement>> modulePackages;

    public SortedSet<ModuleElement> modules;

    protected static final String sharedResourceBundleName;

    public PropertyUtils propertyUtils;

    public BaseConfiguration(Doclet doclet, Locale locale, Reporter reporter) {
    }

    public abstract BaseOptions getOptions();

    protected void initConfiguration(DocletEnvironment docEnv, Function<String, String> resourceKeyMapper);

    public BuilderFactory getBuilderFactory();

    public Reporter getReporter();

    public Set<ModuleElement> getSpecifiedModuleElements();

    public Set<PackageElement> getSpecifiedPackageElements();

    public Set<TypeElement> getSpecifiedTypeElements();

    public Set<ModuleElement> getIncludedModuleElements();

    public Set<PackageElement> getIncludedPackageElements();

    public Set<TypeElement> getIncludedTypeElements();

    protected boolean finishOptionSettings0() throws DocletException;

    public boolean setOptions() throws DocletException;

    public boolean shouldExcludeDocFileDir(String docfilesubdir);

    public boolean shouldExcludeQualifier(String qualifier);

    public String getClassName(TypeElement te);

    @Pure
    public boolean isGeneratedDoc(TypeElement te);

    public abstract WriterFactory getWriterFactory();

    public abstract Locale getLocale();

    public abstract JavaFileObject getOverviewPath();

    public abstract JavaFileManager getFileManager();

    public abstract boolean showMessage(DocTreePath path, String key);

    public abstract boolean showMessage(Element e, String key);

    private static class Splitter {
    }

    @Pure
    public boolean isAllowScriptInComments();

    public synchronized VisibleMemberTable getVisibleMemberTable(TypeElement te);

    public boolean isJavaFXMode();

    public void runDocLint(TreePath path);

    public void initDocLint(List<String> opts, Set<String> customTagNames);

    public boolean haveDocLint();
}
