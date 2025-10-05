/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedMap;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.TreeSet;
    @Positive
import java.util.function.Function;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.util.Elements;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import com.sun.source.tree.CompilationUnitTree;
    @Positive
import com.sun.source.util.DocTreePath;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import jdk.javadoc.doclet.Doclet;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment;
    @Positive
import jdk.javadoc.doclet.Reporter;
    @Positive
import jdk.javadoc.doclet.StandardDoclet;
    @Positive
import jdk.javadoc.doclet.Taglet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.builders.BuilderFactory;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.TagletManager;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Comparators;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFile;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileFactory;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Extern;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Group;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.MetaKeywords;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.SimpleDocletException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.TypeElementCatalog;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils.Pair;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberCache;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
    @Positive
import jdk.javadoc.internal.doclint.DocLint;

    @Positive
public abstract class BaseConfiguration {

    @Positive
    public final Doclet doclet;

    @Positive
    protected BuilderFactory builderFactory;

    @Positive
    public TagletManager tagletManager;

    @Positive
    public MetaKeywords metakeywords;

    @Positive
    public DocletEnvironment docEnv;

    @Positive
    public Utils utils;

    @Positive
    public WorkArounds workArounds;

    @Positive
    public String sourcepath;

    @Positive
    public boolean showModules;

    @Positive
    public TypeElementCatalog typeElementCatalog;

    @Positive
    public final Group group;

    @Positive
    public Extern extern;

    @Positive
    public final Reporter reporter;

    @Positive
    public final Locale locale;

    @Positive
    public abstract Messages getMessages();

    @Positive
    public abstract Resources getDocResources();

    @Positive
    public abstract Runtime.Version getDocletVersion();

    @Positive
    public abstract boolean finishOptionSettings();

    @Positive
    public CommentUtils cmtUtils;

    @Positive
    public SortedSet<PackageElement> packages;

    @Positive
    public OverviewElement overviewElement;

    @Positive
    public DocFileFactory docFileFactory;

    @Positive
    public SortedMap<ModuleElement, Set<PackageElement>> modulePackages;

    @Positive
    public SortedSet<ModuleElement> modules;

    @Positive
    protected static final String sharedResourceBundleName;

    @Positive
    public PropertyUtils propertyUtils;

    @Positive
    public BaseConfiguration(Doclet doclet, Locale locale, Reporter reporter) {
    @Positive
    }

    @Positive
    public abstract BaseOptions getOptions();

    @Positive
    protected void initConfiguration(DocletEnvironment docEnv, Function<String, String> resourceKeyMapper);

    @Positive
    public BuilderFactory getBuilderFactory();

    @Positive
    public Reporter getReporter();

    @Positive
    public Set<ModuleElement> getSpecifiedModuleElements();

    @Positive
    public Set<PackageElement> getSpecifiedPackageElements();

    @Positive
    public Set<TypeElement> getSpecifiedTypeElements();

    @Positive
    public Set<ModuleElement> getIncludedModuleElements();

    @Positive
    public Set<PackageElement> getIncludedPackageElements();

    @Positive
    public Set<TypeElement> getIncludedTypeElements();

    @Positive
    protected boolean finishOptionSettings0() throws DocletException;

    @Positive
    public boolean setOptions() throws DocletException;

    @Positive
    public boolean shouldExcludeDocFileDir(String docfilesubdir);

    @Positive
    public boolean shouldExcludeQualifier(String qualifier);

    @Positive
    public String getClassName(TypeElement te);

    @Positive
    @Pure
    @Positive
    public boolean isGeneratedDoc(TypeElement te);

    @Positive
    public abstract WriterFactory getWriterFactory();

    @Positive
    public abstract Locale getLocale();

    @Positive
    public abstract JavaFileObject getOverviewPath();

    @Positive
    public abstract JavaFileManager getFileManager();

    @Positive
    public abstract boolean showMessage(DocTreePath path, String key);

    @Positive
    public abstract boolean showMessage(Element e, String key);

    @Positive
    private static class Splitter {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean isAllowScriptInComments();

    @Positive
    public synchronized VisibleMemberTable getVisibleMemberTable(TypeElement te);

    @Positive
    public boolean isJavaFXMode();

    @Positive
    public void runDocLint(TreePath path);

    @Positive
    public void initDocLint(List<String> opts, Set<String> customTagNames);

    @Positive
    public boolean haveDocLint();
    @Positive
}

// CFWR semantic augmentation - variant 1
