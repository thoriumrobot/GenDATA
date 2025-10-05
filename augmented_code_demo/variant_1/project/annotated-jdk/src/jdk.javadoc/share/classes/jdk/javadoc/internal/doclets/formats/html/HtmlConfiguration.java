/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.formats.html;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Date;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.source.util.DocTreePath;
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
import jdk.javadoc.internal.Versions;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseOptions;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.DocletException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Messages;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Resources;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.WriterFactory;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DeprecatedAPIListBuilder;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFile;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.NewAPIBuilder;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.PreviewAPIListBuilder;

    @Positive
public class HtmlConfiguration extends BaseConfiguration {

    @Positive
    public static final String HTML_DEFAULT_CHARSET;

    @Positive
    public final Resources docResources;

    @Positive
    public DocPath topFile;

    @Positive
    public TypeElement currentTypeElement;

    @Positive
    protected HtmlIndexBuilder mainIndex;

    @Positive
    protected DeprecatedAPIListBuilder deprecatedAPIListBuilder;

    @Positive
    protected PreviewAPIListBuilder previewAPIListBuilder;

    @Positive
    protected NewAPIBuilder newAPIPageBuilder;

    @Positive
    public Contents contents;

    @Positive
    protected final Messages messages;

    @Positive
    public DocPaths docPaths;

    @Positive
    public HtmlIds htmlIds;

    @Positive
    public Map<Element, List<DocPath>> localStylesheetMap;

    @Positive
    public enum ConditionalPage {

    @Positive
        CONSTANT_VALUES,
    @Positive
        DEPRECATED,
    @Positive
        PREVIEW,
    @Positive
        SERIALIZED_FORM,
    @Positive
        SYSTEM_PROPERTIES,
    @Positive
        NEW
    @Positive
    }

    @Positive
    public final Set<ConditionalPage> conditionalPages;

    @Positive
    public HtmlConfiguration(Doclet doclet, Locale locale, Reporter reporter) {
    @Positive
    }

    @Positive
    protected void initConfiguration(DocletEnvironment docEnv, Function<String, String> resourceKeyMapper);

    @Positive
    public final Date startTime;

    @Positive
    @Override
    @Positive
    public Runtime.Version getDocletVersion();

    @Positive
    @Override
    @Positive
    public Resources getDocResources();

    @Positive
    public Contents getContents();

    @Positive
    @Override
    @Positive
    public Messages getMessages();

    @Positive
    @Override
    @Positive
    public HtmlOptions getOptions();

    @Positive
    @Override
    @Positive
    public boolean finishOptionSettings();

    @Positive
    protected void setTopFile();

    @Positive
    protected TypeElement getValidClass(List<TypeElement> classes);

    @Positive
    protected boolean checkForDeprecation();

    @Positive
    protected void setCreateOverview();

    @Positive
    @Override
    @Positive
    public WriterFactory getWriterFactory();

    @Positive
    @Override
    @Positive
    public Locale getLocale();

    @Positive
    @Override
    @Positive
    public JavaFileObject getOverviewPath();

    @Positive
    public DocPath getMainStylesheet();

    @Positive
    public List<DocPath> getAdditionalStylesheets();

    @Positive
    @Override
    @Positive
    public JavaFileManager getFileManager();

    @Positive
    @Override
    @Positive
    public boolean showMessage(DocTreePath path, String key);

    @Positive
    @Override
    @Positive
    public boolean showMessage(Element e, String key);

    @Positive
    @Override
    @Positive
    protected boolean finishOptionSettings0() throws DocletException;
    @Positive
}

// CFWR semantic augmentation - variant 1
