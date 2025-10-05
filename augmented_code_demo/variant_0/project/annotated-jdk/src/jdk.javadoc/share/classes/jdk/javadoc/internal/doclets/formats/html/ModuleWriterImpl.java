/*
    @Positive
 * Copyright (c) 2016, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.TreeSet;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.util.ElementFilter;
    @Positive
import com.sun.source.doctree.DeprecatedTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import jdk.javadoc.doclet.DocletEnvironment.ModuleMode;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.BodyContents;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.Navigation.PageMode;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.ModuleSummaryWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;

    @Positive
public class ModuleWriterImpl extends HtmlDocletWriter implements ModuleSummaryWriter {

    @Positive
    protected ModuleElement mdle;

    @Positive
    class PackageEntry {
    @Positive
    }

    @Positive
    public ModuleWriterImpl(HtmlConfiguration configuration, ModuleElement mdle) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Content getModuleHeader(String heading);

    @Positive
    @Override
    @Positive
    protected Navigation getNavBar(PageMode pageMode, Element element);

    @Positive
    @Override
    @Positive
    public Content getContentHeader();

    @Positive
    @Override
    @Positive
    public Content getSummariesList();

    @Positive
    @Override
    @Positive
    public Content getSummaryTree(Content summaryContentTree);

    @Positive
    public void computeModulesData();

    @Positive
    public boolean shouldDocument(Element element);

    @Positive
    public boolean display(Set<? extends Element> section);

    @Positive
    public boolean display(Map<? extends Element, ?> section);

    @Positive
    public void addSummaryHeader(Content startMarker, Content heading, Content htmltree);

    @Positive
    @Override
    @Positive
    public void addModulesSummary(Content summariesList);

    @Positive
    @Override
    @Positive
    public void addPackagesSummary(Content summariesList);

    @Positive
    public void addPackageSummary(HtmlTree li);

    @Positive
    public void addIndirectPackages(Table table, Map<ModuleElement, SortedSet<PackageElement>> ip);

    @Positive
    @Override
    @Positive
    public void addServicesSummary(Content summariesList);

    @Positive
    public void addUsesList(Table table);

    @Positive
    public void addProvidesList(Table table);

    @Positive
    public void addDeprecationInfo(Content div);

    @Positive
    @Override
    @Positive
    public void addModuleDescription(Content moduleContentTree);

    @Positive
    @Override
    @Positive
    public void addModuleSignature(Content moduleContentTree);

    @Positive
    @Override
    @Positive
    public void addModuleContent(Content moduleContentTree);

    @Positive
    @Override
    @Positive
    public void addModuleFooter();

    @Positive
    @Override
    @Positive
    public void printDocument(Content contentTree) throws DocFileIOException;

    @Positive
    public void addPackageDeprecationInfo(Content li, PackageElement pkg);
    @Positive
}

// CFWR semantic augmentation - variant 0
