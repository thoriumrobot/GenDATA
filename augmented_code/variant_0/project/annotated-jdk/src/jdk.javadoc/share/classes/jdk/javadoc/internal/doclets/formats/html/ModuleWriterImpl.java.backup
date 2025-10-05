/*
 * Copyright (c) 2016, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.formats.html;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import javax.lang.model.element.Element;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.util.ElementFilter;
import com.sun.source.doctree.DeprecatedTree;
import com.sun.source.doctree.DocTree;
import jdk.javadoc.doclet.DocletEnvironment.ModuleMode;
import jdk.javadoc.internal.doclets.formats.html.markup.BodyContents;
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
import jdk.javadoc.internal.doclets.formats.html.Navigation.PageMode;
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.ModuleSummaryWriter;
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;

public class ModuleWriterImpl extends HtmlDocletWriter implements ModuleSummaryWriter {

    protected ModuleElement mdle;

    class PackageEntry {
    }

    public ModuleWriterImpl(HtmlConfiguration configuration, ModuleElement mdle) {
    }

    @Override
    public Content getModuleHeader(String heading);

    @Override
    protected Navigation getNavBar(PageMode pageMode, Element element);

    @Override
    public Content getContentHeader();

    @Override
    public Content getSummariesList();

    @Override
    public Content getSummaryTree(Content summaryContentTree);

    public void computeModulesData();

    public boolean shouldDocument(Element element);

    public boolean display(Set<? extends Element> section);

    public boolean display(Map<? extends Element, ?> section);

    public void addSummaryHeader(Content startMarker, Content heading, Content htmltree);

    @Override
    public void addModulesSummary(Content summariesList);

    @Override
    public void addPackagesSummary(Content summariesList);

    public void addPackageSummary(HtmlTree li);

    public void addIndirectPackages(Table table, Map<ModuleElement, SortedSet<PackageElement>> ip);

    @Override
    public void addServicesSummary(Content summariesList);

    public void addUsesList(Table table);

    public void addProvidesList(Table table);

    public void addDeprecationInfo(Content div);

    @Override
    public void addModuleDescription(Content moduleContentTree);

    @Override
    public void addModuleSignature(Content moduleContentTree);

    @Override
    public void addModuleContent(Content moduleContentTree);

    @Override
    public void addModuleFooter();

    @Override
    public void printDocument(Content contentTree) throws DocFileIOException;

    public void addPackageDeprecationInfo(Content li, PackageElement pkg);
}
