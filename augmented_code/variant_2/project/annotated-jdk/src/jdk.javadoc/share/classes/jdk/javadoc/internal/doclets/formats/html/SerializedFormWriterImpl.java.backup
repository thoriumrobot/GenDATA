/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Set;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
import jdk.javadoc.internal.doclets.formats.html.Navigation.PageMode;
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.SerializedFormWriter;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;
import jdk.javadoc.internal.doclets.toolkit.util.IndexItem;

public class SerializedFormWriterImpl extends SubWriterHolderWriter implements SerializedFormWriter {

    public SerializedFormWriterImpl(HtmlConfiguration configuration) {
    }

    @Override
    public Content getHeader(String header);

    @Override
    public Content getSerializedSummariesHeader();

    @Override
    public Content getPackageSerializedHeader();

    @Override
    public Content getPackageHeader(PackageElement packageElement);

    @Override
    public Content getClassSerializedHeader();

    @Pure
    public boolean isVisibleClass(TypeElement typeElement);

    @Override
    public Content getClassHeader(TypeElement typeElement);

    @Override
    public Content getSerialUIDInfoHeader();

    @Override
    public void addSerialUIDInfo(String header, String serialUID, Content serialUidTree);

    @Override
    public Content getClassContentHeader();

    @Override
    public void addSerializedContent(Content serializedTreeContent);

    @Override
    public void addPackageSerializedTree(Content serializedSummariesTree, Content packageSerializedTree);

    @Override
    public void addFooter();

    @Override
    public void printDocument(Content serializedTree) throws DocFileIOException;

    @Override
    public SerialFieldWriter getSerialFieldWriter(TypeElement typeElement);

    @Override
    public SerialMethodWriter getSerialMethodWriter(TypeElement typeElement);
}
