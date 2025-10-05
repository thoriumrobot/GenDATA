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
import java.util.Set;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
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
import jdk.javadoc.internal.doclets.toolkit.SerializedFormWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.IndexItem;

    @Positive
public class SerializedFormWriterImpl extends SubWriterHolderWriter implements SerializedFormWriter {

    @Positive
    public SerializedFormWriterImpl(HtmlConfiguration configuration) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Content getHeader(String header);

    @Positive
    @Override
    @Positive
    public Content getSerializedSummariesHeader();

    @Positive
    @Override
    @Positive
    public Content getPackageSerializedHeader();

    @Positive
    @Override
    @Positive
    public Content getPackageHeader(PackageElement packageElement);

    @Positive
    @Override
    @Positive
    public Content getClassSerializedHeader();

    @Positive
    @Pure
    @Positive
    public boolean isVisibleClass(TypeElement typeElement);

    @Positive
    @Override
    @Positive
    public Content getClassHeader(TypeElement typeElement);

    @Positive
    @Override
    @Positive
    public Content getSerialUIDInfoHeader();

    @Positive
    @Override
    @Positive
    public void addSerialUIDInfo(String header, String serialUID, Content serialUidTree);

    @Positive
    @Override
    @Positive
    public Content getClassContentHeader();

    @Positive
    @Override
    @Positive
    public void addSerializedContent(Content serializedTreeContent);

    @Positive
    @Override
    @Positive
    public void addPackageSerializedTree(Content serializedSummariesTree, Content packageSerializedTree);

    @Positive
    @Override
    @Positive
    public void addFooter();

    @Positive
    @Override
    @Positive
    public void printDocument(Content serializedTree) throws DocFileIOException;

    @Positive
    @Override
    @Positive
    public SerialFieldWriter getSerialFieldWriter(TypeElement typeElement);

    @Positive
    @Override
    @Positive
    public SerialMethodWriter getSerialMethodWriter(TypeElement typeElement);
    @Positive
}

// CFWR semantic augmentation - variant 1
