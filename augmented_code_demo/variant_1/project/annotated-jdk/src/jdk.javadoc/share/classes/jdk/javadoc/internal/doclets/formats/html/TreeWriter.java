/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.SortedSet;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.BodyContents;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
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
import jdk.javadoc.internal.doclets.toolkit.util.ClassTree;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;

    @Positive
public class TreeWriter extends AbstractTreeWriter {

    @Positive
    protected BodyContents bodyContents;

    @Positive
    public TreeWriter(HtmlConfiguration configuration, DocPath filename, ClassTree classtree) {
    @Positive
    }

    @Positive
    public static void generate(HtmlConfiguration configuration, ClassTree classtree) throws DocFileIOException;

    @Positive
    public void generateTreeFile() throws DocFileIOException;

    @Positive
    protected void addPackageTreeLinks(Content contentTree);

    @Positive
    protected HtmlTree getTreeHeader();
    @Positive
}

// CFWR semantic augmentation - variant 1
