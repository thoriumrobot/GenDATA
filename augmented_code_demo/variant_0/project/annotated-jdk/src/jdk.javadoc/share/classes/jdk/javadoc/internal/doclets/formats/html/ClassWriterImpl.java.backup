/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.Element;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.SimpleElementVisitor8;
import com.sun.source.doctree.DeprecatedTree;
import com.sun.source.doctree.DocTree;
import jdk.javadoc.internal.doclets.formats.html.Navigation.PageMode;
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlAttr;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.taglets.ParamTaglet;
import jdk.javadoc.internal.doclets.toolkit.util.ClassTree;
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;

public class ClassWriterImpl extends SubWriterHolderWriter implements ClassWriter {

    protected final TypeElement typeElement;

    protected final ClassTree classtree;

    public ClassWriterImpl(HtmlConfiguration configuration, TypeElement typeElement, ClassTree classTree) {
    }

    @Override
    public Content getHeader(String header);

    @Override
    public Content getClassContentHeader();

    @Override
    protected Navigation getNavBar(PageMode pageMode, Element element);

    @Override
    public void addFooter();

    @Override
    public void printDocument(Content contentTree) throws DocFileIOException;

    @Override
    public Content getClassInfoTreeHeader();

    @Override
    public Content getClassInfo(Content classInfoTree);

    @Override
    protected TypeElement getCurrentPageElement();

    @Override
    public void addClassSignature(Content classInfoTree);

    @Override
    public void addClassDescription(Content classInfoTree);

    @Override
    public void addClassTagInfo(Content classInfoTree);

    @Override
    public void addClassTree(Content classContentTree);

    @Override
    public void addParamInfo(Content classInfoTree);

    @Override
    public void addSubClassInfo(Content classInfoTree);

    @Override
    public void addSubInterfacesInfo(Content classInfoTree);

    @Override
    public void addInterfaceUsageInfo(Content classInfoTree);

    @Override
    public void addImplementedInterfacesInfo(Content classInfoTree);

    @Override
    public void addSuperInterfacesInfo(Content classInfoTree);

    @Override
    public void addNestedClassInfo(final Content classInfoTree);

    @Override
    public void addFunctionalInterfaceInfo(Content classInfoTree);

    @Pure
    public boolean isFunctionalInterface();

    @Override
    public void addClassDeprecationInfo(Content classInfoTree);

    @Override
    public TypeElement getTypeElement();

    public Content getMemberDetailsTree(Content contentTree);
}
