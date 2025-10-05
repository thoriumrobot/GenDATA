/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.builders;

import org.checkerframework.dataflow.qual.Pure;
import java.text.MessageFormat;
import java.util.*;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.util.ElementFilter;
import com.sun.source.doctree.DocCommentTree;
import com.sun.source.doctree.DocTree;
import com.sun.source.doctree.DocTree.Kind;
import com.sun.source.doctree.SinceTree;
import com.sun.source.doctree.UnknownBlockTagTree;
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.MemberSummaryWriter;
import jdk.javadoc.internal.doclets.toolkit.WriterFactory;
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
import jdk.javadoc.internal.doclets.toolkit.util.DocFinder;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
import jdk.javadoc.internal.doclets.toolkit.CommentUtils;
import static jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable.Kind.*;

public abstract class MemberSummaryBuilder extends AbstractMemberBuilder {

    public static MemberSummaryBuilder getInstance(ClassWriter classWriter, Context context);

    public VisibleMemberTable getVisibleMemberTable();

    public MemberSummaryWriter getMemberSummaryWriter(VisibleMemberTable.Kind kind);

    public SortedSet<Element> members(VisibleMemberTable.Kind kind);

    protected void buildAnnotationTypeOptionalMemberSummary(Content summariesList);

    protected void buildAnnotationTypeRequiredMemberSummary(Content summariesList);

    protected void buildEnumConstantsSummary(Content summariesList);

    protected void buildFieldsSummary(Content summariesList);

    protected void buildPropertiesSummary(Content summariesList);

    protected void buildNestedClassesSummary(Content summariesList);

    protected void buildMethodsSummary(Content summariesList);

    protected void buildConstructorsSummary(Content summariesList);

    static class PropertyHelper {

        public Element getPropertyElement(Element element);

        public ExecutableElement getGetterForProperty(ExecutableElement propertyMethod);

        public ExecutableElement getSetterForProperty(ExecutableElement propertyMethod);
    }
}
