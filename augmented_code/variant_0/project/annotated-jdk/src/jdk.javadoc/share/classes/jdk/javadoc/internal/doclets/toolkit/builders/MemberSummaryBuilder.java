/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.builders;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.*;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.util.ElementFilter;
    @Positive
import com.sun.source.doctree.DocCommentTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import com.sun.source.doctree.DocTree.Kind;
    @Positive
import com.sun.source.doctree.SinceTree;
    @Positive
import com.sun.source.doctree.UnknownBlockTagTree;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.MemberSummaryWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.WriterFactory;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFinder;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.CommentUtils;
    @Positive
import static jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable.Kind.*;

    @Positive
public abstract class MemberSummaryBuilder extends AbstractMemberBuilder {

    @Positive
    public static MemberSummaryBuilder getInstance(ClassWriter classWriter, Context context);

    @Positive
    public VisibleMemberTable getVisibleMemberTable();

    @Positive
    public MemberSummaryWriter getMemberSummaryWriter(VisibleMemberTable.Kind kind);

    @Positive
    public SortedSet<Element> members(VisibleMemberTable.Kind kind);

    @Positive
    protected void buildAnnotationTypeOptionalMemberSummary(Content summariesList);

    @Positive
    protected void buildAnnotationTypeRequiredMemberSummary(Content summariesList);

    @Positive
    protected void buildEnumConstantsSummary(Content summariesList);

    @Positive
    protected void buildFieldsSummary(Content summariesList);

    @Positive
    protected void buildPropertiesSummary(Content summariesList);

    @Positive
    protected void buildNestedClassesSummary(Content summariesList);

    @Positive
    protected void buildMethodsSummary(Content summariesList);

    @Positive
    protected void buildConstructorsSummary(Content summariesList);

    @Positive
    static class PropertyHelper {

    @Positive
        public Element getPropertyElement(Element element);

    @Positive
        public ExecutableElement getGetterForProperty(ExecutableElement propertyMethod);

    @Positive
        public ExecutableElement getSetterForProperty(ExecutableElement propertyMethod);
    @Positive
    }
    @Positive
}
