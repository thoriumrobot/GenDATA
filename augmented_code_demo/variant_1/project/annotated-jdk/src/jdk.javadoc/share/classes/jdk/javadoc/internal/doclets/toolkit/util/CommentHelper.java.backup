/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import javax.lang.model.element.Element;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeMirror;
import com.sun.source.doctree.AttributeTree;
import com.sun.source.doctree.AttributeTree.ValueKind;
import com.sun.source.doctree.AuthorTree;
import com.sun.source.doctree.BlockTagTree;
import com.sun.source.doctree.CommentTree;
import com.sun.source.doctree.DeprecatedTree;
import com.sun.source.doctree.DocCommentTree;
import com.sun.source.doctree.DocTree;
import com.sun.source.doctree.EndElementTree;
import com.sun.source.doctree.EntityTree;
import com.sun.source.doctree.IdentifierTree;
import com.sun.source.doctree.InlineTagTree;
import com.sun.source.doctree.LinkTree;
import com.sun.source.doctree.LiteralTree;
import com.sun.source.doctree.ParamTree;
import com.sun.source.doctree.ProvidesTree;
import com.sun.source.doctree.ReferenceTree;
import com.sun.source.doctree.ReturnTree;
import com.sun.source.doctree.SeeTree;
import com.sun.source.doctree.SerialDataTree;
import com.sun.source.doctree.SerialFieldTree;
import com.sun.source.doctree.SerialTree;
import com.sun.source.doctree.SinceTree;
import com.sun.source.doctree.StartElementTree;
import com.sun.source.doctree.TextTree;
import com.sun.source.doctree.ThrowsTree;
import com.sun.source.doctree.UnknownBlockTagTree;
import com.sun.source.doctree.UsesTree;
import com.sun.source.doctree.ValueTree;
import com.sun.source.doctree.VersionTree;
import com.sun.source.util.DocTreePath;
import com.sun.source.util.DocTrees;
import com.sun.source.util.SimpleDocTreeVisitor;
import com.sun.source.util.TreePath;
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
import static com.sun.source.doctree.DocTree.Kind.*;

public class CommentHelper {

    public final TreePath path;

    public final DocCommentTree dcTree;

    public final Element element;

    public static final String SPACER;

    public CommentHelper(BaseConfiguration configuration, Element element, TreePath path, DocCommentTree dcTree) {
    }

    public void setOverrideElement(Element ove);

    public String getTagName(DocTree dtree);

    @Pure
    public boolean isTypeParameter(DocTree dtree);

    public String getParameterName(DocTree dtree);

    Element getElement(ReferenceTree rtree);

    public TypeMirror getType(ReferenceTree rtree);

    public Element getException(ThrowsTree tt);

    public List<? extends DocTree> getDescription(DocTree dtree);

    public String getText(List<? extends DocTree> list);

    public String getText(DocTree dt);

    public String getLabel(DocTree dtree);

    public TypeElement getReferencedClass(DocTree dtree);

    public String getReferencedModuleName(DocTree dtree);

    public Element getReferencedMember(DocTree dtree);

    public String getReferencedMemberName(DocTree dtree);

    public PackageElement getReferencedPackage(DocTree dtree);

    public ModuleElement getReferencedModule(DocTree dtree);

    public List<? extends DocTree> getFirstSentenceTrees(List<? extends DocTree> body);

    public List<? extends DocTree> getFirstSentenceTrees(DocTree dtree);

    public TypeMirror getReferencedType(DocTree dtree);

    public TypeElement getServiceType(DocTree dtree);

    public String getReferencedSignature(DocTree dtree);

    private static class ReferenceDocTreeVisitor<R> extends SimpleDocTreeVisitor<R, Void> {

        @Override
        public R visitSee(SeeTree node, Void p);

        @Override
        public R visitLink(LinkTree node, Void p);

        @Override
        public R visitProvides(ProvidesTree node, Void p);

        @Override
        public R visitValue(ValueTree node, Void p);

        @Override
        public R visitSerialField(SerialFieldTree node, Void p);

        @Override
        public R visitUses(UsesTree node, Void p);

        @Override
        protected R defaultAction(DocTree node, Void p);
    }

    public List<? extends DocTree> getReference(DocTree dtree);

    public ReferenceTree getExceptionName(DocTree dtree);

    public IdentifierTree getName(DocTree dtree);

    public List<? extends DocTree> getTags(DocTree dtree);

    public List<? extends DocTree> getBody(DocTree dtree);

    public ReferenceTree getType(DocTree dtree);

    public DocTreePath getDocTreePath(DocTree dtree);

    public Element getOverriddenElement();

    @Override
    public String toString();
}
