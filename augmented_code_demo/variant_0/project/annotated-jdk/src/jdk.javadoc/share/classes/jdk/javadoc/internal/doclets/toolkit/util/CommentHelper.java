/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import com.sun.source.doctree.AttributeTree;
    @Positive
import com.sun.source.doctree.AttributeTree.ValueKind;
    @Positive
import com.sun.source.doctree.AuthorTree;
    @Positive
import com.sun.source.doctree.BlockTagTree;
    @Positive
import com.sun.source.doctree.CommentTree;
    @Positive
import com.sun.source.doctree.DeprecatedTree;
    @Positive
import com.sun.source.doctree.DocCommentTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import com.sun.source.doctree.EndElementTree;
    @Positive
import com.sun.source.doctree.EntityTree;
    @Positive
import com.sun.source.doctree.IdentifierTree;
    @Positive
import com.sun.source.doctree.InlineTagTree;
    @Positive
import com.sun.source.doctree.LinkTree;
    @Positive
import com.sun.source.doctree.LiteralTree;
    @Positive
import com.sun.source.doctree.ParamTree;
    @Positive
import com.sun.source.doctree.ProvidesTree;
    @Positive
import com.sun.source.doctree.ReferenceTree;
    @Positive
import com.sun.source.doctree.ReturnTree;
    @Positive
import com.sun.source.doctree.SeeTree;
    @Positive
import com.sun.source.doctree.SerialDataTree;
    @Positive
import com.sun.source.doctree.SerialFieldTree;
    @Positive
import com.sun.source.doctree.SerialTree;
    @Positive
import com.sun.source.doctree.SinceTree;
    @Positive
import com.sun.source.doctree.StartElementTree;
    @Positive
import com.sun.source.doctree.TextTree;
    @Positive
import com.sun.source.doctree.ThrowsTree;
    @Positive
import com.sun.source.doctree.UnknownBlockTagTree;
    @Positive
import com.sun.source.doctree.UsesTree;
    @Positive
import com.sun.source.doctree.ValueTree;
    @Positive
import com.sun.source.doctree.VersionTree;
    @Positive
import com.sun.source.util.DocTreePath;
    @Positive
import com.sun.source.util.DocTrees;
    @Positive
import com.sun.source.util.SimpleDocTreeVisitor;
    @Positive
import com.sun.source.util.TreePath;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.*;

    @Positive
public class CommentHelper {

    @Positive
    public final TreePath path;

    @Positive
    public final DocCommentTree dcTree;

    @Positive
    public final Element element;

    @Positive
    public static final String SPACER;

    @Positive
    public CommentHelper(BaseConfiguration configuration, Element element, TreePath path, DocCommentTree dcTree) {
    @Positive
    }

    @Positive
    public void setOverrideElement(Element ove);

    @Positive
    public String getTagName(DocTree dtree);

    @Positive
    @Pure
    @Positive
    public boolean isTypeParameter(DocTree dtree);

    @Positive
    public String getParameterName(DocTree dtree);

    @Positive
    Element getElement(ReferenceTree rtree);

    @Positive
    public TypeMirror getType(ReferenceTree rtree);

    @Positive
    public Element getException(ThrowsTree tt);

    @Positive
    public List<? extends DocTree> getDescription(DocTree dtree);

    @Positive
    public String getText(List<? extends DocTree> list);

    @Positive
    public String getText(DocTree dt);

    @Positive
    public String getLabel(DocTree dtree);

    @Positive
    public TypeElement getReferencedClass(DocTree dtree);

    @Positive
    public String getReferencedModuleName(DocTree dtree);

    @Positive
    public Element getReferencedMember(DocTree dtree);

    @Positive
    public String getReferencedMemberName(DocTree dtree);

    @Positive
    public PackageElement getReferencedPackage(DocTree dtree);

    @Positive
    public ModuleElement getReferencedModule(DocTree dtree);

    @Positive
    public List<? extends DocTree> getFirstSentenceTrees(List<? extends DocTree> body);

    @Positive
    public List<? extends DocTree> getFirstSentenceTrees(DocTree dtree);

    @Positive
    public TypeMirror getReferencedType(DocTree dtree);

    @Positive
    public TypeElement getServiceType(DocTree dtree);

    @Positive
    public String getReferencedSignature(DocTree dtree);

    @Positive
    private static class ReferenceDocTreeVisitor<R> extends SimpleDocTreeVisitor<R, Void> {

    @Positive
        @Override
    @Positive
        public R visitSee(SeeTree node, Void p);

    @Positive
        @Override
    @Positive
        public R visitLink(LinkTree node, Void p);

    @Positive
        @Override
    @Positive
        public R visitProvides(ProvidesTree node, Void p);

    @Positive
        @Override
    @Positive
        public R visitValue(ValueTree node, Void p);

    @Positive
        @Override
    @Positive
        public R visitSerialField(SerialFieldTree node, Void p);

    @Positive
        @Override
    @Positive
        public R visitUses(UsesTree node, Void p);

    @Positive
        @Override
    @Positive
        protected R defaultAction(DocTree node, Void p);
    @Positive
    }

    @Positive
    public List<? extends DocTree> getReference(DocTree dtree);

    @Positive
    public ReferenceTree getExceptionName(DocTree dtree);

    @Positive
    public IdentifierTree getName(DocTree dtree);

    @Positive
    public List<? extends DocTree> getTags(DocTree dtree);

    @Positive
    public List<? extends DocTree> getBody(DocTree dtree);

    @Positive
    public ReferenceTree getType(DocTree dtree);

    @Positive
    public DocTreePath getDocTreePath(DocTree dtree);

    @Positive
    public Element getOverriddenElement();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
