/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package jdk.javadoc.internal.doclets.formats.html;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.TreeSet;
    @Positive
import javax.lang.model.element.AnnotationMirror;
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
import javax.lang.model.util.SimpleElementVisitor8;
    @Positive
import com.sun.source.doctree.DeprecatedTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.Navigation.PageMode;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlAttr;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.ParamTaglet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.ClassTree;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;

    @Positive
public class ClassWriterImpl extends SubWriterHolderWriter implements ClassWriter {

    @Positive
    protected final TypeElement typeElement;

    @Positive
    protected final ClassTree classtree;

    @Positive
    public ClassWriterImpl(HtmlConfiguration configuration, TypeElement typeElement, ClassTree classTree) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Content getHeader(String header);

    @Positive
    @Override
    @Positive
    public Content getClassContentHeader();

    @Positive
    @Override
    @Positive
    protected Navigation getNavBar(PageMode pageMode, Element element);

    @Positive
    @Override
    @Positive
    public void addFooter();

    @Positive
    @Override
    @Positive
    public void printDocument(Content contentTree) throws DocFileIOException;

    @Positive
    @Override
    @Positive
    public Content getClassInfoTreeHeader();

    @Positive
    @Override
    @Positive
    public Content getClassInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    protected TypeElement getCurrentPageElement();

    @Positive
    @Override
    @Positive
    public void addClassSignature(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addClassDescription(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addClassTagInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addClassTree(Content classContentTree);

    @Positive
    @Override
    @Positive
    public void addParamInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addSubClassInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addSubInterfacesInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addInterfaceUsageInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addImplementedInterfacesInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addSuperInterfacesInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addNestedClassInfo(final Content classInfoTree);

    @Positive
    @Override
    @Positive
    public void addFunctionalInterfaceInfo(Content classInfoTree);

    @Positive
    @Pure
    @Positive
    public boolean isFunctionalInterface();

    @Positive
    @Override
    @Positive
    public void addClassDeprecationInfo(Content classInfoTree);

    @Positive
    @Override
    @Positive
    public TypeElement getTypeElement();

    @Positive
    public Content getMemberDetailsTree(Content contentTree);
    @Positive
}
