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
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.Name;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.QualifiedNameable;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.SimpleAnnotationValueVisitor9;
import javax.lang.model.util.SimpleElementVisitor14;
import javax.lang.model.util.SimpleTypeVisitor9;
import com.sun.source.doctree.AttributeTree;
import com.sun.source.doctree.AttributeTree.ValueKind;
import com.sun.source.doctree.CommentTree;
import com.sun.source.doctree.DeprecatedTree;
import com.sun.source.doctree.DocRootTree;
import com.sun.source.doctree.DocTree;
import com.sun.source.doctree.DocTree.Kind;
import com.sun.source.doctree.EndElementTree;
import com.sun.source.doctree.EntityTree;
import com.sun.source.doctree.ErroneousTree;
import com.sun.source.doctree.IndexTree;
import com.sun.source.doctree.InheritDocTree;
import com.sun.source.doctree.LinkTree;
import com.sun.source.doctree.LiteralTree;
import com.sun.source.doctree.SeeTree;
import com.sun.source.doctree.StartElementTree;
import com.sun.source.doctree.SummaryTree;
import com.sun.source.doctree.SystemPropertyTree;
import com.sun.source.doctree.TextTree;
import com.sun.source.util.SimpleDocTreeVisitor;
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
import jdk.javadoc.internal.doclets.formats.html.markup.Head;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlDocument;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlId;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
import jdk.javadoc.internal.doclets.formats.html.markup.Links;
import jdk.javadoc.internal.doclets.formats.html.markup.RawHtml;
import jdk.javadoc.internal.doclets.formats.html.markup.Script;
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.Messages;
import jdk.javadoc.internal.doclets.toolkit.PackageSummaryWriter;
import jdk.javadoc.internal.doclets.toolkit.Resources;
import jdk.javadoc.internal.doclets.toolkit.taglets.DocRootTaglet;
import jdk.javadoc.internal.doclets.toolkit.taglets.Taglet;
import jdk.javadoc.internal.doclets.toolkit.taglets.TagletWriter;
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
import jdk.javadoc.internal.doclets.toolkit.util.Comparators;
import jdk.javadoc.internal.doclets.toolkit.util.DocFile;
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
import jdk.javadoc.internal.doclets.toolkit.util.DocLink;
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;
import jdk.javadoc.internal.doclets.toolkit.util.DocletConstants;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import jdk.javadoc.internal.doclets.toolkit.util.Utils.DeclarationPreviewLanguageFeatures;
import jdk.javadoc.internal.doclets.toolkit.util.Utils.ElementFlag;
import jdk.javadoc.internal.doclets.toolkit.util.Utils.PreviewSummary;
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
import jdk.javadoc.internal.doclint.HtmlTag;
import static com.sun.source.doctree.DocTree.Kind.CODE;
import static com.sun.source.doctree.DocTree.Kind.COMMENT;
import static com.sun.source.doctree.DocTree.Kind.LINK;
import static com.sun.source.doctree.DocTree.Kind.LINK_PLAIN;
import static com.sun.source.doctree.DocTree.Kind.SEE;
import static com.sun.source.doctree.DocTree.Kind.TEXT;
import static jdk.javadoc.internal.doclets.toolkit.util.CommentHelper.SPACER;

public class HtmlDocletWriter {

    public final DocPath pathToRoot;

    public final DocPath path;

    public final DocPath filename;

    public final HtmlConfiguration configuration;

    protected final HtmlOptions options;

    protected final Utils utils;

    protected final Contents contents;

    protected final Messages messages;

    protected final Resources resources;

    protected final Links links;

    protected final DocPaths docPaths;

    protected final Comparators comparators;

    protected final HtmlIds htmlIds;

    protected boolean printedAnnotationHeading;

    protected boolean printedAnnotationFieldHeading;

    protected String winTitle;

    protected Script mainBodyScript;

    public HtmlDocletWriter(HtmlConfiguration configuration, DocPath path) {
    }

    public String replaceDocRootDir(String htmlstr);

    protected void addTagsInfo(Element e, Content htmlTree);

    protected Content getBlockTagOutput(Element element);

    protected Content getBlockTagOutput(Element element, List<Taglet> taglets);

    protected boolean hasSerializationOverviewTags(VariableElement field);

    public TagletWriter getTagletWriterInstance(boolean isFirstSentence);

    public TagletWriter getTagletWriterInstance(TagletWriterImpl.Context context);

    public void printHtmlDocument(List<String> metakeywords, String description, Content body) throws DocFileIOException;

    public void printHtmlDocument(List<String> metakeywords, String description, List<DocPath> localStylesheets, Content body) throws DocFileIOException;

    public void printHtmlDocument(List<String> metakeywords, String description, Content extraHeadContent, List<DocPath> localStylesheets, Content body) throws DocFileIOException;

    public String getWindowTitle(String title);

    protected HtmlTree getHeader(Navigation.PageMode pageMode);

    protected HtmlTree getHeader(Navigation.PageMode pageMode, Element element);

    protected Navigation getNavBar(Navigation.PageMode pageMode, Element element);

    public HtmlTree getFooter();

    protected Content getNavLinkMainTree(String label);

    public Content getLocalizedPackageName(PackageElement packageElement);

    public Content getPackageLabel(CharSequence packageName);

    protected DocPath pathString(TypeElement te, DocPath name);

    protected DocPath pathString(PackageElement packageElement, DocPath name);

    public Content getPackageLink(PackageElement packageElement, Content label);

    public Content getModuleLink(ModuleElement mdle, Content label);

    public void addSrcLink(Element element, Content label, Content htmltree);

    public Content getLink(HtmlLinkInfo linkInfo);

    public Content getTypeParameterLinks(HtmlLinkInfo linkInfo);

    public Content getCrossClassLink(TypeElement classElement, String refMemName, Content label, HtmlStyle style, boolean code);

    public DocLink getCrossPackageLink(PackageElement element);

    public DocLink getCrossModuleLink(ModuleElement element);

    public Content getQualifiedClassLink(HtmlLinkInfo.Kind context, Element element);

    public void addPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Content contentTree);

    public Content getPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement);

    public void addPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, HtmlStyle style, Content contentTree);

    public String getEnclosingPackageName(TypeElement te);

    protected TypeElement getCurrentPageElement();

    public void addPreQualifiedStrongClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Content contentTree);

    public Content getDocLink(HtmlLinkInfo.Kind context, Element element, CharSequence label);

    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label);

    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label, HtmlStyle style);

    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label, boolean isProperty);

    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, Content label, HtmlStyle style, boolean isProperty);

    public Content seeTagToContent(Element element, DocTree see, TagletWriterImpl.Context context);

    public void addInlineComment(Element element, DocTree tag, Content htmltree);

    public Content getDeprecatedPhrase(Element e);

    public void addInlineDeprecatedComment(Element e, DeprecatedTree tag, Content htmltree);

    public void addSummaryComment(Element element, Content htmltree);

    public void addPreviewComment(Element element, List<? extends DocTree> firstSentenceTags, Content htmltree);

    public void addSummaryComment(Element element, List<? extends DocTree> firstSentenceTags, Content htmltree);

    public void addSummaryDeprecatedComment(Element element, DeprecatedTree tag, Content htmltree);

    public void addInlineComment(Element element, Content htmltree);

    boolean ignoreNonInlineTag(DocTree dtree);

    boolean isAllWhiteSpace(String body);

    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> tags, boolean isFirstSentence);

    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> trees, boolean isFirstSentence, boolean inSummary);

    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> trees, TagletWriterImpl.Context context);

    @Pure
    public boolean isCoreClass(TypeElement typeElement);

    Content getAnnotationInfo(Element element, boolean lineBreak);

    Content getAnnotationInfo(List<? extends AnnotationMirror> descList, boolean lineBreak);

    public List<Content> getAnnotations(List<? extends AnnotationMirror> descList, boolean lineBreak);

    protected TableHeader getPackageTableHeader();

    static String getDescription(String prefix, Element elem);

    static String getGenerator(Class<?> clazz);

    public HtmlTree getBody(String title);

    public HtmlStyle getBodyStyle();

    Script getMainBodyScript();

    List<DocPath> getLocalStylesheets(Element element) throws DocFileIOException;

    public void addPreviewSummary(Element forWhat, Content target);

    public void addPreviewInfo(Element forWhat, Content target);
}
