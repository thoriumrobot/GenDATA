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
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedList;
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import javax.lang.model.element.AnnotationMirror;
    @Positive
import javax.lang.model.element.AnnotationValue;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ElementKind;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.ModuleElement;
    @Positive
import javax.lang.model.element.Name;
    @Positive
import javax.lang.model.element.PackageElement;
    @Positive
import javax.lang.model.element.QualifiedNameable;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.element.VariableElement;
    @Positive
import javax.lang.model.type.DeclaredType;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import javax.lang.model.util.SimpleAnnotationValueVisitor9;
    @Positive
import javax.lang.model.util.SimpleElementVisitor14;
    @Positive
import javax.lang.model.util.SimpleTypeVisitor9;
    @Positive
import com.sun.source.doctree.AttributeTree;
    @Positive
import com.sun.source.doctree.AttributeTree.ValueKind;
    @Positive
import com.sun.source.doctree.CommentTree;
    @Positive
import com.sun.source.doctree.DeprecatedTree;
    @Positive
import com.sun.source.doctree.DocRootTree;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import com.sun.source.doctree.DocTree.Kind;
    @Positive
import com.sun.source.doctree.EndElementTree;
    @Positive
import com.sun.source.doctree.EntityTree;
    @Positive
import com.sun.source.doctree.ErroneousTree;
    @Positive
import com.sun.source.doctree.IndexTree;
    @Positive
import com.sun.source.doctree.InheritDocTree;
    @Positive
import com.sun.source.doctree.LinkTree;
    @Positive
import com.sun.source.doctree.LiteralTree;
    @Positive
import com.sun.source.doctree.SeeTree;
    @Positive
import com.sun.source.doctree.StartElementTree;
    @Positive
import com.sun.source.doctree.SummaryTree;
    @Positive
import com.sun.source.doctree.SystemPropertyTree;
    @Positive
import com.sun.source.doctree.TextTree;
    @Positive
import com.sun.source.util.SimpleDocTreeVisitor;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Entity;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Head;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlDocument;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlId;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Links;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.RawHtml;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Script;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.TagName;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.ClassWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Messages;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.PackageSummaryWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Resources;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.DocRootTaglet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.Taglet;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.taglets.TagletWriter;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.CommentHelper;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Comparators;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFile;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocFileIOException;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocLink;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPath;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocPaths;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocletConstants;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils.DeclarationPreviewLanguageFeatures;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils.ElementFlag;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils.PreviewSummary;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.VisibleMemberTable;
    @Positive
import jdk.javadoc.internal.doclint.HtmlTag;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.CODE;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.COMMENT;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.LINK;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.LINK_PLAIN;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.SEE;
    @Positive
import static com.sun.source.doctree.DocTree.Kind.TEXT;
    @Positive
import static jdk.javadoc.internal.doclets.toolkit.util.CommentHelper.SPACER;

    @Positive
public class HtmlDocletWriter {

    @Positive
    public final DocPath pathToRoot;

    @Positive
    public final DocPath path;

    @Positive
    public final DocPath filename;

    @Positive
    public final HtmlConfiguration configuration;

    @Positive
    protected final HtmlOptions options;

    @Positive
    protected final Utils utils;

    @Positive
    protected final Contents contents;

    @Positive
    protected final Messages messages;

    @Positive
    protected final Resources resources;

    @Positive
    protected final Links links;

    @Positive
    protected final DocPaths docPaths;

    @Positive
    protected final Comparators comparators;

    @Positive
    protected final HtmlIds htmlIds;

    @Positive
    protected boolean printedAnnotationHeading;

    @Positive
    protected boolean printedAnnotationFieldHeading;

    @Positive
    protected String winTitle;

    @Positive
    protected Script mainBodyScript;

    @Positive
    public HtmlDocletWriter(HtmlConfiguration configuration, DocPath path) {
    @Positive
    }

    @Positive
    public String replaceDocRootDir(String htmlstr);

    @Positive
    protected void addTagsInfo(Element e, Content htmlTree);

    @Positive
    protected Content getBlockTagOutput(Element element);

    @Positive
    protected Content getBlockTagOutput(Element element, List<Taglet> taglets);

    @Positive
    protected boolean hasSerializationOverviewTags(VariableElement field);

    @Positive
    public TagletWriter getTagletWriterInstance(boolean isFirstSentence);

    @Positive
    public TagletWriter getTagletWriterInstance(TagletWriterImpl.Context context);

    @Positive
    public void printHtmlDocument(List<String> metakeywords, String description, Content body) throws DocFileIOException;

    @Positive
    public void printHtmlDocument(List<String> metakeywords, String description, List<DocPath> localStylesheets, Content body) throws DocFileIOException;

    @Positive
    public void printHtmlDocument(List<String> metakeywords, String description, Content extraHeadContent, List<DocPath> localStylesheets, Content body) throws DocFileIOException;

    @Positive
    public String getWindowTitle(String title);

    @Positive
    protected HtmlTree getHeader(Navigation.PageMode pageMode);

    @Positive
    protected HtmlTree getHeader(Navigation.PageMode pageMode, Element element);

    @Positive
    protected Navigation getNavBar(Navigation.PageMode pageMode, Element element);

    @Positive
    public HtmlTree getFooter();

    @Positive
    protected Content getNavLinkMainTree(String label);

    @Positive
    public Content getLocalizedPackageName(PackageElement packageElement);

    @Positive
    public Content getPackageLabel(CharSequence packageName);

    @Positive
    protected DocPath pathString(TypeElement te, DocPath name);

    @Positive
    protected DocPath pathString(PackageElement packageElement, DocPath name);

    @Positive
    public Content getPackageLink(PackageElement packageElement, Content label);

    @Positive
    public Content getModuleLink(ModuleElement mdle, Content label);

    @Positive
    public void addSrcLink(Element element, Content label, Content htmltree);

    @Positive
    public Content getLink(HtmlLinkInfo linkInfo);

    @Positive
    public Content getTypeParameterLinks(HtmlLinkInfo linkInfo);

    @Positive
    public Content getCrossClassLink(TypeElement classElement, String refMemName, Content label, HtmlStyle style, boolean code);

    @Positive
    public DocLink getCrossPackageLink(PackageElement element);

    @Positive
    public DocLink getCrossModuleLink(ModuleElement element);

    @Positive
    public Content getQualifiedClassLink(HtmlLinkInfo.Kind context, Element element);

    @Positive
    public void addPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Content contentTree);

    @Positive
    public Content getPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement);

    @Positive
    public void addPreQualifiedClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, HtmlStyle style, Content contentTree);

    @Positive
    public String getEnclosingPackageName(TypeElement te);

    @Positive
    protected TypeElement getCurrentPageElement();

    @Positive
    public void addPreQualifiedStrongClassLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Content contentTree);

    @Positive
    public Content getDocLink(HtmlLinkInfo.Kind context, Element element, CharSequence label);

    @Positive
    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label);

    @Positive
    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label, HtmlStyle style);

    @Positive
    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, CharSequence label, boolean isProperty);

    @Positive
    public Content getDocLink(HtmlLinkInfo.Kind context, TypeElement typeElement, Element element, Content label, HtmlStyle style, boolean isProperty);

    @Positive
    public Content seeTagToContent(Element element, DocTree see, TagletWriterImpl.Context context);

    @Positive
    public void addInlineComment(Element element, DocTree tag, Content htmltree);

    @Positive
    public Content getDeprecatedPhrase(Element e);

    @Positive
    public void addInlineDeprecatedComment(Element e, DeprecatedTree tag, Content htmltree);

    @Positive
    public void addSummaryComment(Element element, Content htmltree);

    @Positive
    public void addPreviewComment(Element element, List<? extends DocTree> firstSentenceTags, Content htmltree);

    @Positive
    public void addSummaryComment(Element element, List<? extends DocTree> firstSentenceTags, Content htmltree);

    @Positive
    public void addSummaryDeprecatedComment(Element element, DeprecatedTree tag, Content htmltree);

    @Positive
    public void addInlineComment(Element element, Content htmltree);

    @Positive
    boolean ignoreNonInlineTag(DocTree dtree);

    @Positive
    boolean isAllWhiteSpace(String body);

    @Positive
    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> tags, boolean isFirstSentence);

    @Positive
    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> trees, boolean isFirstSentence, boolean inSummary);

    @Positive
    public Content commentTagsToContent(DocTree holderTag, Element element, List<? extends DocTree> trees, TagletWriterImpl.Context context);

    @Positive
    @Pure
    @Positive
    public boolean isCoreClass(TypeElement typeElement);

    @Positive
    Content getAnnotationInfo(Element element, boolean lineBreak);

    @Positive
    Content getAnnotationInfo(List<? extends AnnotationMirror> descList, boolean lineBreak);

    @Positive
    public List<Content> getAnnotations(List<? extends AnnotationMirror> descList, boolean lineBreak);

    @Positive
    protected TableHeader getPackageTableHeader();

    @Positive
    static String getDescription(String prefix, Element elem);

    @Positive
    static String getGenerator(Class<?> clazz);

    @Positive
    public HtmlTree getBody(String title);

    @Positive
    public HtmlStyle getBodyStyle();

    @Positive
    Script getMainBodyScript();

    @Positive
    List<DocPath> getLocalStylesheets(Element element) throws DocFileIOException;

    @Positive
    public void addPreviewSummary(Element forWhat, Content target);

    @Positive
    public void addPreviewInfo(Element forWhat, Content target);
    @Positive
}

// CFWR semantic augmentation - variant 1
