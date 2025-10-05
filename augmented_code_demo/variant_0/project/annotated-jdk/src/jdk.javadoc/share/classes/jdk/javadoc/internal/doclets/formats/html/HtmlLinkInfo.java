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
package jdk.javadoc.internal.doclets.formats.html;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import javax.lang.model.type.TypeMirror;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.links.LinkInfo;

    @Positive
public class HtmlLinkInfo extends LinkInfo {

    @Positive
    public enum Kind {

    @Positive
        DEFAULT,
    @Positive
        CLASS,
    @Positive
        MEMBER,
    @Positive
        MEMBER_DEPRECATED_PREVIEW,
    @Positive
        CLASS_USE,
    @Positive
        INDEX,
    @Positive
        CONSTANT_SUMMARY,
    @Positive
        SERIALIZED_FORM,
    @Positive
        SERIAL_MEMBER,
    @Positive
        PACKAGE,
    @Positive
        SEE_TAG,
    @Positive
        VALUE_TAG,
    @Positive
        TREE,
    @Positive
        CLASS_HEADER,
    @Positive
        CLASS_SIGNATURE,
    @Positive
        RETURN_TYPE,
    @Positive
        SUMMARY_RETURN_TYPE,
    @Positive
        EXECUTABLE_MEMBER_PARAM,
    @Positive
        SUPER_INTERFACES,
    @Positive
        IMPLEMENTED_INTERFACES,
    @Positive
        IMPLEMENTED_CLASSES,
    @Positive
        SUBINTERFACES,
    @Positive
        SUBCLASSES,
    @Positive
        CLASS_SIGNATURE_PARENT_NAME,
    @Positive
        PERMITTED_SUBCLASSES,
    @Positive
        EXECUTABLE_ELEMENT_COPY,
    @Positive
        METHOD_SPECIFIED_BY,
    @Positive
        METHOD_OVERRIDES,
    @Positive
        ANNOTATION,
    @Positive
        CLASS_TREE_PARENT,
    @Positive
        MEMBER_TYPE_PARAMS,
    @Positive
        CLASS_USE_HEADER,
    @Positive
        PROPERTY_COPY,
    @Positive
        RECEIVER_TYPE,
    @Positive
        RECORD_COMPONENT,
    @Positive
        THROWS_TYPE
    @Positive
    }

    @Positive
    public final HtmlConfiguration configuration;

    @Positive
    public Kind context;

    @Positive
    public String where;

    @Positive
    public Element targetMember;

    @Positive
    public HtmlStyle style;

    @Positive
    public final Utils utils;

    @Positive
    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, ExecutableElement ee) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    protected Content newContent();

    @Positive
    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, TypeElement typeElement) {
    @Positive
    }

    @Positive
    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, TypeMirror type) {
    @Positive
    }

    @Positive
    public HtmlLinkInfo label(CharSequence label);

    @Positive
    public HtmlLinkInfo label(Content label);

    @Positive
    public HtmlLinkInfo style(HtmlStyle style);

    @Positive
    public HtmlLinkInfo varargs(boolean varargs);

    @Positive
    public HtmlLinkInfo where(String where);

    @Positive
    public HtmlLinkInfo targetMember(Element el);

    @Positive
    public HtmlLinkInfo skipPreview(boolean skipPreview);

    @Positive
    public Kind getContext();

    @Positive
    public final void setContext(Kind c);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean isLinkable();

    @Positive
    @Override
    @Positive
    public boolean includeTypeParameterLinks();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
