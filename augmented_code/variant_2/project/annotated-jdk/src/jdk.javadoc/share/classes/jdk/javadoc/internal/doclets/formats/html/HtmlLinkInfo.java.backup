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
package jdk.javadoc.internal.doclets.formats.html;

import org.checkerframework.dataflow.qual.Pure;
import javax.lang.model.element.Element;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.type.TypeMirror;
import jdk.javadoc.internal.doclets.formats.html.markup.ContentBuilder;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlStyle;
import jdk.javadoc.internal.doclets.formats.html.markup.Text;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import jdk.javadoc.internal.doclets.toolkit.util.links.LinkInfo;

public class HtmlLinkInfo extends LinkInfo {

    public enum Kind {

        DEFAULT,
        CLASS,
        MEMBER,
        MEMBER_DEPRECATED_PREVIEW,
        CLASS_USE,
        INDEX,
        CONSTANT_SUMMARY,
        SERIALIZED_FORM,
        SERIAL_MEMBER,
        PACKAGE,
        SEE_TAG,
        VALUE_TAG,
        TREE,
        CLASS_HEADER,
        CLASS_SIGNATURE,
        RETURN_TYPE,
        SUMMARY_RETURN_TYPE,
        EXECUTABLE_MEMBER_PARAM,
        SUPER_INTERFACES,
        IMPLEMENTED_INTERFACES,
        IMPLEMENTED_CLASSES,
        SUBINTERFACES,
        SUBCLASSES,
        CLASS_SIGNATURE_PARENT_NAME,
        PERMITTED_SUBCLASSES,
        EXECUTABLE_ELEMENT_COPY,
        METHOD_SPECIFIED_BY,
        METHOD_OVERRIDES,
        ANNOTATION,
        CLASS_TREE_PARENT,
        MEMBER_TYPE_PARAMS,
        CLASS_USE_HEADER,
        PROPERTY_COPY,
        RECEIVER_TYPE,
        RECORD_COMPONENT,
        THROWS_TYPE
    }

    public final HtmlConfiguration configuration;

    public Kind context;

    public String where;

    public Element targetMember;

    public HtmlStyle style;

    public final Utils utils;

    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, ExecutableElement ee) {
    }

    @Override
    protected Content newContent();

    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, TypeElement typeElement) {
    }

    public HtmlLinkInfo(HtmlConfiguration configuration, Kind context, TypeMirror type) {
    }

    public HtmlLinkInfo label(CharSequence label);

    public HtmlLinkInfo label(Content label);

    public HtmlLinkInfo style(HtmlStyle style);

    public HtmlLinkInfo varargs(boolean varargs);

    public HtmlLinkInfo where(String where);

    public HtmlLinkInfo targetMember(Element el);

    public HtmlLinkInfo skipPreview(boolean skipPreview);

    public Kind getContext();

    public final void setContext(Kind c);

    @Override
    @Pure
    public boolean isLinkable();

    @Override
    public boolean includeTypeParameterLinks();

    @Override
    public String toString();
}
