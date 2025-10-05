/*
 * Copyright (c) 2010, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.formats.html.markup;

import org.checkerframework.dataflow.qual.Pure;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlAttr.Role;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.util.DocletConstants;

public class HtmlTree extends Content {

    public final TagName tagName;

    public static final Content EMPTY;

    public HtmlTree(TagName tagName) {
    }

    public HtmlTree put(HtmlAttr attrName, String attrValue);

    public HtmlTree setId(HtmlId id);

    public HtmlTree setTitle(Content body);

    public HtmlTree setRole(Role role);

    public HtmlTree setStyle(HtmlStyle style);

    public HtmlTree addStyle(HtmlStyle style);

    public HtmlTree addStyle(String style);

    @Override
    public HtmlTree add(Content content);

    @Override
    public HtmlTree add(CharSequence stringContent);

    public HtmlTree add(List<? extends Content> list);

    @Override
    public int charCount();

    public static final BitSet MAIN_CHARS;

    public static final BitSet QUERY_FRAGMENT_CHARS;

    public static String encodeURL(String url);

    public static HtmlTree A(String ref, Content body);

    public static HtmlTree CAPTION(Content body);

    public static HtmlTree CODE(Content body);

    public static HtmlTree DD(Content body);

    public static HtmlTree DL(HtmlStyle style);

    public static HtmlTree DL(HtmlStyle style, Content body);

    public static HtmlTree DIV(HtmlStyle style);

    public static HtmlTree DIV(HtmlStyle style, Content body);

    public static HtmlTree DIV(Content body);

    public static HtmlTree DT(Content body);

    public static HtmlTree FOOTER();

    public static HtmlTree HEADER();

    public static HtmlTree HEADING(TagName headingTag, Content body);

    public static HtmlTree HEADING(TagName headingTag, HtmlStyle style, Content body);

    public static HtmlTree HEADING_TITLE(TagName headingTag, HtmlStyle style, Content body);

    public static HtmlTree HEADING_TITLE(TagName headingTag, Content body);

    public static HtmlTree HTML(String lang, Content head, Content body);

    public static HtmlTree INPUT(String type, HtmlId id, String value);

    public static HtmlTree LABEL(String forLabel, Content body);

    public static HtmlTree LI(Content body);

    public static HtmlTree LI(HtmlStyle style, Content body);

    public static HtmlTree LINK(String rel, String type, String href, String title);

    public static HtmlTree MAIN();

    public static HtmlTree MAIN(Content body);

    public static HtmlTree META(String httpEquiv, String content, String charset);

    public static HtmlTree META(String name, String content);

    public static HtmlTree NAV();

    public static HtmlTree NOSCRIPT(Content body);

    public static HtmlTree P(Content body);

    public static HtmlTree P(HtmlStyle style, Content body);

    public static HtmlTree SCRIPT(String src);

    public static HtmlTree SECTION(HtmlStyle style);

    public static HtmlTree SECTION(HtmlStyle style, Content body);

    public static HtmlTree SMALL(Content body);

    public static HtmlTree SPAN(Content body);

    public static HtmlTree SPAN(HtmlStyle styleClass, Content body);

    public static HtmlTree SPAN_ID(HtmlId id, Content body);

    public static HtmlTree SPAN(HtmlId id, HtmlStyle style, Content body);

    public static HtmlTree SUP(Content body);

    public static HtmlTree TD(HtmlStyle style, Content body);

    public static HtmlTree TH(HtmlStyle style, String scope, Content body);

    public static HtmlTree TH(String scope, Content body);

    public static HtmlTree TITLE(String body);

    public static HtmlTree UL(HtmlStyle style, Content first, Content... more);

    @Override
    @Pure
    public boolean isEmpty();

    public boolean hasContent();

    public boolean hasAttrs();

    public boolean hasAttr(HtmlAttr attrName);

    @Override
    @Pure
    public boolean isValid();

    @Pure
    public boolean isInline();

    public boolean isVoid();

    @Override
    public boolean write(Writer out, boolean atNewline) throws IOException;
}
