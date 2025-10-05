/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2010, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.formats.html.markup;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.io.Writer;
    @Positive
import java.nio.charset.StandardCharsets;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.BitSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.HtmlAttr.Role;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.DocletConstants;

    @Positive
public class HtmlTree extends Content {

    @Positive
    public final TagName tagName;

    @Positive
    public static final Content EMPTY;

    @Positive
    public HtmlTree(TagName tagName) {
    @Positive
    }

    @Positive
    public HtmlTree put(HtmlAttr attrName, String attrValue);

    @Positive
    public HtmlTree setId(HtmlId id);

    @Positive
    public HtmlTree setTitle(Content body);

    @Positive
    public HtmlTree setRole(Role role);

    @Positive
    public HtmlTree setStyle(HtmlStyle style);

    @Positive
    public HtmlTree addStyle(HtmlStyle style);

    @Positive
    public HtmlTree addStyle(String style);

    @Positive
    @Override
    @Positive
    public HtmlTree add(Content content);

    @Positive
    @Override
    @Positive
    public HtmlTree add(CharSequence stringContent);

    @Positive
    public HtmlTree add(List<? extends Content> list);

    @Positive
    @Override
    @Positive
    public int charCount();

    @Positive
    public static final BitSet MAIN_CHARS;

    @Positive
    public static final BitSet QUERY_FRAGMENT_CHARS;

    @Positive
    public static String encodeURL(String url);

    @Positive
    public static HtmlTree A(String ref, Content body);

    @Positive
    public static HtmlTree CAPTION(Content body);

    @Positive
    public static HtmlTree CODE(Content body);

    @Positive
    public static HtmlTree DD(Content body);

    @Positive
    public static HtmlTree DL(HtmlStyle style);

    @Positive
    public static HtmlTree DL(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree DIV(HtmlStyle style);

    @Positive
    public static HtmlTree DIV(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree DIV(Content body);

    @Positive
    public static HtmlTree DT(Content body);

    @Positive
    public static HtmlTree FOOTER();

    @Positive
    public static HtmlTree HEADER();

    @Positive
    public static HtmlTree HEADING(TagName headingTag, Content body);

    @Positive
    public static HtmlTree HEADING(TagName headingTag, HtmlStyle style, Content body);

    @Positive
    public static HtmlTree HEADING_TITLE(TagName headingTag, HtmlStyle style, Content body);

    @Positive
    public static HtmlTree HEADING_TITLE(TagName headingTag, Content body);

    @Positive
    public static HtmlTree HTML(String lang, Content head, Content body);

    @Positive
    public static HtmlTree INPUT(String type, HtmlId id, String value);

    @Positive
    public static HtmlTree LABEL(String forLabel, Content body);

    @Positive
    public static HtmlTree LI(Content body);

    @Positive
    public static HtmlTree LI(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree LINK(String rel, String type, String href, String title);

    @Positive
    public static HtmlTree MAIN();

    @Positive
    public static HtmlTree MAIN(Content body);

    @Positive
    public static HtmlTree META(String httpEquiv, String content, String charset);

    @Positive
    public static HtmlTree META(String name, String content);

    @Positive
    public static HtmlTree NAV();

    @Positive
    public static HtmlTree NOSCRIPT(Content body);

    @Positive
    public static HtmlTree P(Content body);

    @Positive
    public static HtmlTree P(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree SCRIPT(String src);

    @Positive
    public static HtmlTree SECTION(HtmlStyle style);

    @Positive
    public static HtmlTree SECTION(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree SMALL(Content body);

    @Positive
    public static HtmlTree SPAN(Content body);

    @Positive
    public static HtmlTree SPAN(HtmlStyle styleClass, Content body);

    @Positive
    public static HtmlTree SPAN_ID(HtmlId id, Content body);

    @Positive
    public static HtmlTree SPAN(HtmlId id, HtmlStyle style, Content body);

    @Positive
    public static HtmlTree SUP(Content body);

    @Positive
    public static HtmlTree TD(HtmlStyle style, Content body);

    @Positive
    public static HtmlTree TH(HtmlStyle style, String scope, Content body);

    @Positive
    public static HtmlTree TH(String scope, Content body);

    @Positive
    public static HtmlTree TITLE(String body);

    @Positive
    public static HtmlTree UL(HtmlStyle style, Content first, Content... more);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean isEmpty();

    @Positive
    public boolean hasContent();

    @Positive
    public boolean hasAttrs();

    @Positive
    public boolean hasAttr(HtmlAttr attrName);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean isValid();

    @Positive
    @Pure
    @Positive
    public boolean isInline();

    @Positive
    public boolean isVoid();

    @Positive
    @Override
    @Positive
    public boolean write(Writer out, boolean atNewline) throws IOException;
    @Positive
}
