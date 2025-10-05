/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package javax.swing.text.html;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Hashtable;
    @Positive
import javax.swing.text.AttributeSet;
    @Positive
import javax.swing.text.StyleConstants;
    @Positive
import javax.swing.text.StyleContext;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class HTML {

    @Positive
    public HTML() {
    @Positive
    }

    @Positive
    public static class Tag {

    @Positive
        public Tag() {
    @Positive
        }

    @Positive
        protected Tag(String id) {
    @Positive
        }

    @Positive
        protected Tag(String id, boolean causesBreak, boolean isBlock) {
    @Positive
        }

    @Positive
        public boolean isBlock();

    @Positive
        public boolean breaksFlow();

    @Positive
        public boolean isPreformatted();

    @Positive
        public String toString();

    @Positive
        boolean isParagraph();

    @Positive
        public static final Tag A;

    @Positive
        public static final Tag ADDRESS;

    @Positive
        public static final Tag APPLET;

    @Positive
        public static final Tag AREA;

    @Positive
        public static final Tag B;

    @Positive
        public static final Tag BASE;

    @Positive
        public static final Tag BASEFONT;

    @Positive
        public static final Tag BIG;

    @Positive
        public static final Tag BLOCKQUOTE;

    @Positive
        public static final Tag BODY;

    @Positive
        public static final Tag BR;

    @Positive
        public static final Tag CAPTION;

    @Positive
        public static final Tag CENTER;

    @Positive
        public static final Tag CITE;

    @Positive
        public static final Tag CODE;

    @Positive
        public static final Tag DD;

    @Positive
        public static final Tag DFN;

    @Positive
        public static final Tag DIR;

    @Positive
        public static final Tag DIV;

    @Positive
        public static final Tag DL;

    @Positive
        public static final Tag DT;

    @Positive
        public static final Tag EM;

    @Positive
        public static final Tag FONT;

    @Positive
        public static final Tag FORM;

    @Positive
        public static final Tag FRAME;

    @Positive
        public static final Tag FRAMESET;

    @Positive
        public static final Tag H1;

    @Positive
        public static final Tag H2;

    @Positive
        public static final Tag H3;

    @Positive
        public static final Tag H4;

    @Positive
        public static final Tag H5;

    @Positive
        public static final Tag H6;

    @Positive
        public static final Tag HEAD;

    @Positive
        public static final Tag HR;

    @Positive
        public static final Tag HTML;

    @Positive
        public static final Tag I;

    @Positive
        public static final Tag IMG;

    @Positive
        public static final Tag INPUT;

    @Positive
        public static final Tag ISINDEX;

    @Positive
        public static final Tag KBD;

    @Positive
        public static final Tag LI;

    @Positive
        public static final Tag LINK;

    @Positive
        public static final Tag MAP;

    @Positive
        public static final Tag MENU;

    @Positive
        public static final Tag META;

    @Positive
        public static final Tag NOFRAMES;

    @Positive
        public static final Tag OBJECT;

    @Positive
        public static final Tag OL;

    @Positive
        public static final Tag OPTION;

    @Positive
        public static final Tag P;

    @Positive
        public static final Tag PARAM;

    @Positive
        public static final Tag PRE;

    @Positive
        public static final Tag SAMP;

    @Positive
        public static final Tag SCRIPT;

    @Positive
        public static final Tag SELECT;

    @Positive
        public static final Tag SMALL;

    @Positive
        public static final Tag SPAN;

    @Positive
        public static final Tag STRIKE;

    @Positive
        public static final Tag S;

    @Positive
        public static final Tag STRONG;

    @Positive
        public static final Tag STYLE;

    @Positive
        public static final Tag SUB;

    @Positive
        public static final Tag SUP;

    @Positive
        public static final Tag TABLE;

    @Positive
        public static final Tag TD;

    @Positive
        public static final Tag TEXTAREA;

    @Positive
        public static final Tag TH;

    @Positive
        public static final Tag TITLE;

    @Positive
        public static final Tag TR;

    @Positive
        public static final Tag TT;

    @Positive
        public static final Tag U;

    @Positive
        public static final Tag UL;

    @Positive
        public static final Tag VAR;

    @Positive
        public static final Tag IMPLIED;

    @Positive
        public static final Tag CONTENT;

    @Positive
        public static final Tag COMMENT;
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class UnknownTag extends Tag implements Serializable {

    @Positive
        public UnknownTag(String id) {
    @Positive
        }

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    public static final class Attribute {

    @Positive
        public String toString();

    @Positive
        public static final Attribute SIZE;

    @Positive
        public static final Attribute COLOR;

    @Positive
        public static final Attribute CLEAR;

    @Positive
        public static final Attribute BACKGROUND;

    @Positive
        public static final Attribute BGCOLOR;

    @Positive
        public static final Attribute TEXT;

    @Positive
        public static final Attribute LINK;

    @Positive
        public static final Attribute VLINK;

    @Positive
        public static final Attribute ALINK;

    @Positive
        public static final Attribute WIDTH;

    @Positive
        public static final Attribute HEIGHT;

    @Positive
        public static final Attribute ALIGN;

    @Positive
        public static final Attribute NAME;

    @Positive
        public static final Attribute HREF;

    @Positive
        public static final Attribute REL;

    @Positive
        public static final Attribute REV;

    @Positive
        public static final Attribute TITLE;

    @Positive
        public static final Attribute TARGET;

    @Positive
        public static final Attribute SHAPE;

    @Positive
        public static final Attribute COORDS;

    @Positive
        public static final Attribute ISMAP;

    @Positive
        public static final Attribute NOHREF;

    @Positive
        public static final Attribute ALT;

    @Positive
        public static final Attribute ID;

    @Positive
        public static final Attribute SRC;

    @Positive
        public static final Attribute HSPACE;

    @Positive
        public static final Attribute VSPACE;

    @Positive
        public static final Attribute USEMAP;

    @Positive
        public static final Attribute LOWSRC;

    @Positive
        public static final Attribute CODEBASE;

    @Positive
        public static final Attribute CODE;

    @Positive
        public static final Attribute ARCHIVE;

    @Positive
        public static final Attribute VALUE;

    @Positive
        public static final Attribute VALUETYPE;

    @Positive
        public static final Attribute TYPE;

    @Positive
        public static final Attribute CLASS;

    @Positive
        public static final Attribute STYLE;

    @Positive
        public static final Attribute LANG;

    @Positive
        public static final Attribute FACE;

    @Positive
        public static final Attribute DIR;

    @Positive
        public static final Attribute DECLARE;

    @Positive
        public static final Attribute CLASSID;

    @Positive
        public static final Attribute DATA;

    @Positive
        public static final Attribute CODETYPE;

    @Positive
        public static final Attribute STANDBY;

    @Positive
        public static final Attribute BORDER;

    @Positive
        public static final Attribute SHAPES;

    @Positive
        public static final Attribute NOSHADE;

    @Positive
        public static final Attribute COMPACT;

    @Positive
        public static final Attribute START;

    @Positive
        public static final Attribute ACTION;

    @Positive
        public static final Attribute METHOD;

    @Positive
        public static final Attribute ENCTYPE;

    @Positive
        public static final Attribute CHECKED;

    @Positive
        public static final Attribute MAXLENGTH;

    @Positive
        public static final Attribute MULTIPLE;

    @Positive
        public static final Attribute SELECTED;

    @Positive
        public static final Attribute ROWS;

    @Positive
        public static final Attribute COLS;

    @Positive
        public static final Attribute DUMMY;

    @Positive
        public static final Attribute CELLSPACING;

    @Positive
        public static final Attribute CELLPADDING;

    @Positive
        public static final Attribute VALIGN;

    @Positive
        public static final Attribute HALIGN;

    @Positive
        public static final Attribute NOWRAP;

    @Positive
        public static final Attribute ROWSPAN;

    @Positive
        public static final Attribute COLSPAN;

    @Positive
        public static final Attribute PROMPT;

    @Positive
        public static final Attribute HTTPEQUIV;

    @Positive
        public static final Attribute CONTENT;

    @Positive
        public static final Attribute LANGUAGE;

    @Positive
        public static final Attribute VERSION;

    @Positive
        public static final Attribute N;

    @Positive
        public static final Attribute FRAMEBORDER;

    @Positive
        public static final Attribute MARGINWIDTH;

    @Positive
        public static final Attribute MARGINHEIGHT;

    @Positive
        public static final Attribute SCROLLING;

    @Positive
        public static final Attribute NORESIZE;

    @Positive
        public static final Attribute ENDTAG;

    @Positive
        public static final Attribute COMMENT;
    @Positive
    }

    @Positive
    public static Tag[] getAllTags();

    @Positive
    public static Tag getTag(String tagName);

    @Positive
    static Tag getTagForStyleConstantsKey(StyleConstants sc);

    @Positive
    public static int getIntegerAttributeValue(AttributeSet attr, Attribute key, int def);

    @Positive
    @Interned
    @Positive
    public static final String NULL_ATTRIBUTE_VALUE;

    @Positive
    public static Attribute[] getAllAttributeKeys();

    @Positive
    public static Attribute getAttributeKey(String attName);
    @Positive
}
