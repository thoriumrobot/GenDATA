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
package javax.swing.text.html;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Hashtable;
import javax.swing.text.AttributeSet;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyleContext;

@AnnotatedFor({ "interning" })
public class HTML {

    public HTML() {
    }

    public static class Tag {

        public Tag() {
        }

        protected Tag(String id) {
        }

        protected Tag(String id, boolean causesBreak, boolean isBlock) {
        }

        public boolean isBlock();

        public boolean breaksFlow();

        public boolean isPreformatted();

        public String toString();

        boolean isParagraph();

        public static final Tag A;

        public static final Tag ADDRESS;

        public static final Tag APPLET;

        public static final Tag AREA;

        public static final Tag B;

        public static final Tag BASE;

        public static final Tag BASEFONT;

        public static final Tag BIG;

        public static final Tag BLOCKQUOTE;

        public static final Tag BODY;

        public static final Tag BR;

        public static final Tag CAPTION;

        public static final Tag CENTER;

        public static final Tag CITE;

        public static final Tag CODE;

        public static final Tag DD;

        public static final Tag DFN;

        public static final Tag DIR;

        public static final Tag DIV;

        public static final Tag DL;

        public static final Tag DT;

        public static final Tag EM;

        public static final Tag FONT;

        public static final Tag FORM;

        public static final Tag FRAME;

        public static final Tag FRAMESET;

        public static final Tag H1;

        public static final Tag H2;

        public static final Tag H3;

        public static final Tag H4;

        public static final Tag H5;

        public static final Tag H6;

        public static final Tag HEAD;

        public static final Tag HR;

        public static final Tag HTML;

        public static final Tag I;

        public static final Tag IMG;

        public static final Tag INPUT;

        public static final Tag ISINDEX;

        public static final Tag KBD;

        public static final Tag LI;

        public static final Tag LINK;

        public static final Tag MAP;

        public static final Tag MENU;

        public static final Tag META;

        public static final Tag NOFRAMES;

        public static final Tag OBJECT;

        public static final Tag OL;

        public static final Tag OPTION;

        public static final Tag P;

        public static final Tag PARAM;

        public static final Tag PRE;

        public static final Tag SAMP;

        public static final Tag SCRIPT;

        public static final Tag SELECT;

        public static final Tag SMALL;

        public static final Tag SPAN;

        public static final Tag STRIKE;

        public static final Tag S;

        public static final Tag STRONG;

        public static final Tag STYLE;

        public static final Tag SUB;

        public static final Tag SUP;

        public static final Tag TABLE;

        public static final Tag TD;

        public static final Tag TEXTAREA;

        public static final Tag TH;

        public static final Tag TITLE;

        public static final Tag TR;

        public static final Tag TT;

        public static final Tag U;

        public static final Tag UL;

        public static final Tag VAR;

        public static final Tag IMPLIED;

        public static final Tag CONTENT;

        public static final Tag COMMENT;
    }

    @SuppressWarnings("serial")
    public static class UnknownTag extends Tag implements Serializable {

        public UnknownTag(String id) {
        }

        public int hashCode();

        public boolean equals(Object obj);
    }

    public static final class Attribute {

        public String toString();

        public static final Attribute SIZE;

        public static final Attribute COLOR;

        public static final Attribute CLEAR;

        public static final Attribute BACKGROUND;

        public static final Attribute BGCOLOR;

        public static final Attribute TEXT;

        public static final Attribute LINK;

        public static final Attribute VLINK;

        public static final Attribute ALINK;

        public static final Attribute WIDTH;

        public static final Attribute HEIGHT;

        public static final Attribute ALIGN;

        public static final Attribute NAME;

        public static final Attribute HREF;

        public static final Attribute REL;

        public static final Attribute REV;

        public static final Attribute TITLE;

        public static final Attribute TARGET;

        public static final Attribute SHAPE;

        public static final Attribute COORDS;

        public static final Attribute ISMAP;

        public static final Attribute NOHREF;

        public static final Attribute ALT;

        public static final Attribute ID;

        public static final Attribute SRC;

        public static final Attribute HSPACE;

        public static final Attribute VSPACE;

        public static final Attribute USEMAP;

        public static final Attribute LOWSRC;

        public static final Attribute CODEBASE;

        public static final Attribute CODE;

        public static final Attribute ARCHIVE;

        public static final Attribute VALUE;

        public static final Attribute VALUETYPE;

        public static final Attribute TYPE;

        public static final Attribute CLASS;

        public static final Attribute STYLE;

        public static final Attribute LANG;

        public static final Attribute FACE;

        public static final Attribute DIR;

        public static final Attribute DECLARE;

        public static final Attribute CLASSID;

        public static final Attribute DATA;

        public static final Attribute CODETYPE;

        public static final Attribute STANDBY;

        public static final Attribute BORDER;

        public static final Attribute SHAPES;

        public static final Attribute NOSHADE;

        public static final Attribute COMPACT;

        public static final Attribute START;

        public static final Attribute ACTION;

        public static final Attribute METHOD;

        public static final Attribute ENCTYPE;

        public static final Attribute CHECKED;

        public static final Attribute MAXLENGTH;

        public static final Attribute MULTIPLE;

        public static final Attribute SELECTED;

        public static final Attribute ROWS;

        public static final Attribute COLS;

        public static final Attribute DUMMY;

        public static final Attribute CELLSPACING;

        public static final Attribute CELLPADDING;

        public static final Attribute VALIGN;

        public static final Attribute HALIGN;

        public static final Attribute NOWRAP;

        public static final Attribute ROWSPAN;

        public static final Attribute COLSPAN;

        public static final Attribute PROMPT;

        public static final Attribute HTTPEQUIV;

        public static final Attribute CONTENT;

        public static final Attribute LANGUAGE;

        public static final Attribute VERSION;

        public static final Attribute N;

        public static final Attribute FRAMEBORDER;

        public static final Attribute MARGINWIDTH;

        public static final Attribute MARGINHEIGHT;

        public static final Attribute SCROLLING;

        public static final Attribute NORESIZE;

        public static final Attribute ENDTAG;

        public static final Attribute COMMENT;
    }

    public static Tag[] getAllTags();

    public static Tag getTag(String tagName);

    static Tag getTagForStyleConstantsKey(StyleConstants sc);

    public static int getIntegerAttributeValue(AttributeSet attr, Attribute key, int def);

    @Interned
    public static final String NULL_ATTRIBUTE_VALUE;

    public static Attribute[] getAllAttributeKeys();

    public static Attribute getAttributeKey(String attName);
}
