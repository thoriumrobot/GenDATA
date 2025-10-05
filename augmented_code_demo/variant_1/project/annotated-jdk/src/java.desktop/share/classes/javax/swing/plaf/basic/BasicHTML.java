/*
    @Positive
 * Copyright (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.awt.*;
    @Positive
import java.net.URL;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.text.*;
    @Positive
import javax.swing.text.html.*;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class BasicHTML {

    @Positive
    public BasicHTML() {
    @Positive
    }

    @Positive
    public static View createHTMLView(JComponent c, String html);

    @Positive
    public static int getHTMLBaseline(View view, int w, int h);

    @Positive
    static int getBaseline(JComponent c, int y, int ascent, int w, int h);

    @Positive
    static int getBaseline(View view, int w, int h);

    @Positive
    public static boolean isHTMLString(String s);

    @Positive
    public static void updateRenderer(JComponent c, String text);

    @Positive
    @Interned
    @Positive
    public static final String propertyKey;

    @Positive
    @Interned
    @Positive
    public static final String documentBaseKey;

    @Positive
    static BasicEditorKit getFactory();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BasicEditorKit extends HTMLEditorKit {

    @Positive
        public StyleSheet getStyleSheet();

    @Positive
        public Document createDefaultDocument(Font defaultFont, Color foreground);

    @Positive
        public ViewFactory getViewFactory();
    @Positive
    }

    @Positive
    static class BasicHTMLViewFactory extends HTMLEditorKit.HTMLFactory {

    @Positive
        public View create(Element elem);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BasicDocument extends HTMLDocument {
    @Positive
    }

    @Positive
    static class Renderer extends View {

    @Positive
        public AttributeSet getAttributes();

    @Positive
        public float getPreferredSpan(int axis);

    @Positive
        public float getMinimumSpan(int axis);

    @Positive
        public float getMaximumSpan(int axis);

    @Positive
        public void preferenceChanged(View child, boolean width, boolean height);

    @Positive
        public float getAlignment(int axis);

    @Positive
        public void paint(Graphics g, Shape allocation);

    @Positive
        public void setParent(View parent);

    @Positive
        public int getViewCount();

    @Positive
        public View getView(int n);

    @Positive
        public Shape modelToView(int pos, Shape a, Position.Bias b) throws BadLocationException;

    @Positive
        public Shape modelToView(int p0, Position.Bias b0, int p1, Position.Bias b1, Shape a) throws BadLocationException;

    @Positive
        public int viewToModel(float x, float y, Shape a, Position.Bias[] bias);

    @Positive
        public Document getDocument();

    @Positive
        public int getStartOffset();

    @Positive
        public int getEndOffset();

    @Positive
        public Element getElement();

    @Positive
        public void setSize(float width, float height);

    @Positive
        public Container getContainer();

    @Positive
        public ViewFactory getViewFactory();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
