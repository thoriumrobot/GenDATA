/*
 * Copyright (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.awt.*;
import java.net.URL;
import javax.accessibility.AccessibleContext;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.text.html.*;
import sun.swing.SwingUtilities2;

@AnnotatedFor({ "interning" })
public class BasicHTML {

    public BasicHTML() {
    }

    public static View createHTMLView(JComponent c, String html);

    public static int getHTMLBaseline(View view, int w, int h);

    static int getBaseline(JComponent c, int y, int ascent, int w, int h);

    static int getBaseline(View view, int w, int h);

    public static boolean isHTMLString(String s);

    public static void updateRenderer(JComponent c, String text);

    @Interned
    public static final String propertyKey;

    @Interned
    public static final String documentBaseKey;

    static BasicEditorKit getFactory();

    @SuppressWarnings("serial")
    static class BasicEditorKit extends HTMLEditorKit {

        public StyleSheet getStyleSheet();

        public Document createDefaultDocument(Font defaultFont, Color foreground);

        public ViewFactory getViewFactory();
    }

    static class BasicHTMLViewFactory extends HTMLEditorKit.HTMLFactory {

        public View create(Element elem);
    }

    @SuppressWarnings("serial")
    static class BasicDocument extends HTMLDocument {
    }

    static class Renderer extends View {

        public AttributeSet getAttributes();

        public float getPreferredSpan(int axis);

        public float getMinimumSpan(int axis);

        public float getMaximumSpan(int axis);

        public void preferenceChanged(View child, boolean width, boolean height);

        public float getAlignment(int axis);

        public void paint(Graphics g, Shape allocation);

        public void setParent(View parent);

        public int getViewCount();

        public View getView(int n);

        public Shape modelToView(int pos, Shape a, Position.Bias b) throws BadLocationException;

        public Shape modelToView(int p0, Position.Bias b0, int p1, Position.Bias b1, Shape a) throws BadLocationException;

        public int viewToModel(float x, float y, Shape a, Position.Bias[] bias);

        public Document getDocument();

        public int getStartOffset();

        public int getEndOffset();

        public Element getElement();

        public void setSize(float width, float height);

        public Container getContainer();

        public ViewFactory getViewFactory();
    }
}
