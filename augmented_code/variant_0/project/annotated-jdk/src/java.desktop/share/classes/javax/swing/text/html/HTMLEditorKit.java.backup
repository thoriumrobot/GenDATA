/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import sun.awt.AppContext;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import javax.swing.text.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.plaf.TextUI;
import java.util.*;
import javax.accessibility.*;
import java.lang.ref.*;
import java.security.AccessController;
import java.security.PrivilegedAction;
import javax.swing.text.html.parser.ParserDelegator;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class HTMLEditorKit extends StyledEditorKit implements Accessible {

    public HTMLEditorKit() {
    }

    public String getContentType();

    public ViewFactory getViewFactory();

    public Document createDefaultDocument();

    public void read(Reader in, Document doc, int pos) throws IOException, BadLocationException;

    public void insertHTML(HTMLDocument doc, int offset, String html, int popDepth, int pushDepth, HTML.Tag insertTag) throws BadLocationException, IOException;

    public void write(Writer out, Document doc, int pos, int len) throws IOException, BadLocationException;

    public void install(JEditorPane c);

    public void deinstall(JEditorPane c);

    @Interned
    public static final String DEFAULT_CSS;

    public void setStyleSheet(StyleSheet s);

    public StyleSheet getStyleSheet();

    @SuppressWarnings("removal")
    static InputStream getResourceAsStream(final String name);

    public Action[] getActions();

    protected void createInputAttributes(Element element, MutableAttributeSet set);

    public MutableAttributeSet getInputAttributes();

    public void setDefaultCursor(Cursor cursor);

    public Cursor getDefaultCursor();

    public void setLinkCursor(Cursor cursor);

    public Cursor getLinkCursor();

    public boolean isAutoFormSubmission();

    public void setAutoFormSubmission(boolean isAuto);

    public Object clone();

    protected Parser getParser();

    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    public static class LinkController extends MouseAdapter implements MouseMotionListener, Serializable {

        public LinkController() {
        }

        @SuppressWarnings("deprecation")
        public void mouseClicked(MouseEvent e);

        public void mouseDragged(MouseEvent e);

        @SuppressWarnings("deprecation")
        public void mouseMoved(MouseEvent e);

        protected void activateLink(int pos, JEditorPane editor);

        void activateLink(int pos, JEditorPane html, MouseEvent mouseEvent);

        HyperlinkEvent createHyperlinkEvent(JEditorPane html, HTMLDocument hdoc, String href, AttributeSet anchor, Element element, MouseEvent mouseEvent);

        void fireEvents(JEditorPane editor, HTMLDocument doc, String href, Element lastElem, MouseEvent mouseEvent);
    }

    public abstract static class Parser {

        protected Parser() {
        }

        public abstract void parse(Reader r, ParserCallback cb, boolean ignoreCharSet) throws IOException;
    }

    public static class ParserCallback {

        public ParserCallback() {
        }

        public static final Object IMPLIED;

        public void flush() throws BadLocationException;

        public void handleText(char[] data, int pos);

        public void handleComment(char[] data, int pos);

        public void handleStartTag(HTML.Tag t, MutableAttributeSet a, int pos);

        public void handleEndTag(HTML.Tag t, int pos);

        public void handleSimpleTag(HTML.Tag t, MutableAttributeSet a, int pos);

        public void handleError(String errorMsg, int pos);

        public void handleEndOfLineString(String eol);
    }

    public static class HTMLFactory implements ViewFactory {

        public HTMLFactory() {
        }

        public View create(Element elem);

        static class BodyBlockView extends BlockView implements ComponentListener {

            public BodyBlockView(Element elem) {
            }

            protected SizeRequirements calculateMajorAxisRequirements(int axis, SizeRequirements r);

            protected void layoutMinorAxis(int targetSpan, int axis, int[] offsets, int[] spans);

            public void setParent(View parent);

            public void componentResized(ComponentEvent e);

            public void componentHidden(ComponentEvent e);

            public void componentMoved(ComponentEvent e);

            public void componentShown(ComponentEvent e);
        }
    }

    @Interned
    public static final String BOLD_ACTION;

    @Interned
    public static final String ITALIC_ACTION;

    @Interned
    public static final String PARA_INDENT_LEFT;

    @Interned
    public static final String PARA_INDENT_RIGHT;

    @Interned
    public static final String FONT_CHANGE_BIGGER;

    @Interned
    public static final String FONT_CHANGE_SMALLER;

    @Interned
    public static final String COLOR_ACTION;

    @Interned
    public static final String LOGICAL_STYLE_ACTION;

    @Interned
    public static final String IMG_ALIGN_TOP;

    @Interned
    public static final String IMG_ALIGN_MIDDLE;

    @Interned
    public static final String IMG_ALIGN_BOTTOM;

    @Interned
    public static final String IMG_BORDER;

    @SuppressWarnings("serial")
    public abstract static class HTMLTextAction extends StyledTextAction {

        public HTMLTextAction(String name) {
        }

        protected HTMLDocument getHTMLDocument(JEditorPane e);

        protected HTMLEditorKit getHTMLEditorKit(JEditorPane e);

        protected Element[] getElementsAt(HTMLDocument doc, int offset);

        protected int elementCountToTag(HTMLDocument doc, int offset, HTML.Tag tag);

        protected Element findElementMatchingTag(HTMLDocument doc, int offset, HTML.Tag tag);
    }

    @SuppressWarnings("serial")
    public static class InsertHTMLTextAction extends HTMLTextAction {

        public InsertHTMLTextAction(String name, String html, HTML.Tag parentTag, HTML.Tag addTag) {
        }

        public InsertHTMLTextAction(String name, String html, HTML.Tag parentTag, HTML.Tag addTag, HTML.Tag alternateParentTag, HTML.Tag alternateAddTag) {
        }

        protected void insertHTML(JEditorPane editor, HTMLDocument doc, int offset, String html, int popDepth, int pushDepth, HTML.Tag addTag);

        protected void insertAtBoundary(JEditorPane editor, HTMLDocument doc, int offset, Element insertElement, String html, HTML.Tag parentTag, HTML.Tag addTag);

        @Deprecated
        protected void insertAtBoundry(JEditorPane editor, HTMLDocument doc, int offset, Element insertElement, String html, HTML.Tag parentTag, HTML.Tag addTag);

        boolean insertIntoTag(JEditorPane editor, HTMLDocument doc, int offset, HTML.Tag tag, HTML.Tag addTag);

        void adjustSelection(JEditorPane pane, HTMLDocument doc, int startOffset, int oldLength);

        public void actionPerformed(ActionEvent ae);

        protected String html;

        protected HTML.Tag parentTag;

        protected HTML.Tag addTag;

        protected HTML.Tag alternateParentTag;

        protected HTML.Tag alternateAddTag;
    }

    @SuppressWarnings("serial")
    static class InsertHRAction extends InsertHTMLTextAction {

        public void actionPerformed(ActionEvent ae);
    }

    @SuppressWarnings("serial")
    static class NavigateLinkAction extends TextAction implements CaretListener {

        public NavigateLinkAction(String actionName) {
        }

        public void caretUpdate(CaretEvent e);

        public void actionPerformed(ActionEvent e);

        static class FocusHighlightPainter extends DefaultHighlighter.DefaultHighlightPainter {

            public Shape paintLayer(Graphics g, int offs0, int offs1, Shape bounds, JTextComponent c, View view);
        }
    }

    @SuppressWarnings("serial")
    static class ActivateLinkAction extends TextAction {

        public ActivateLinkAction(String actionName) {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class BeginAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }
}
