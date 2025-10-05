/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import sun.awt.AppContext;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.io.*;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URL;
    @Positive
import javax.swing.text.*;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.plaf.TextUI;
    @Positive
import java.util.*;
    @Positive
import javax.accessibility.*;
    @Positive
import java.lang.ref.*;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import javax.swing.text.html.parser.ParserDelegator;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class HTMLEditorKit extends StyledEditorKit implements Accessible {

    @Positive
    public HTMLEditorKit() {
    @Positive
    }

    @Positive
    public String getContentType();

    @Positive
    public ViewFactory getViewFactory();

    @Positive
    public Document createDefaultDocument();

    @Positive
    public void read(Reader in, Document doc, int pos) throws IOException, BadLocationException;

    @Positive
    public void insertHTML(HTMLDocument doc, int offset, String html, int popDepth, int pushDepth, HTML.Tag insertTag) throws BadLocationException, IOException;

    @Positive
    public void write(Writer out, Document doc, int pos, int len) throws IOException, BadLocationException;

    @Positive
    public void install(JEditorPane c);

    @Positive
    public void deinstall(JEditorPane c);

    @Positive
    @Interned
    @Positive
    public static final String DEFAULT_CSS;

    @Positive
    public void setStyleSheet(StyleSheet s);

    @Positive
    public StyleSheet getStyleSheet();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static InputStream getResourceAsStream(final String name);

    @Positive
    public Action[] getActions();

    @Positive
    protected void createInputAttributes(Element element, MutableAttributeSet set);

    @Positive
    public MutableAttributeSet getInputAttributes();

    @Positive
    public void setDefaultCursor(Cursor cursor);

    @Positive
    public Cursor getDefaultCursor();

    @Positive
    public void setLinkCursor(Cursor cursor);

    @Positive
    public Cursor getLinkCursor();

    @Positive
    public boolean isAutoFormSubmission();

    @Positive
    public void setAutoFormSubmission(boolean isAuto);

    @Positive
    public Object clone();

    @Positive
    protected Parser getParser();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class LinkController extends MouseAdapter implements MouseMotionListener, Serializable {

    @Positive
        public LinkController() {
    @Positive
        }

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void mouseClicked(MouseEvent e);

    @Positive
        public void mouseDragged(MouseEvent e);

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void mouseMoved(MouseEvent e);

    @Positive
        protected void activateLink(int pos, JEditorPane editor);

    @Positive
        void activateLink(int pos, JEditorPane html, MouseEvent mouseEvent);

    @Positive
        HyperlinkEvent createHyperlinkEvent(JEditorPane html, HTMLDocument hdoc, String href, AttributeSet anchor, Element element, MouseEvent mouseEvent);

    @Positive
        void fireEvents(JEditorPane editor, HTMLDocument doc, String href, Element lastElem, MouseEvent mouseEvent);
    @Positive
    }

    @Positive
    public abstract static class Parser {

    @Positive
        protected Parser() {
    @Positive
        }

    @Positive
        public abstract void parse(Reader r, ParserCallback cb, boolean ignoreCharSet) throws IOException;
    @Positive
    }

    @Positive
    public static class ParserCallback {

    @Positive
        public ParserCallback() {
    @Positive
        }

    @Positive
        public static final Object IMPLIED;

    @Positive
        public void flush() throws BadLocationException;

    @Positive
        public void handleText(char[] data, int pos);

    @Positive
        public void handleComment(char[] data, int pos);

    @Positive
        public void handleStartTag(HTML.Tag t, MutableAttributeSet a, int pos);

    @Positive
        public void handleEndTag(HTML.Tag t, int pos);

    @Positive
        public void handleSimpleTag(HTML.Tag t, MutableAttributeSet a, int pos);

    @Positive
        public void handleError(String errorMsg, int pos);

    @Positive
        public void handleEndOfLineString(String eol);
    @Positive
    }

    @Positive
    public static class HTMLFactory implements ViewFactory {

    @Positive
        public HTMLFactory() {
    @Positive
        }

    @Positive
        public View create(Element elem);

    @Positive
        static class BodyBlockView extends BlockView implements ComponentListener {

    @Positive
            public BodyBlockView(Element elem) {
    @Positive
            }

    @Positive
            protected SizeRequirements calculateMajorAxisRequirements(int axis, SizeRequirements r);

    @Positive
            protected void layoutMinorAxis(int targetSpan, int axis, int[] offsets, int[] spans);

    @Positive
            public void setParent(View parent);

    @Positive
            public void componentResized(ComponentEvent e);

    @Positive
            public void componentHidden(ComponentEvent e);

    @Positive
            public void componentMoved(ComponentEvent e);

    @Positive
            public void componentShown(ComponentEvent e);
    @Positive
        }
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final String BOLD_ACTION;

    @Positive
    @Interned
    @Positive
    public static final String ITALIC_ACTION;

    @Positive
    @Interned
    @Positive
    public static final String PARA_INDENT_LEFT;

    @Positive
    @Interned
    @Positive
    public static final String PARA_INDENT_RIGHT;

    @Positive
    @Interned
    @Positive
    public static final String FONT_CHANGE_BIGGER;

    @Positive
    @Interned
    @Positive
    public static final String FONT_CHANGE_SMALLER;

    @Positive
    @Interned
    @Positive
    public static final String COLOR_ACTION;

    @Positive
    @Interned
    @Positive
    public static final String LOGICAL_STYLE_ACTION;

    @Positive
    @Interned
    @Positive
    public static final String IMG_ALIGN_TOP;

    @Positive
    @Interned
    @Positive
    public static final String IMG_ALIGN_MIDDLE;

    @Positive
    @Interned
    @Positive
    public static final String IMG_ALIGN_BOTTOM;

    @Positive
    @Interned
    @Positive
    public static final String IMG_BORDER;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public abstract static class HTMLTextAction extends StyledTextAction {

    @Positive
        public HTMLTextAction(String name) {
    @Positive
        }

    @Positive
        protected HTMLDocument getHTMLDocument(JEditorPane e);

    @Positive
        protected HTMLEditorKit getHTMLEditorKit(JEditorPane e);

    @Positive
        protected Element[] getElementsAt(HTMLDocument doc, int offset);

    @Positive
        protected int elementCountToTag(HTMLDocument doc, int offset, HTML.Tag tag);

    @Positive
        protected Element findElementMatchingTag(HTMLDocument doc, int offset, HTML.Tag tag);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class InsertHTMLTextAction extends HTMLTextAction {

    @Positive
        public InsertHTMLTextAction(String name, String html, HTML.Tag parentTag, HTML.Tag addTag) {
    @Positive
        }

    @Positive
        public InsertHTMLTextAction(String name, String html, HTML.Tag parentTag, HTML.Tag addTag, HTML.Tag alternateParentTag, HTML.Tag alternateAddTag) {
    @Positive
        }

    @Positive
        protected void insertHTML(JEditorPane editor, HTMLDocument doc, int offset, String html, int popDepth, int pushDepth, HTML.Tag addTag);

    @Positive
        protected void insertAtBoundary(JEditorPane editor, HTMLDocument doc, int offset, Element insertElement, String html, HTML.Tag parentTag, HTML.Tag addTag);

    @Positive
        @Deprecated
    @Positive
        protected void insertAtBoundry(JEditorPane editor, HTMLDocument doc, int offset, Element insertElement, String html, HTML.Tag parentTag, HTML.Tag addTag);

    @Positive
        boolean insertIntoTag(JEditorPane editor, HTMLDocument doc, int offset, HTML.Tag tag, HTML.Tag addTag);

    @Positive
        void adjustSelection(JEditorPane pane, HTMLDocument doc, int startOffset, int oldLength);

    @Positive
        public void actionPerformed(ActionEvent ae);

    @Positive
        protected String html;

    @Positive
        protected HTML.Tag parentTag;

    @Positive
        protected HTML.Tag addTag;

    @Positive
        protected HTML.Tag alternateParentTag;

    @Positive
        protected HTML.Tag alternateAddTag;
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class InsertHRAction extends InsertHTMLTextAction {

    @Positive
        public void actionPerformed(ActionEvent ae);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class NavigateLinkAction extends TextAction implements CaretListener {

    @Positive
        public NavigateLinkAction(String actionName) {
    @Positive
        }

    @Positive
        public void caretUpdate(CaretEvent e);

    @Positive
        public void actionPerformed(ActionEvent e);

    @Positive
        static class FocusHighlightPainter extends DefaultHighlighter.DefaultHighlightPainter {

    @Positive
            public Shape paintLayer(Graphics g, int offs0, int offs1, Shape bounds, JTextComponent c, View view);
    @Positive
        }
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class ActivateLinkAction extends TextAction {

    @Positive
        public ActivateLinkAction(String actionName) {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BeginAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }
    @Positive
}
