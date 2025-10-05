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
package javax.swing;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.IllegalComponentStateException;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Shape;
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.io.BufferedInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.Serial;
import java.io.StringReader;
import java.io.StringWriter;
import java.lang.reflect.InvocationTargetException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.Vector;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleComponent;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleHyperlink;
import javax.accessibility.AccessibleHypertext;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import javax.accessibility.AccessibleText;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.event.EventListenerList;
import javax.swing.event.HyperlinkEvent;
import javax.swing.event.HyperlinkListener;
import javax.swing.plaf.TextUI;
import javax.swing.text.AbstractDocument;
import javax.swing.text.AttributeSet;
import javax.swing.text.BadLocationException;
import javax.swing.text.BoxView;
import javax.swing.text.Caret;
import javax.swing.text.ChangedCharSetException;
import javax.swing.text.CompositeView;
import javax.swing.text.DefaultEditorKit;
import javax.swing.text.Document;
import javax.swing.text.EditorKit;
import javax.swing.text.Element;
import javax.swing.text.ElementIterator;
import javax.swing.text.GlyphView;
import javax.swing.text.JTextComponent;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledEditorKit;
import javax.swing.text.View;
import javax.swing.text.ViewFactory;
import javax.swing.text.WrappedPlainView;
import javax.swing.text.html.HTML;
import javax.swing.text.html.HTMLDocument;
import javax.swing.text.html.HTMLEditorKit;
import sun.reflect.misc.ReflectUtil;

@AnnotatedFor({ "interning" })
@JavaBean(defaultProperty = "UIClassID", description = "A text component to edit various types of content.")
@SwingContainer(false)
@SuppressWarnings("serial")
public class JEditorPane extends JTextComponent {

    public JEditorPane() {
    }

    public JEditorPane(URL initialPage) throws IOException {
    }

    public JEditorPane(String url) throws IOException {
    }

    public JEditorPane(String type, String text) {
    }

    public synchronized void addHyperlinkListener(HyperlinkListener listener);

    public synchronized void removeHyperlinkListener(HyperlinkListener listener);

    @BeanProperty(bound = false)
    public synchronized HyperlinkListener[] getHyperlinkListeners();

    public void fireHyperlinkUpdate(HyperlinkEvent e);

    @BeanProperty(expert = true, description = "the URL used to set content")
    public void setPage(URL page) throws IOException;

    public void read(InputStream in, Object desc) throws IOException;

    void read(InputStream in, Document doc) throws IOException;

    class PageLoader extends SwingWorker<URL, Object> {

        protected URL doInBackground();
    }

    protected InputStream getStream(URL page) throws IOException;

    @SuppressWarnings("deprecation")
    public void scrollToReference(String reference);

    public URL getPage();

    public void setPage(String url) throws IOException;

    @BeanProperty(bound = false)
    public String getUIClassID();

    protected EditorKit createDefaultEditorKit();

    public EditorKit getEditorKit();

    public final String getContentType();

    @BeanProperty(bound = false, description = "the type of content")
    public final void setContentType(String type);

    @BeanProperty(expert = true, description = "the currently installed kit for handling content")
    public void setEditorKit(EditorKit kit);

    public EditorKit getEditorKitForContentType(String type);

    public void setEditorKitForContentType(String type, EditorKit k);

    @Override
    public void replaceSelection(String content);

    @SuppressWarnings("deprecation")
    public static EditorKit createEditorKitForContentType(String type);

    public static void registerEditorKitForContentType(String type, String classname);

    public static void registerEditorKitForContentType(String type, String classname, ClassLoader loader);

    public static String getEditorKitClassNameForContentType(String type);

    public Dimension getPreferredSize();

    @BeanProperty(bound = false, description = "the text of this component")
    public void setText(String t);

    public String getText();

    @BeanProperty(bound = false)
    public boolean getScrollableTracksViewportWidth();

    @BeanProperty(bound = false)
    public boolean getScrollableTracksViewportHeight();

    @Interned
    public static final String W3C_LENGTH_UNITS;

    @Interned
    public static final String HONOR_DISPLAY_PROPERTIES;

    protected String paramString();

    @BeanProperty(bound = false)
    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJEditorPane extends AccessibleJTextComponent {

        protected AccessibleJEditorPane() {
        }

        public String getAccessibleDescription();

        public AccessibleStateSet getAccessibleStateSet();
    }

    @SuppressWarnings("serial")
    protected class AccessibleJEditorPaneHTML extends AccessibleJEditorPane {

        public AccessibleText getAccessibleText();

        protected AccessibleJEditorPaneHTML() {
        }

        public int getAccessibleChildrenCount();

        public Accessible getAccessibleChild(int i);

        public Accessible getAccessibleAt(Point p);
    }

    protected class JEditorPaneAccessibleHypertextSupport extends AccessibleJEditorPane implements AccessibleHypertext {

        public class HTMLLink extends AccessibleHyperlink {

            public HTMLLink(Element e) {
            }

            public boolean isValid();

            public int getAccessibleActionCount();

            public boolean doAccessibleAction(int i);

            public String getAccessibleActionDescription(int i);

            public Object getAccessibleActionObject(int i);

            public Object getAccessibleActionAnchor(int i);

            public int getStartIndex();

            public int getEndIndex();
        }

        private class LinkVector extends Vector<HTMLLink> {

            public int baseElementIndex(Element e);
        }

        public JEditorPaneAccessibleHypertextSupport() {
        }

        public int getLinkCount();

        public int getLinkIndex(int charIndex);

        public AccessibleHyperlink getLink(int linkIndex);

        public String getLinkText(int linkIndex);
    }

    static class PlainEditorKit extends DefaultEditorKit implements ViewFactory {

        public ViewFactory getViewFactory();

        public View create(Element elem);

        View createI18N(Element elem);

        static class PlainParagraph extends javax.swing.text.ParagraphView {

            protected void setPropertiesFromAttributes();

            public int getFlowSpan(int index);

            protected SizeRequirements calculateMinorAxisRequirements(int axis, SizeRequirements r);

            static class LogicalView extends CompositeView {

                protected int getViewIndexAtPosition(int pos);

                protected boolean updateChildren(DocumentEvent.ElementChange ec, DocumentEvent e, ViewFactory f);

                protected void loadChildren(ViewFactory f);

                public float getPreferredSpan(int axis);

                protected void forwardUpdateToView(View v, DocumentEvent e, Shape a, ViewFactory f);

                public void paint(Graphics g, Shape allocation);

                protected boolean isBefore(int x, int y, Rectangle alloc);

                protected boolean isAfter(int x, int y, Rectangle alloc);

                protected View getViewAtPoint(int x, int y, Rectangle alloc);

                protected void childAllocation(int index, Rectangle a);
            }
        }
    }

    static class HeaderParser {

        public HeaderParser(String raw) {
        }

        public String findKey(int i);

        public String findValue(int i);

        public String findValue(String key);

        public String findValue(String k, String Default);

        public int findInt(String k, int Default);
    }
}
