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
package javax.swing;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.IllegalComponentStateException;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Shape;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.DataOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Serial;
    @Positive
import java.io.StringReader;
    @Positive
import java.io.StringWriter;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.net.HttpURLConnection;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLConnection;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Map;
    @Positive
import java.util.Vector;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleComponent;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleHyperlink;
    @Positive
import javax.accessibility.AccessibleHypertext;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import javax.accessibility.AccessibleText;
    @Positive
import javax.swing.event.DocumentEvent;
    @Positive
import javax.swing.event.DocumentListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.event.HyperlinkEvent;
    @Positive
import javax.swing.event.HyperlinkListener;
    @Positive
import javax.swing.plaf.TextUI;
    @Positive
import javax.swing.text.AbstractDocument;
    @Positive
import javax.swing.text.AttributeSet;
    @Positive
import javax.swing.text.BadLocationException;
    @Positive
import javax.swing.text.BoxView;
    @Positive
import javax.swing.text.Caret;
    @Positive
import javax.swing.text.ChangedCharSetException;
    @Positive
import javax.swing.text.CompositeView;
    @Positive
import javax.swing.text.DefaultEditorKit;
    @Positive
import javax.swing.text.Document;
    @Positive
import javax.swing.text.EditorKit;
    @Positive
import javax.swing.text.Element;
    @Positive
import javax.swing.text.ElementIterator;
    @Positive
import javax.swing.text.GlyphView;
    @Positive
import javax.swing.text.JTextComponent;
    @Positive
import javax.swing.text.StyleConstants;
    @Positive
import javax.swing.text.StyledEditorKit;
    @Positive
import javax.swing.text.View;
    @Positive
import javax.swing.text.ViewFactory;
    @Positive
import javax.swing.text.WrappedPlainView;
    @Positive
import javax.swing.text.html.HTML;
    @Positive
import javax.swing.text.html.HTMLDocument;
    @Positive
import javax.swing.text.html.HTMLEditorKit;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(defaultProperty = "UIClassID", description = "A text component to edit various types of content.")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JEditorPane extends JTextComponent {

    @Positive
    public JEditorPane() {
    @Positive
    }

    @Positive
    public JEditorPane(URL initialPage) throws IOException {
    @Positive
    }

    @Positive
    public JEditorPane(String url) throws IOException {
    @Positive
    }

    @Positive
    public JEditorPane(String type, String text) {
    @Positive
    }

    @Positive
    public synchronized void addHyperlinkListener(HyperlinkListener listener);

    @Positive
    public synchronized void removeHyperlinkListener(HyperlinkListener listener);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public synchronized HyperlinkListener[] getHyperlinkListeners();

    @Positive
    public void fireHyperlinkUpdate(HyperlinkEvent e);

    @Positive
    @BeanProperty(expert = true, description = "the URL used to set content")
    @Positive
    public void setPage(URL page) throws IOException;

    @Positive
    public void read(InputStream in, Object desc) throws IOException;

    @Positive
    void read(InputStream in, Document doc) throws IOException;

    @Positive
    class PageLoader extends SwingWorker<URL, Object> {

    @Positive
        protected URL doInBackground();
    @Positive
    }

    @Positive
    protected InputStream getStream(URL page) throws IOException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void scrollToReference(String reference);

    @Positive
    public URL getPage();

    @Positive
    public void setPage(String url) throws IOException;

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public String getUIClassID();

    @Positive
    protected EditorKit createDefaultEditorKit();

    @Positive
    public EditorKit getEditorKit();

    @Positive
    public final String getContentType();

    @Positive
    @BeanProperty(bound = false, description = "the type of content")
    @Positive
    public final void setContentType(String type);

    @Positive
    @BeanProperty(expert = true, description = "the currently installed kit for handling content")
    @Positive
    public void setEditorKit(EditorKit kit);

    @Positive
    public EditorKit getEditorKitForContentType(String type);

    @Positive
    public void setEditorKitForContentType(String type, EditorKit k);

    @Positive
    @Override
    @Positive
    public void replaceSelection(String content);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static EditorKit createEditorKitForContentType(String type);

    @Positive
    public static void registerEditorKitForContentType(String type, String classname);

    @Positive
    public static void registerEditorKitForContentType(String type, String classname, ClassLoader loader);

    @Positive
    public static String getEditorKitClassNameForContentType(String type);

    @Positive
    public Dimension getPreferredSize();

    @Positive
    @BeanProperty(bound = false, description = "the text of this component")
    @Positive
    public void setText(String t);

    @Positive
    public String getText();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean getScrollableTracksViewportWidth();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean getScrollableTracksViewportHeight();

    @Positive
    @Interned
    @Positive
    public static final String W3C_LENGTH_UNITS;

    @Positive
    @Interned
    @Positive
    public static final String HONOR_DISPLAY_PROPERTIES;

    @Positive
    protected String paramString();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJEditorPane extends AccessibleJTextComponent {

    @Positive
        protected AccessibleJEditorPane() {
    @Positive
        }

    @Positive
        public String getAccessibleDescription();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJEditorPaneHTML extends AccessibleJEditorPane {

    @Positive
        public AccessibleText getAccessibleText();

    @Positive
        protected AccessibleJEditorPaneHTML() {
    @Positive
        }

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);

    @Positive
        public Accessible getAccessibleAt(Point p);
    @Positive
    }

    @Positive
    protected class JEditorPaneAccessibleHypertextSupport extends AccessibleJEditorPane implements AccessibleHypertext {

    @Positive
        public class HTMLLink extends AccessibleHyperlink {

    @Positive
            public HTMLLink(Element e) {
    @Positive
            }

    @Positive
            public boolean isValid();

    @Positive
            public int getAccessibleActionCount();

    @Positive
            public boolean doAccessibleAction(int i);

    @Positive
            public String getAccessibleActionDescription(int i);

    @Positive
            public Object getAccessibleActionObject(int i);

    @Positive
            public Object getAccessibleActionAnchor(int i);

    @Positive
            public int getStartIndex();

    @Positive
            public int getEndIndex();
    @Positive
        }

    @Positive
        private class LinkVector extends Vector<HTMLLink> {

    @Positive
            public int baseElementIndex(Element e);
    @Positive
        }

    @Positive
        public JEditorPaneAccessibleHypertextSupport() {
    @Positive
        }

    @Positive
        public int getLinkCount();

    @Positive
        public int getLinkIndex(int charIndex);

    @Positive
        public AccessibleHyperlink getLink(int linkIndex);

    @Positive
        public String getLinkText(int linkIndex);
    @Positive
    }

    @Positive
    static class PlainEditorKit extends DefaultEditorKit implements ViewFactory {

    @Positive
        public ViewFactory getViewFactory();

    @Positive
        public View create(Element elem);

    @Positive
        View createI18N(Element elem);

    @Positive
        static class PlainParagraph extends javax.swing.text.ParagraphView {

    @Positive
            protected void setPropertiesFromAttributes();

    @Positive
            public int getFlowSpan(int index);

    @Positive
            protected SizeRequirements calculateMinorAxisRequirements(int axis, SizeRequirements r);

    @Positive
            static class LogicalView extends CompositeView {

    @Positive
                protected int getViewIndexAtPosition(int pos);

    @Positive
                protected boolean updateChildren(DocumentEvent.ElementChange ec, DocumentEvent e, ViewFactory f);

    @Positive
                protected void loadChildren(ViewFactory f);

    @Positive
                public float getPreferredSpan(int axis);

    @Positive
                protected void forwardUpdateToView(View v, DocumentEvent e, Shape a, ViewFactory f);

    @Positive
                public void paint(Graphics g, Shape allocation);

    @Positive
                protected boolean isBefore(int x, int y, Rectangle alloc);

    @Positive
                protected boolean isAfter(int x, int y, Rectangle alloc);

    @Positive
                protected View getViewAtPoint(int x, int y, Rectangle alloc);

    @Positive
                protected void childAllocation(int index, Rectangle a);
    @Positive
            }
    @Positive
        }
    @Positive
    }

    @Positive
    static class HeaderParser {

    @Positive
        public HeaderParser(String raw) {
    @Positive
        }

    @Positive
        public String findKey(int i);

    @Positive
        public String findValue(int i);

    @Positive
        public String findValue(String key);

    @Positive
        public String findValue(String k, String Default);

    @Positive
        public int findInt(String k, int Default);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
