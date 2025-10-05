/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import java.awt.font.TextAttribute;
    @Positive
import java.util.*;
    @Positive
import java.net.URL;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.io.*;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.text.*;
    @Positive
import javax.swing.undo.*;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import static sun.swing.SwingUtilities2.IMPLIED_CR;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class HTMLDocument extends DefaultStyledDocument {

    @Positive
    public HTMLDocument() {
    @Positive
    }

    @Positive
    public HTMLDocument(StyleSheet styles) {
    @Positive
    }

    @Positive
    public HTMLDocument(Content c, StyleSheet styles) {
    @Positive
    }

    @Positive
    public HTMLEditorKit.ParserCallback getReader(int pos);

    @Positive
    public HTMLEditorKit.ParserCallback getReader(int pos, int popDepth, int pushDepth, HTML.Tag insertTag);

    @Positive
    HTMLEditorKit.ParserCallback getReader(int pos, int popDepth, int pushDepth, HTML.Tag insertTag, boolean insertInsertTag);

    @Positive
    public URL getBase();

    @Positive
    public void setBase(URL u);

    @Positive
    protected void insert(int offset, ElementSpec[] data) throws BadLocationException;

    @Positive
    protected void insertUpdate(DefaultDocumentEvent chng, AttributeSet attr);

    @Positive
    protected void create(ElementSpec[] data);

    @Positive
    public void setParagraphAttributes(int offset, int length, AttributeSet s, boolean replace);

    @Positive
    public StyleSheet getStyleSheet();

    @Positive
    public Iterator getIterator(HTML.Tag t);

    @Positive
    protected Element createLeafElement(Element parent, AttributeSet a, int p0, int p1);

    @Positive
    protected Element createBranchElement(Element parent, AttributeSet a);

    @Positive
    protected AbstractElement createDefaultRoot();

    @Positive
    public void setTokenThreshold(int n);

    @Positive
    public int getTokenThreshold();

    @Positive
    public void setPreservesUnknownTags(boolean preservesTags);

    @Positive
    public boolean getPreservesUnknownTags();

    @Positive
    public void processHTMLFrameHyperlinkEvent(HTMLFrameHyperlinkEvent e);

    @Positive
    static boolean matchNameAttribute(AttributeSet attr, HTML.Tag tag);

    @Positive
    boolean isFrameDocument();

    @Positive
    void setFrameDocumentState(boolean frameDoc);

    @Positive
    void addMap(Map map);

    @Positive
    void removeMap(Map map);

    @Positive
    Map getMap(String name);

    @Positive
    Enumeration<Object> getMaps();

    @Positive
    void setDefaultStyleSheetType(String contentType);

    @Positive
    String getDefaultStyleSheetType();

    @Positive
    public void setParser(HTMLEditorKit.Parser parser);

    @Positive
    public HTMLEditorKit.Parser getParser();

    @Positive
    public void setInnerHTML(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public void setOuterHTML(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public void insertAfterStart(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public void insertBeforeEnd(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public void insertBeforeStart(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public void insertAfterEnd(Element elem, String htmlText) throws BadLocationException, IOException;

    @Positive
    public Element getElement(String id);

    @Positive
    public Element getElement(Element e, Object attribute, Object value);

    @Positive
    void obtainLock();

    @Positive
    void releaseLock();

    @Positive
    protected void fireChangedUpdate(DocumentEvent e);

    @Positive
    protected void fireUndoableEditUpdate(UndoableEditEvent e);

    @Positive
    boolean hasBaseTag();

    @Positive
    String getBaseTarget();

    @Positive
    @Interned
    @Positive
    public static final String AdditionalComments;

    @Positive
    public abstract static class Iterator {

    @Positive
        protected Iterator() {
    @Positive
        }

    @Positive
        public abstract AttributeSet getAttributes();

    @Positive
        public abstract int getStartOffset();

    @Positive
        public abstract int getEndOffset();

    @Positive
        public abstract void next();

    @Positive
        public abstract boolean isValid();

    @Positive
        public abstract HTML.Tag getTag();
    @Positive
    }

    @Positive
    static class LeafIterator extends Iterator {

    @Positive
        public AttributeSet getAttributes();

    @Positive
        public int getStartOffset();

    @Positive
        public int getEndOffset();

    @Positive
        public void next();

    @Positive
        public HTML.Tag getTag();

    @Positive
        public boolean isValid();

    @Positive
        void nextLeaf(ElementIterator iter);

    @Positive
        void setEndOffset();
    @Positive
    }

    @Positive
    public class HTMLReader extends HTMLEditorKit.ParserCallback {

    @Positive
        public HTMLReader(int offset) {
    @Positive
        }

    @Positive
        public HTMLReader(int offset, int popDepth, int pushDepth, HTML.Tag insertTag) {
    @Positive
        }

    @Positive
        public void flush() throws BadLocationException;

    @Positive
        public void handleText(char[] data, int pos);

    @Positive
        public void handleStartTag(HTML.Tag t, MutableAttributeSet a, int pos);

    @Positive
        public void handleComment(char[] data, int pos);

    @Positive
        public void handleEndTag(HTML.Tag t, int pos);

    @Positive
        public void handleSimpleTag(HTML.Tag t, MutableAttributeSet a, int pos);

    @Positive
        public void handleEndOfLineString(String eol);

    @Positive
        protected void registerTag(HTML.Tag t, TagAction a);

    @Positive
        public class TagAction {

    @Positive
            public TagAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        public class BlockAction extends TagAction {

    @Positive
            public BlockAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        private class FormTagAction extends BlockAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        public class ParagraphAction extends BlockAction {

    @Positive
            public ParagraphAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        public class SpecialAction extends TagAction {

    @Positive
            public SpecialAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);
    @Positive
        }

    @Positive
        public class IsindexAction extends TagAction {

    @Positive
            public IsindexAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);
    @Positive
        }

    @Positive
        public class HiddenAction extends TagAction {

    @Positive
            public HiddenAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            boolean isEmpty(HTML.Tag t);
    @Positive
        }

    @Positive
        class MetaAction extends HiddenAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            boolean isEmpty(HTML.Tag t);
    @Positive
        }

    @Positive
        class HeadAction extends BlockAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            boolean isEmpty(HTML.Tag t);
    @Positive
        }

    @Positive
        class LinkAction extends HiddenAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);
    @Positive
        }

    @Positive
        class MapAction extends TagAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        class AreaAction extends TagAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        class StyleAction extends TagAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            boolean isEmpty(HTML.Tag t);
    @Positive
        }

    @Positive
        public class PreAction extends BlockAction {

    @Positive
            public PreAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        public class CharacterAction extends TagAction {

    @Positive
            public CharacterAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        class ConvertAction extends TagAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        class AnchorAction extends CharacterAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);
    @Positive
        }

    @Positive
        class TitleAction extends HiddenAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            boolean isEmpty(HTML.Tag t);
    @Positive
        }

    @Positive
        class BaseAction extends TagAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);
    @Positive
        }

    @Positive
        class ObjectAction extends SpecialAction {

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet a);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            void addParameter(AttributeSet a);
    @Positive
        }

    @Positive
        public class FormAction extends SpecialAction {

    @Positive
            public FormAction() {
    @Positive
            }

    @Positive
            public void start(HTML.Tag t, MutableAttributeSet attr);

    @Positive
            public void end(HTML.Tag t);

    @Positive
            void setModel(String type, MutableAttributeSet attr);
    @Positive
        }

    @Positive
        protected void pushCharacterStyle();

    @Positive
        protected void popCharacterStyle();

    @Positive
        protected void textAreaContent(char[] data);

    @Positive
        protected void preContent(char[] data);

    @Positive
        protected void blockOpen(HTML.Tag t, MutableAttributeSet attr);

    @Positive
        protected void blockClose(HTML.Tag t);

    @Positive
        protected void addContent(char[] data, int offs, int length);

    @Positive
        protected void addContent(char[] data, int offs, int length, boolean generateImpliedPIfNecessary);

    @Positive
        protected void addSpecialElement(HTML.Tag t, MutableAttributeSet a);

    @Positive
        void flushBuffer(boolean endOfStream) throws BadLocationException;

    @Positive
        void addCSSRules(String rules);

    @Positive
        void linkCSSStyleSheet(String href);

    @Positive
        protected Vector<ElementSpec> parseBuffer;

    @Positive
        protected MutableAttributeSet charAttr;
    @Positive
    }

    @Positive
    static class TaggedAttributeSet extends SimpleAttributeSet {
    @Positive
    }

    @Positive
    public class RunElement extends LeafElement {

    @Positive
        public RunElement(Element parent, AttributeSet a, int offs0, int offs1) {
    @Positive
        }

    @Positive
        public String getName();

    @Positive
        public AttributeSet getResolveParent();
    @Positive
    }

    @Positive
    public class BlockElement extends BranchElement {

    @Positive
        public BlockElement(Element parent, AttributeSet a) {
    @Positive
        }

    @Positive
        public String getName();

    @Positive
        public AttributeSet getResolveParent();
    @Positive
    }

    @Positive
    private static class FixedLengthDocument extends PlainDocument {

    @Positive
        public FixedLengthDocument(int maxLength) {
    @Positive
        }

    @Positive
        public void insertString(int offset, String str, AttributeSet a) throws BadLocationException;
    @Positive
    }
    @Positive
}
