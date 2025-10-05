/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.font.TextAttribute;
import java.util.*;
import java.net.URL;
import java.net.MalformedURLException;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.text.*;
import javax.swing.undo.*;
import sun.swing.SwingUtilities2;
import static sun.swing.SwingUtilities2.IMPLIED_CR;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class HTMLDocument extends DefaultStyledDocument {

    public HTMLDocument() {
    }

    public HTMLDocument(StyleSheet styles) {
    }

    public HTMLDocument(Content c, StyleSheet styles) {
    }

    public HTMLEditorKit.ParserCallback getReader(int pos);

    public HTMLEditorKit.ParserCallback getReader(int pos, int popDepth, int pushDepth, HTML.Tag insertTag);

    HTMLEditorKit.ParserCallback getReader(int pos, int popDepth, int pushDepth, HTML.Tag insertTag, boolean insertInsertTag);

    public URL getBase();

    public void setBase(URL u);

    protected void insert(int offset, ElementSpec[] data) throws BadLocationException;

    protected void insertUpdate(DefaultDocumentEvent chng, AttributeSet attr);

    protected void create(ElementSpec[] data);

    public void setParagraphAttributes(int offset, int length, AttributeSet s, boolean replace);

    public StyleSheet getStyleSheet();

    public Iterator getIterator(HTML.Tag t);

    protected Element createLeafElement(Element parent, AttributeSet a, int p0, int p1);

    protected Element createBranchElement(Element parent, AttributeSet a);

    protected AbstractElement createDefaultRoot();

    public void setTokenThreshold(int n);

    public int getTokenThreshold();

    public void setPreservesUnknownTags(boolean preservesTags);

    public boolean getPreservesUnknownTags();

    public void processHTMLFrameHyperlinkEvent(HTMLFrameHyperlinkEvent e);

    static boolean matchNameAttribute(AttributeSet attr, HTML.Tag tag);

    boolean isFrameDocument();

    void setFrameDocumentState(boolean frameDoc);

    void addMap(Map map);

    void removeMap(Map map);

    Map getMap(String name);

    Enumeration<Object> getMaps();

    void setDefaultStyleSheetType(String contentType);

    String getDefaultStyleSheetType();

    public void setParser(HTMLEditorKit.Parser parser);

    public HTMLEditorKit.Parser getParser();

    public void setInnerHTML(Element elem, String htmlText) throws BadLocationException, IOException;

    public void setOuterHTML(Element elem, String htmlText) throws BadLocationException, IOException;

    public void insertAfterStart(Element elem, String htmlText) throws BadLocationException, IOException;

    public void insertBeforeEnd(Element elem, String htmlText) throws BadLocationException, IOException;

    public void insertBeforeStart(Element elem, String htmlText) throws BadLocationException, IOException;

    public void insertAfterEnd(Element elem, String htmlText) throws BadLocationException, IOException;

    public Element getElement(String id);

    public Element getElement(Element e, Object attribute, Object value);

    void obtainLock();

    void releaseLock();

    protected void fireChangedUpdate(DocumentEvent e);

    protected void fireUndoableEditUpdate(UndoableEditEvent e);

    boolean hasBaseTag();

    String getBaseTarget();

    @Interned
    public static final String AdditionalComments;

    public abstract static class Iterator {

        protected Iterator() {
        }

        public abstract AttributeSet getAttributes();

        public abstract int getStartOffset();

        public abstract int getEndOffset();

        public abstract void next();

        public abstract boolean isValid();

        public abstract HTML.Tag getTag();
    }

    static class LeafIterator extends Iterator {

        public AttributeSet getAttributes();

        public int getStartOffset();

        public int getEndOffset();

        public void next();

        public HTML.Tag getTag();

        public boolean isValid();

        void nextLeaf(ElementIterator iter);

        void setEndOffset();
    }

    public class HTMLReader extends HTMLEditorKit.ParserCallback {

        public HTMLReader(int offset) {
        }

        public HTMLReader(int offset, int popDepth, int pushDepth, HTML.Tag insertTag) {
        }

        public void flush() throws BadLocationException;

        public void handleText(char[] data, int pos);

        public void handleStartTag(HTML.Tag t, MutableAttributeSet a, int pos);

        public void handleComment(char[] data, int pos);

        public void handleEndTag(HTML.Tag t, int pos);

        public void handleSimpleTag(HTML.Tag t, MutableAttributeSet a, int pos);

        public void handleEndOfLineString(String eol);

        protected void registerTag(HTML.Tag t, TagAction a);

        public class TagAction {

            public TagAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);
        }

        public class BlockAction extends TagAction {

            public BlockAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        private class FormTagAction extends BlockAction {

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        public class ParagraphAction extends BlockAction {

            public ParagraphAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);
        }

        public class SpecialAction extends TagAction {

            public SpecialAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet a);
        }

        public class IsindexAction extends TagAction {

            public IsindexAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet a);
        }

        public class HiddenAction extends TagAction {

            public HiddenAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);

            boolean isEmpty(HTML.Tag t);
        }

        class MetaAction extends HiddenAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            boolean isEmpty(HTML.Tag t);
        }

        class HeadAction extends BlockAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);

            boolean isEmpty(HTML.Tag t);
        }

        class LinkAction extends HiddenAction {

            public void start(HTML.Tag t, MutableAttributeSet a);
        }

        class MapAction extends TagAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);
        }

        class AreaAction extends TagAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);
        }

        class StyleAction extends TagAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);

            boolean isEmpty(HTML.Tag t);
        }

        public class PreAction extends BlockAction {

            public PreAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        public class CharacterAction extends TagAction {

            public CharacterAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        class ConvertAction extends TagAction {

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        class AnchorAction extends CharacterAction {

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);
        }

        class TitleAction extends HiddenAction {

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);

            boolean isEmpty(HTML.Tag t);
        }

        class BaseAction extends TagAction {

            public void start(HTML.Tag t, MutableAttributeSet attr);
        }

        class ObjectAction extends SpecialAction {

            public void start(HTML.Tag t, MutableAttributeSet a);

            public void end(HTML.Tag t);

            void addParameter(AttributeSet a);
        }

        public class FormAction extends SpecialAction {

            public FormAction() {
            }

            public void start(HTML.Tag t, MutableAttributeSet attr);

            public void end(HTML.Tag t);

            void setModel(String type, MutableAttributeSet attr);
        }

        protected void pushCharacterStyle();

        protected void popCharacterStyle();

        protected void textAreaContent(char[] data);

        protected void preContent(char[] data);

        protected void blockOpen(HTML.Tag t, MutableAttributeSet attr);

        protected void blockClose(HTML.Tag t);

        protected void addContent(char[] data, int offs, int length);

        protected void addContent(char[] data, int offs, int length, boolean generateImpliedPIfNecessary);

        protected void addSpecialElement(HTML.Tag t, MutableAttributeSet a);

        void flushBuffer(boolean endOfStream) throws BadLocationException;

        void addCSSRules(String rules);

        void linkCSSStyleSheet(String href);

        protected Vector<ElementSpec> parseBuffer;

        protected MutableAttributeSet charAttr;
    }

    static class TaggedAttributeSet extends SimpleAttributeSet {
    }

    public class RunElement extends LeafElement {

        public RunElement(Element parent, AttributeSet a, int offs0, int offs1) {
        }

        public String getName();

        public AttributeSet getResolveParent();
    }

    public class BlockElement extends BranchElement {

        public BlockElement(Element parent, AttributeSet a) {
        }

        public String getName();

        public AttributeSet getResolveParent();
    }

    private static class FixedLengthDocument extends PlainDocument {

        public FixedLengthDocument(int maxLength) {
        }

        public void insertString(int offset, String str, AttributeSet a) throws BadLocationException;
    }
}
