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
package javax.swing.text;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.font.TextAttribute;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectInputValidation;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.text.Bidi;
    @Positive
import java.util.Dictionary;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Vector;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.event.DocumentEvent;
    @Positive
import javax.swing.event.DocumentListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.event.UndoableEditEvent;
    @Positive
import javax.swing.event.UndoableEditListener;
    @Positive
import javax.swing.tree.TreeNode;
    @Positive
import javax.swing.undo.AbstractUndoableEdit;
    @Positive
import javax.swing.undo.CannotRedoException;
    @Positive
import javax.swing.undo.CannotUndoException;
    @Positive
import javax.swing.undo.CompoundEdit;
    @Positive
import javax.swing.undo.UndoableEdit;
    @Positive
import sun.font.BidiUtils;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import sun.swing.text.UndoableEditLockSupport;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class AbstractDocument implements Document, Serializable {

    @Positive
    protected AbstractDocument(Content data) {
    @Positive
    }

    @Positive
    protected AbstractDocument(Content data, AttributeContext context) {
    @Positive
    }

    @Positive
    public Dictionary<Object, Object> getDocumentProperties();

    @Positive
    public void setDocumentProperties(Dictionary<Object, Object> x);

    @Positive
    protected void fireInsertUpdate(DocumentEvent e);

    @Positive
    protected void fireChangedUpdate(DocumentEvent e);

    @Positive
    protected void fireRemoveUpdate(DocumentEvent e);

    @Positive
    protected void fireUndoableEditUpdate(UndoableEditEvent e);

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    public int getAsynchronousLoadPriority();

    @Positive
    public void setAsynchronousLoadPriority(int p);

    @Positive
    public void setDocumentFilter(DocumentFilter filter);

    @Positive
    public DocumentFilter getDocumentFilter();

    @Positive
    public void render(Runnable r);

    @Positive
    public int getLength();

    @Positive
    public void addDocumentListener(DocumentListener listener);

    @Positive
    public void removeDocumentListener(DocumentListener listener);

    @Positive
    public DocumentListener[] getDocumentListeners();

    @Positive
    public void addUndoableEditListener(UndoableEditListener listener);

    @Positive
    public void removeUndoableEditListener(UndoableEditListener listener);

    @Positive
    public UndoableEditListener[] getUndoableEditListeners();

    @Positive
    public final Object getProperty(Object key);

    @Positive
    public final void putProperty(Object key, Object value);

    @Positive
    public void remove(int offs, int len) throws BadLocationException;

    @Positive
    void handleRemove(int offs, int len) throws BadLocationException;

    @Positive
    public void replace(int offset, int length, String text, AttributeSet attrs) throws BadLocationException;

    @Positive
    public void insertString(int offs, String str, AttributeSet a) throws BadLocationException;

    @Positive
    public String getText(int offset, int length) throws BadLocationException;

    @Positive
    public void getText(int offset, int length, Segment txt) throws BadLocationException;

    @Positive
    public synchronized Position createPosition(int offs) throws BadLocationException;

    @Positive
    public final Position getStartPosition();

    @Positive
    public final Position getEndPosition();

    @Positive
    public Element[] getRootElements();

    @Positive
    public abstract Element getDefaultRootElement();

    @Positive
    public Element getBidiRootElement();

    @Positive
    static boolean isLeftToRight(Document doc, int p0, int p1);

    @Positive
    public abstract Element getParagraphElement(int pos);

    @Positive
    protected final AttributeContext getAttributeContext();

    @Positive
    protected void insertUpdate(DefaultDocumentEvent chng, AttributeSet attr);

    @Positive
    protected void removeUpdate(DefaultDocumentEvent chng);

    @Positive
    protected void postRemoveUpdate(DefaultDocumentEvent chng);

    @Positive
    void updateBidi(DefaultDocumentEvent chng);

    @Positive
    public void dump(PrintStream out);

    @Positive
    protected final Content getContent();

    @Positive
    protected Element createLeafElement(Element parent, AttributeSet a, int p0, int p1);

    @Positive
    protected Element createBranchElement(Element parent, AttributeSet a);

    @Positive
    protected final synchronized Thread getCurrentWriter();

    @Positive
    protected final synchronized void writeLock();

    @Positive
    protected final synchronized void writeUnlock();

    @Positive
    public final synchronized void readLock();

    @Positive
    public final synchronized void readUnlock();

    @Positive
    protected EventListenerList listenerList;

    @Positive
    protected static final String BAD_LOCATION;

    @Positive
    @Interned
    @Positive
    public static final String ParagraphElementName;

    @Positive
    @Interned
    @Positive
    public static final String ContentElementName;

    @Positive
    @Interned
    @Positive
    public static final String SectionElementName;

    @Positive
    @Interned
    @Positive
    public static final String BidiElementName;

    @Positive
    @Interned
    @Positive
    public static final String ElementNameAttribute;

    @Positive
    public interface Content {

    @Positive
        public Position createPosition(int offset) throws BadLocationException;

    @Positive
        public int length();

    @Positive
        public UndoableEdit insertString(int where, String str) throws BadLocationException;

    @Positive
        public UndoableEdit remove(int where, int nitems) throws BadLocationException;

    @Positive
        public String getString(int where, int len) throws BadLocationException;

    @Positive
        public void getChars(int where, int len, Segment txt) throws BadLocationException;
    @Positive
    }

    @Positive
    public interface AttributeContext {

    @Positive
        public AttributeSet addAttribute(AttributeSet old, Object name, Object value);

    @Positive
        public AttributeSet addAttributes(AttributeSet old, AttributeSet attr);

    @Positive
        public AttributeSet removeAttribute(AttributeSet old, Object name);

    @Positive
        public AttributeSet removeAttributes(AttributeSet old, Enumeration<?> names);

    @Positive
        public AttributeSet removeAttributes(AttributeSet old, AttributeSet attrs);

    @Positive
        public AttributeSet getEmptySet();

    @Positive
        public void reclaim(AttributeSet a);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public abstract class AbstractElement implements Element, MutableAttributeSet, Serializable, TreeNode {

    @Positive
        public AbstractElement(Element parent, AttributeSet a) {
    @Positive
        }

    @Positive
        public void dump(PrintStream psOut, int indentAmount);

    @Positive
        public int getAttributeCount();

    @Positive
        public boolean isDefined(Object attrName);

    @Positive
        public boolean isEqual(AttributeSet attr);

    @Positive
        public AttributeSet copyAttributes();

    @Positive
        public Object getAttribute(Object attrName);

    @Positive
        public Enumeration<?> getAttributeNames();

    @Positive
        public boolean containsAttribute(Object name, Object value);

    @Positive
        public boolean containsAttributes(AttributeSet attrs);

    @Positive
        public AttributeSet getResolveParent();

    @Positive
        public void addAttribute(Object name, Object value);

    @Positive
        public void addAttributes(AttributeSet attr);

    @Positive
        public void removeAttribute(Object name);

    @Positive
        public void removeAttributes(Enumeration<?> names);

    @Positive
        public void removeAttributes(AttributeSet attrs);

    @Positive
        public void setResolveParent(AttributeSet parent);

    @Positive
        public Document getDocument();

    @Positive
        public Element getParentElement();

    @Positive
        public AttributeSet getAttributes();

    @Positive
        public String getName();

    @Positive
        public abstract int getStartOffset();

    @Positive
        public abstract int getEndOffset();

    @Positive
        public abstract Element getElement(int index);

    @Positive
        public abstract int getElementCount();

    @Positive
        public abstract int getElementIndex(int offset);

    @Positive
        public abstract boolean isLeaf();

    @Positive
        public TreeNode getChildAt(int childIndex);

    @Positive
        public int getChildCount();

    @Positive
        public TreeNode getParent();

    @Positive
        public int getIndex(TreeNode node);

    @Positive
        public abstract boolean getAllowsChildren();

    @Positive
        public abstract Enumeration<TreeNode> children();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public class BranchElement extends AbstractElement {

    @Positive
        public BranchElement(Element parent, AttributeSet a) {
    @Positive
        }

    @Positive
        public Element positionToElement(int pos);

    @Positive
        public void replace(int offset, int length, Element[] elems);

    @Positive
        public String toString();

    @Positive
        public String getName();

    @Positive
        public int getStartOffset();

    @Positive
        public int getEndOffset();

    @Positive
        public Element getElement(int index);

    @Positive
        public int getElementCount();

    @Positive
        public int getElementIndex(int offset);

    @Positive
        public boolean isLeaf();

    @Positive
        public boolean getAllowsChildren();

    @Positive
        public Enumeration<TreeNode> children();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public class LeafElement extends AbstractElement {

    @Positive
        public LeafElement(Element parent, AttributeSet a, int offs0, int offs1) {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        public int getStartOffset();

    @Positive
        public int getEndOffset();

    @Positive
        public String getName();

    @Positive
        public int getElementIndex(int pos);

    @Positive
        public Element getElement(int index);

    @Positive
        public int getElementCount();

    @Positive
        public boolean isLeaf();

    @Positive
        public boolean getAllowsChildren();

    @Positive
        @Override
    @Positive
        public Enumeration<TreeNode> children();
    @Positive
    }

    @Positive
    class BidiRootElement extends BranchElement {

    @Positive
        public String getName();
    @Positive
    }

    @Positive
    class BidiElement extends LeafElement {

    @Positive
        public String getName();

    @Positive
        int getLevel();

    @Positive
        boolean isLeftToRight();
    @Positive
    }

    @Positive
    public class DefaultDocumentEvent extends CompoundEdit implements DocumentEvent {

    @Positive
        public DefaultDocumentEvent(int offs, int len, DocumentEvent.EventType type) {
    @Positive
        }

    @Positive
        public String toString();

    @Positive
        public boolean addEdit(UndoableEdit anEdit);

    @Positive
        public void redo() throws CannotRedoException;

    @Positive
        public void undo() throws CannotUndoException;

    @Positive
        public boolean isSignificant();

    @Positive
        public String getPresentationName();

    @Positive
        public String getUndoPresentationName();

    @Positive
        public String getRedoPresentationName();

    @Positive
        public DocumentEvent.EventType getType();

    @Positive
        public int getOffset();

    @Positive
        public int getLength();

    @Positive
        public Document getDocument();

    @Positive
        public DocumentEvent.ElementChange getChange(Element elem);
    @Positive
    }

    @Positive
    class DefaultDocumentEventUndoableWrapper extends DefaultDocumentEvent implements UndoableEdit, UndoableEditLockSupport {

    @Positive
        public DefaultDocumentEventUndoableWrapper(DefaultDocumentEvent dde) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void undo() throws CannotUndoException;

    @Positive
        @Override
    @Positive
        public boolean canUndo();

    @Positive
        @Override
    @Positive
        public void redo() throws CannotRedoException;

    @Positive
        @Override
    @Positive
        public boolean canRedo();

    @Positive
        @Override
    @Positive
        public void die();

    @Positive
        @Override
    @Positive
        public boolean addEdit(UndoableEdit anEdit);

    @Positive
        @Override
    @Positive
        public boolean replaceEdit(UndoableEdit anEdit);

    @Positive
        @Override
    @Positive
        public boolean isSignificant();

    @Positive
        @Override
    @Positive
        public String getPresentationName();

    @Positive
        @Override
    @Positive
        public String getUndoPresentationName();

    @Positive
        @Override
    @Positive
        public String getRedoPresentationName();

    @Positive
        @Override
    @Positive
        public void lockEdit();

    @Positive
        @Override
    @Positive
        public void unlockEdit();
    @Positive
    }

    @Positive
    class UndoRedoDocumentEvent implements DocumentEvent {

    @Positive
        public UndoRedoDocumentEvent(DefaultDocumentEvent src, boolean isUndo) {
    @Positive
        }

    @Positive
        public DefaultDocumentEvent getSource();

    @Positive
        public int getOffset();

    @Positive
        public int getLength();

    @Positive
        public Document getDocument();

    @Positive
        public DocumentEvent.EventType getType();

    @Positive
        public DocumentEvent.ElementChange getChange(Element elem);
    @Positive
    }

    @Positive
    public static class ElementEdit extends AbstractUndoableEdit implements DocumentEvent.ElementChange {

    @Positive
        public ElementEdit(Element e, int index, Element[] removed, Element[] added) {
    @Positive
        }

    @Positive
        public Element getElement();

    @Positive
        public int getIndex();

    @Positive
        public Element[] getChildrenRemoved();

    @Positive
        public Element[] getChildrenAdded();

    @Positive
        public void redo() throws CannotRedoException;

    @Positive
        public void undo() throws CannotUndoException;
    @Positive
    }

    @Positive
    private class DefaultFilterBypass extends DocumentFilter.FilterBypass {

    @Positive
        public Document getDocument();

    @Positive
        public void remove(int offset, int length) throws BadLocationException;

    @Positive
        public void insertString(int offset, String string, AttributeSet attr) throws BadLocationException;

    @Positive
        public void replace(int offset, int length, String text, AttributeSet attrs) throws BadLocationException;
    @Positive
    }
    @Positive
}
