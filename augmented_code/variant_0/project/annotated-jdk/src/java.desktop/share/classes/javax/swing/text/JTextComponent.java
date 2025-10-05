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
package javax.swing.text;

    @Positive
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import com.sun.beans.util.Cache;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.Transient;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Vector;
    @Positive
import java.util.concurrent.*;
    @Positive
import java.io.*;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.awt.print.*;
    @Positive
import java.awt.datatransfer.*;
    @Positive
import java.awt.im.InputContext;
    @Positive
import java.awt.im.InputMethodRequests;
    @Positive
import java.awt.font.TextHitInfo;
    @Positive
import java.awt.font.TextAttribute;
    @Positive
import java.awt.geom.Point2D;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import java.awt.print.Printable;
    @Positive
import java.awt.print.PrinterException;
    @Positive
import javax.print.PrintService;
    @Positive
import javax.print.attribute.PrintRequestAttributeSet;
    @Positive
import java.text.*;
    @Positive
import java.text.AttributedCharacterIterator.Attribute;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.plaf.*;
    @Positive
import javax.accessibility.*;
    @Positive
import javax.print.attribute.*;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.swing.PrintingStatus;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import sun.swing.text.TextComponentPrintable;
    @Positive
import sun.swing.SwingAccessor;

    @Positive
@AnnotatedFor({ "guieffect", "interning" })
    @Positive
@JavaBean(defaultProperty = "UI")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class JTextComponent extends JComponent implements Scrollable, Accessible {

    @Positive
    public JTextComponent() {
    @Positive
    }

    @Positive
    public TextUI getUI();

    @Positive
    public void setUI(TextUI ui);

    @Positive
    public void updateUI();

    @Positive
    public void addCaretListener(CaretListener listener);

    @Positive
    public void removeCaretListener(CaretListener listener);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public CaretListener[] getCaretListeners();

    @Positive
    protected void fireCaretUpdate(CaretEvent e);

    @Positive
    @BeanProperty(expert = true, description = "the text document model")
    @Positive
    public void setDocument(Document doc);

    @Positive
    public Document getDocument();

    @Positive
    public void setComponentOrientation(ComponentOrientation o);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Action[] getActions();

    @Positive
    @BeanProperty(description = "desired space between the border and text area")
    @Positive
    public void setMargin(Insets m);

    @Positive
    public Insets getMargin();

    @Positive
    public void setNavigationFilter(NavigationFilter filter);

    @Positive
    public NavigationFilter getNavigationFilter();

    @Positive
    @Transient
    @Positive
    public Caret getCaret();

    @Positive
    @BeanProperty(expert = true, description = "the caret used to select/navigate")
    @Positive
    public void setCaret(Caret c);

    @Positive
    public Highlighter getHighlighter();

    @Positive
    @BeanProperty(expert = true, description = "object responsible for background highlights")
    @Positive
    public void setHighlighter(Highlighter h);

    @Positive
    @BeanProperty(description = "set of key event to action bindings to use")
    @Positive
    public void setKeymap(Keymap map);

    @Positive
    @BeanProperty(bound = false, description = "determines whether automatic drag handling is enabled")
    @Positive
    public void setDragEnabled(boolean b);

    @Positive
    public boolean getDragEnabled();

    @Positive
    public final void setDropMode(DropMode dropMode);

    @Positive
    public final DropMode getDropMode();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    DropLocation dropLocationForPoint(Point p);

    @Positive
    Object setDropLocation(TransferHandler.DropLocation location, Object state, boolean forDrop);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public final DropLocation getDropLocation();

    @Positive
    void updateInputMap(Keymap oldKm, Keymap newKm);

    @Positive
    public Keymap getKeymap();

    @Positive
    public static Keymap addKeymap(String nm, Keymap parent);

    @Positive
    public static Keymap removeKeymap(String nm);

    @Positive
    public static Keymap getKeymap(String nm);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class KeyBinding {

    @Positive
        public KeyStroke key;

    @Positive
        public String actionName;

    @Positive
        public KeyBinding(KeyStroke key, String actionName) {
    @Positive
        }
    @Positive
    }

    @Positive
    public static void loadKeymap(Keymap map, KeyBinding[] bindings, Action[] actions);

    @Positive
    public Color getCaretColor();

    @Positive
    @BeanProperty(preferred = true, description = "the color used to render the caret")
    @Positive
    public void setCaretColor(Color c);

    @Positive
    public Color getSelectionColor();

    @Positive
    @BeanProperty(preferred = true, description = "color used to render selection background")
    @Positive
    public void setSelectionColor(Color c);

    @Positive
    public Color getSelectedTextColor();

    @Positive
    @BeanProperty(preferred = true, description = "color used to render selected text")
    @Positive
    public void setSelectedTextColor(Color c);

    @Positive
    public Color getDisabledTextColor();

    @Positive
    @BeanProperty(preferred = true, description = "color used to render disabled text")
    @Positive
    public void setDisabledTextColor(Color c);

    @Positive
    public void replaceSelection(String content);

    @Positive
    public String getText(int offs, int len) throws BadLocationException;

    @Positive
    @Deprecated()
    @Positive
    public Rectangle modelToView(int pos) throws BadLocationException;

    @Positive
    public Rectangle2D modelToView2D(int pos) throws BadLocationException;

    @Positive
    @Deprecated()
    @Positive
    public int viewToModel(Point pt);

    @Positive
    public int viewToModel2D(Point2D pt);

    @Positive
    public void cut();

    @Positive
    public void copy();

    @Positive
    public void paste();

    @Positive
    public void moveCaretPosition(int pos);

    @Positive
    @Interned
    @Positive
    public static final String FOCUS_ACCELERATOR_KEY;

    @Positive
    @BeanProperty(description = "accelerator character used to grab focus")
    @Positive
    public void setFocusAccelerator(char aKey);

    @Positive
    public char getFocusAccelerator();

    @Positive
    public void read(Reader in, Object desc) throws IOException;

    @Positive
    public void write(Writer out) throws IOException;

    @Positive
    public void removeNotify();

    @Positive
    @BeanProperty(bound = false, description = "the caret position")
    @Positive
    public void setCaretPosition(int position);

    @Positive
    @Transient
    @Positive
    public int getCaretPosition();

    @Positive
    @SafeEffect
    @Positive
    @BeanProperty(bound = false, description = "the text of this component")
    @Positive
    public void setText(String t);

    @Positive
    public String getText();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public String getSelectedText();

    @Positive
    public boolean isEditable();

    @Positive
    @BeanProperty(description = "specifies if the text can be edited")
    @Positive
    public void setEditable(boolean b);

    @Positive
    @Transient
    @Positive
    public int getSelectionStart();

    @Positive
    @BeanProperty(bound = false, description = "starting location of the selection.")
    @Positive
    public void setSelectionStart(int selectionStart);

    @Positive
    @Transient
    @Positive
    public int getSelectionEnd();

    @Positive
    @BeanProperty(bound = false, description = "ending location of the selection.")
    @Positive
    public void setSelectionEnd(int selectionEnd);

    @Positive
    public void select(int selectionStart, int selectionEnd);

    @Positive
    public void selectAll();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public String getToolTipText(MouseEvent event);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Dimension getPreferredScrollableViewportSize();

    @Positive
    public int getScrollableUnitIncrement(Rectangle visibleRect, int orientation, int direction);

    @Positive
    public int getScrollableBlockIncrement(Rectangle visibleRect, int orientation, int direction);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean getScrollableTracksViewportWidth();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean getScrollableTracksViewportHeight();

    @Positive
    public boolean print() throws PrinterException;

    @Positive
    public boolean print(final MessageFormat headerFormat, final MessageFormat footerFormat) throws PrinterException;

    @Positive
    public boolean print(final MessageFormat headerFormat, final MessageFormat footerFormat, final boolean showPrintDialog, final PrintService service, final PrintRequestAttributeSet attributes, final boolean interactive) throws PrinterException;

    @Positive
    public Printable getPrintable(final MessageFormat headerFormat, final MessageFormat footerFormat);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public class AccessibleJTextComponent extends AccessibleJComponent implements AccessibleText, CaretListener, DocumentListener, AccessibleAction, AccessibleEditableText, AccessibleExtendedText {

    @Positive
        public AccessibleJTextComponent() {
    @Positive
        }

    @Positive
        public void caretUpdate(CaretEvent e);

    @Positive
        public void insertUpdate(DocumentEvent e);

    @Positive
        public void removeUpdate(DocumentEvent e);

    @Positive
        public void changedUpdate(DocumentEvent e);

    @Positive
        public AccessibleStateSet getAccessibleStateSet();

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleText getAccessibleText();

    @Positive
        public int getIndexAtPoint(Point p);

    @Positive
        Rectangle getRootEditorRect();

    @Positive
        public Rectangle getCharacterBounds(int i);

    @Positive
        public int getCharCount();

    @Positive
        public int getCaretPosition();

    @Positive
        public AttributeSet getCharacterAttribute(int i);

    @Positive
        public int getSelectionStart();

    @Positive
        public int getSelectionEnd();

    @Positive
        public String getSelectedText();

    @Positive
        private class IndexedSegment extends Segment {

    @Positive
            public int modelOffset;
    @Positive
        }

    @Positive
        public String getAtIndex(int part, int index);

    @Positive
        public String getAfterIndex(int part, int index);

    @Positive
        public String getBeforeIndex(int part, int index);

    @Positive
        public AccessibleEditableText getAccessibleEditableText();

    @Positive
        public void setTextContents(String s);

    @Positive
        public void insertTextAtIndex(int index, String s);

    @Positive
        public String getTextRange(int startIndex, int endIndex);

    @Positive
        public void delete(int startIndex, int endIndex);

    @Positive
        public void cut(int startIndex, int endIndex);

    @Positive
        public void paste(int startIndex);

    @Positive
        public void replaceText(int startIndex, int endIndex, String s);

    @Positive
        public void selectText(int startIndex, int endIndex);

    @Positive
        public void setAttributes(int startIndex, int endIndex, AttributeSet as);

    @Positive
        public AccessibleTextSequence getTextSequenceAt(int part, int index);

    @Positive
        public AccessibleTextSequence getTextSequenceAfter(int part, int index);

    @Positive
        public AccessibleTextSequence getTextSequenceBefore(int part, int index);

    @Positive
        public Rectangle getTextBounds(int startIndex, int endIndex);

    @Positive
        public AccessibleAction getAccessibleAction();

    @Positive
        public int getAccessibleActionCount();

    @Positive
        public String getAccessibleActionDescription(int i);

    @Positive
        public boolean doAccessibleAction(int i);
    @Positive
    }

    @Positive
    public static final class DropLocation extends TransferHandler.DropLocation {

    @Positive
        public int getIndex();

    @Positive
        public Position.Bias getBias();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected String paramString();

    @Positive
    static class DefaultTransferHandler extends TransferHandler implements UIResource {

    @Positive
        public void exportToClipboard(JComponent comp, Clipboard clipboard, int action) throws IllegalStateException;

    @Positive
        public boolean importData(JComponent comp, Transferable t);

    @Positive
        public boolean canImport(JComponent comp, DataFlavor[] transferFlavors);

    @Positive
        public int getSourceActions(JComponent c);
    @Positive
    }

    @Positive
    static final JTextComponent getFocusedComponent();

    @Positive
    static class DefaultKeymap implements Keymap {

    @Positive
        public Action getDefaultAction();

    @Positive
        public void setDefaultAction(Action a);

    @Positive
        public String getName();

    @Positive
        public Action getAction(KeyStroke key);

    @Positive
        public KeyStroke[] getBoundKeyStrokes();

    @Positive
        public Action[] getBoundActions();

    @Positive
        public KeyStroke[] getKeyStrokesForAction(Action a);

    @Positive
        public boolean isLocallyDefined(KeyStroke key);

    @Positive
        public void addActionForKeyStroke(KeyStroke key, Action a);

    @Positive
        public void removeKeyStrokeBinding(KeyStroke key);

    @Positive
        public void removeBindings();

    @Positive
        public Keymap getResolveParent();

    @Positive
        public void setResolveParent(Keymap parent);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static class KeymapWrapper extends InputMap {

    @Positive
        public KeyStroke[] keys();

    @Positive
        public int size();

    @Positive
        public Object get(KeyStroke keyStroke);
    @Positive
    }

    @Positive
    static class KeymapActionMap extends ActionMap {

    @Positive
        public Object[] keys();

    @Positive
        public int size();

    @Positive
        public Action get(Object key);
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final String DEFAULT_KEYMAP;

    @Positive
    static class MutableCaretEvent extends CaretEvent implements ChangeListener, FocusListener, MouseListener {

    @Positive
        final void fire();

    @Positive
        public final String toString();

    @Positive
        public final int getDot();

    @Positive
        public final int getMark();

    @Positive
        public final void stateChanged(ChangeEvent e);

    @Positive
        public void focusGained(FocusEvent fe);

    @Positive
        public void focusLost(FocusEvent fe);

    @Positive
        public final void mousePressed(MouseEvent e);

    @Positive
        public final void mouseReleased(MouseEvent e);

    @Positive
        public final void mouseClicked(MouseEvent e);

    @Positive
        public final void mouseEntered(MouseEvent e);

    @Positive
        public final void mouseExited(MouseEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    protected void processInputMethodEvent(InputMethodEvent e);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public InputMethodRequests getInputMethodRequests();

    @Positive
    public void addInputMethodListener(InputMethodListener l);

    @Positive
    class InputMethodRequestsHandler implements InputMethodRequests, DocumentListener {

    @Positive
        public AttributedCharacterIterator cancelLatestCommittedText(Attribute[] attributes);

    @Positive
        public AttributedCharacterIterator getCommittedText(int beginIndex, int endIndex, Attribute[] attributes);

    @Positive
        public int getCommittedTextLength();

    @Positive
        public int getInsertPositionOffset();

    @Positive
        public TextHitInfo getLocationOffset(int x, int y);

    @Positive
        public Rectangle getTextLocation(TextHitInfo offset);

    @Positive
        public AttributedCharacterIterator getSelectedText(Attribute[] attributes);

    @Positive
        public void changedUpdate(DocumentEvent e);

    @Positive
        public void insertUpdate(DocumentEvent e);

    @Positive
        public void removeUpdate(DocumentEvent e);
    @Positive
    }

    @Positive
    protected boolean saveComposedText(int pos);

    @Positive
    protected void restoreComposedText();

    @Positive
    boolean composedTextExists();

    @Positive
    class ComposedTextCaret extends DefaultCaret implements Serializable {

    @Positive
        public void install(JTextComponent c);

    @Positive
        public void paint(Graphics g);

    @Positive
        protected void positionCaret(MouseEvent me);
    @Positive
    }

    @Positive
    private class DoSetCaretPosition implements Runnable {

    @Positive
        public void run();
    @Positive
    }
    @Positive
}
