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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.datatransfer.Clipboard;
    @Positive
import java.awt.datatransfer.ClipboardOwner;
    @Positive
import java.awt.datatransfer.StringSelection;
    @Positive
import java.awt.datatransfer.Transferable;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.FocusListener;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.awt.event.MouseListener;
    @Positive
import java.awt.event.MouseMotionListener;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.util.EventListener;
    @Positive
import javax.swing.Action;
    @Positive
import javax.swing.ActionMap;
    @Positive
import javax.swing.JPasswordField;
    @Positive
import javax.swing.JRootPane;
    @Positive
import javax.swing.SwingUtilities;
    @Positive
import javax.swing.Timer;
    @Positive
import javax.swing.TransferHandler;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.event.ChangeEvent;
    @Positive
import javax.swing.event.ChangeListener;
    @Positive
import javax.swing.event.DocumentEvent;
    @Positive
import javax.swing.event.DocumentListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.plaf.TextUI;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@SuppressWarnings("serial")
    @Positive
public class DefaultCaret extends Rectangle implements Caret, FocusListener, MouseListener, MouseMotionListener {

    @Positive
    public static final int UPDATE_WHEN_ON_EDT;

    @Positive
    public static final int NEVER_UPDATE;

    @Positive
    public static final int ALWAYS_UPDATE;

    @Positive
    public DefaultCaret() {
    @Positive
    }

    @Positive
    public void setUpdatePolicy(int policy);

    @Positive
    public int getUpdatePolicy();

    @Positive
    protected final JTextComponent getComponent();

    @Positive
    protected final synchronized void repaint();

    @Positive
    protected synchronized void damage(Rectangle r);

    @Positive
    protected void adjustVisibility(Rectangle nloc);

    @Positive
    protected Highlighter.HighlightPainter getSelectionPainter();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void positionCaret(MouseEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void moveCaret(MouseEvent e);

    @Positive
    public void focusGained(FocusEvent e);

    @Positive
    public void focusLost(FocusEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void mouseClicked(MouseEvent e);

    @Positive
    public void mousePressed(MouseEvent e);

    @Positive
    void adjustCaretAndFocus(MouseEvent e);

    @Positive
    public void mouseReleased(MouseEvent e);

    @Positive
    public void mouseEntered(MouseEvent e);

    @Positive
    public void mouseExited(MouseEvent e);

    @Positive
    public void mouseDragged(MouseEvent e);

    @Positive
    public void mouseMoved(MouseEvent e);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void paint(Graphics g);

    @Positive
    public void install(JTextComponent c);

    @Positive
    public void deinstall(JTextComponent c);

    @Positive
    public void addChangeListener(ChangeListener l);

    @Positive
    public void removeChangeListener(ChangeListener l);

    @Positive
    public ChangeListener[] getChangeListeners();

    @Positive
    protected void fireStateChanged();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    public void setSelectionVisible(boolean vis);

    @Positive
    public boolean isSelectionVisible();

    @Positive
    public boolean isActive();

    @Positive
    public boolean isVisible();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void setVisible(boolean e);

    @Positive
    public void setBlinkRate(int rate);

    @Positive
    public int getBlinkRate();

    @Positive
    public int getDot();

    @Positive
    public int getMark();

    @Positive
    public void setDot(int dot);

    @Positive
    public void moveDot(int dot);

    @Positive
    public void moveDot(int dot, Position.Bias dotBias);

    @Positive
    void handleMoveDot(int dot, Position.Bias dotBias);

    @Positive
    public void setDot(int dot, Position.Bias dotBias);

    @Positive
    void handleSetDot(int dot, Position.Bias dotBias);

    @Positive
    public Position.Bias getDotBias();

    @Positive
    public Position.Bias getMarkBias();

    @Positive
    boolean isDotLeftToRight();

    @Positive
    boolean isMarkLeftToRight();

    @Positive
    boolean isPositionLTR(int position, Position.Bias bias);

    @Positive
    Position.Bias guessBiasForOffset(int offset, Position.Bias lastBias, boolean lastLTR);

    @Positive
    void changeCaretPosition(int dot, Position.Bias dotBias);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    void repaintNewCaret();

    @Positive
    public void setMagicCaretPosition(Point p);

    @Positive
    public Point getMagicCaretPosition();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String toString();

    @Positive
    int getCaretWidth(int height);

    @Positive
    protected EventListenerList listenerList;

    @Positive
    protected transient ChangeEvent changeEvent;

    @Positive
    class SafeScroller implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    class Handler implements PropertyChangeListener, DocumentListener, ActionListener, ClipboardOwner {

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void actionPerformed(ActionEvent e);

    @Positive
        public void insertUpdate(DocumentEvent e);

    @Positive
        public void removeUpdate(DocumentEvent e);

    @Positive
        public void changedUpdate(DocumentEvent e);

    @Positive
        public void propertyChange(PropertyChangeEvent evt);

    @Positive
        public void lostOwnership(Clipboard clipboard, Transferable contents);
    @Positive
    }

    @Positive
    private class DefaultFilterBypass extends NavigationFilter.FilterBypass {

    @Positive
        public Caret getCaret();

    @Positive
        public void setDot(int dot, Position.Bias bias);

    @Positive
        public void moveDot(int dot, Position.Bias bias);
    @Positive
    }
    @Positive
}
