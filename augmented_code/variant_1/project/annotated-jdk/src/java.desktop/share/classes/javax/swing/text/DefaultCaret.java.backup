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
package javax.swing.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.Graphics;
import java.awt.HeadlessException;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.util.EventListener;
import javax.swing.Action;
import javax.swing.ActionMap;
import javax.swing.JPasswordField;
import javax.swing.JRootPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.TransferHandler;
import javax.swing.UIManager;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.event.EventListenerList;
import javax.swing.plaf.TextUI;
import sun.swing.SwingUtilities2;

@SuppressWarnings("serial")
public class DefaultCaret extends Rectangle implements Caret, FocusListener, MouseListener, MouseMotionListener {

    public static final int UPDATE_WHEN_ON_EDT;

    public static final int NEVER_UPDATE;

    public static final int ALWAYS_UPDATE;

    public DefaultCaret() {
    }

    public void setUpdatePolicy(int policy);

    public int getUpdatePolicy();

    protected final JTextComponent getComponent();

    protected final synchronized void repaint();

    protected synchronized void damage(Rectangle r);

    protected void adjustVisibility(Rectangle nloc);

    protected Highlighter.HighlightPainter getSelectionPainter();

    @SuppressWarnings("deprecation")
    protected void positionCaret(MouseEvent e);

    @SuppressWarnings("deprecation")
    protected void moveCaret(MouseEvent e);

    public void focusGained(FocusEvent e);

    public void focusLost(FocusEvent e);

    @SuppressWarnings("deprecation")
    public void mouseClicked(MouseEvent e);

    public void mousePressed(MouseEvent e);

    void adjustCaretAndFocus(MouseEvent e);

    public void mouseReleased(MouseEvent e);

    public void mouseEntered(MouseEvent e);

    public void mouseExited(MouseEvent e);

    public void mouseDragged(MouseEvent e);

    public void mouseMoved(MouseEvent e);

    @SuppressWarnings("deprecation")
    public void paint(Graphics g);

    public void install(JTextComponent c);

    public void deinstall(JTextComponent c);

    public void addChangeListener(ChangeListener l);

    public void removeChangeListener(ChangeListener l);

    public ChangeListener[] getChangeListeners();

    protected void fireStateChanged();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    public void setSelectionVisible(boolean vis);

    public boolean isSelectionVisible();

    public boolean isActive();

    public boolean isVisible();

    @SuppressWarnings("deprecation")
    public void setVisible(boolean e);

    public void setBlinkRate(int rate);

    public int getBlinkRate();

    public int getDot();

    public int getMark();

    public void setDot(int dot);

    public void moveDot(int dot);

    public void moveDot(int dot, Position.Bias dotBias);

    void handleMoveDot(int dot, Position.Bias dotBias);

    public void setDot(int dot, Position.Bias dotBias);

    void handleSetDot(int dot, Position.Bias dotBias);

    public Position.Bias getDotBias();

    public Position.Bias getMarkBias();

    boolean isDotLeftToRight();

    boolean isMarkLeftToRight();

    boolean isPositionLTR(int position, Position.Bias bias);

    Position.Bias guessBiasForOffset(int offset, Position.Bias lastBias, boolean lastLTR);

    void changeCaretPosition(int dot, Position.Bias dotBias);

    @SuppressWarnings("deprecation")
    void repaintNewCaret();

    public void setMagicCaretPosition(Point p);

    public Point getMagicCaretPosition();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String toString();

    int getCaretWidth(int height);

    protected EventListenerList listenerList;

    protected transient ChangeEvent changeEvent;

    class SafeScroller implements Runnable {

        public void run();
    }

    class Handler implements PropertyChangeListener, DocumentListener, ActionListener, ClipboardOwner {

        @SuppressWarnings("deprecation")
        public void actionPerformed(ActionEvent e);

        public void insertUpdate(DocumentEvent e);

        public void removeUpdate(DocumentEvent e);

        public void changedUpdate(DocumentEvent e);

        public void propertyChange(PropertyChangeEvent evt);

        public void lostOwnership(Clipboard clipboard, Transferable contents);
    }

    private class DefaultFilterBypass extends NavigationFilter.FilterBypass {

        public Caret getCaret();

        public void setDot(int dot, Position.Bias bias);

        public void moveDot(int dot, Position.Bias bias);
    }
}
