/*
 * Copyright (c) 1996, 2015, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.*;
import java.lang.reflect.Array;
import java.util.EventListener;
import java.io.Serializable;
import java.io.ObjectOutputStream;
import java.io.IOException;
import java.util.EventListener;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class AWTEventMulticaster implements ComponentListener, ContainerListener, FocusListener, KeyListener, MouseListener, MouseMotionListener, WindowListener, WindowFocusListener, WindowStateListener, ActionListener, ItemListener, AdjustmentListener, TextListener, InputMethodListener, HierarchyListener, HierarchyBoundsListener, MouseWheelListener {

    protected final EventListener a;

    protected final EventListener b;

    protected AWTEventMulticaster(EventListener a, EventListener b) {
    }

    protected EventListener remove(EventListener oldl);

    public void componentResized(ComponentEvent e);

    public void componentMoved(ComponentEvent e);

    public void componentShown(ComponentEvent e);

    public void componentHidden(ComponentEvent e);

    public void componentAdded(ContainerEvent e);

    public void componentRemoved(ContainerEvent e);

    public void focusGained(FocusEvent e);

    public void focusLost(FocusEvent e);

    public void keyTyped(KeyEvent e);

    public void keyPressed(KeyEvent e);

    public void keyReleased(KeyEvent e);

    public void mouseClicked(MouseEvent e);

    public void mousePressed(MouseEvent e);

    public void mouseReleased(MouseEvent e);

    public void mouseEntered(MouseEvent e);

    public void mouseExited(MouseEvent e);

    public void mouseDragged(MouseEvent e);

    public void mouseMoved(MouseEvent e);

    public void windowOpened(WindowEvent e);

    public void windowClosing(WindowEvent e);

    public void windowClosed(WindowEvent e);

    public void windowIconified(WindowEvent e);

    public void windowDeiconified(WindowEvent e);

    public void windowActivated(WindowEvent e);

    public void windowDeactivated(WindowEvent e);

    public void windowStateChanged(WindowEvent e);

    public void windowGainedFocus(WindowEvent e);

    public void windowLostFocus(WindowEvent e);

    public void actionPerformed(ActionEvent e);

    public void itemStateChanged(ItemEvent e);

    public void adjustmentValueChanged(AdjustmentEvent e);

    public void textValueChanged(TextEvent e);

    public void inputMethodTextChanged(InputMethodEvent e);

    public void caretPositionChanged(InputMethodEvent e);

    public void hierarchyChanged(HierarchyEvent e);

    public void ancestorMoved(HierarchyEvent e);

    public void ancestorResized(HierarchyEvent e);

    public void mouseWheelMoved(MouseWheelEvent e);

    public static ComponentListener add(ComponentListener a, ComponentListener b);

    public static ContainerListener add(ContainerListener a, ContainerListener b);

    public static FocusListener add(FocusListener a, FocusListener b);

    public static KeyListener add(KeyListener a, KeyListener b);

    public static MouseListener add(MouseListener a, MouseListener b);

    public static MouseMotionListener add(MouseMotionListener a, MouseMotionListener b);

    public static WindowListener add(WindowListener a, WindowListener b);

    @SuppressWarnings("overloads")
    public static WindowStateListener add(WindowStateListener a, WindowStateListener b);

    public static WindowFocusListener add(WindowFocusListener a, WindowFocusListener b);

    @SuppressWarnings("overloads")
    public static ActionListener add(ActionListener a, ActionListener b);

    @SuppressWarnings("overloads")
    public static ItemListener add(ItemListener a, ItemListener b);

    @SuppressWarnings("overloads")
    public static AdjustmentListener add(AdjustmentListener a, AdjustmentListener b);

    @SuppressWarnings("overloads")
    public static TextListener add(TextListener a, TextListener b);

    public static InputMethodListener add(InputMethodListener a, InputMethodListener b);

    @SuppressWarnings("overloads")
    public static HierarchyListener add(HierarchyListener a, HierarchyListener b);

    public static HierarchyBoundsListener add(HierarchyBoundsListener a, HierarchyBoundsListener b);

    @SuppressWarnings("overloads")
    public static MouseWheelListener add(MouseWheelListener a, MouseWheelListener b);

    public static ComponentListener remove(ComponentListener l, ComponentListener oldl);

    public static ContainerListener remove(ContainerListener l, ContainerListener oldl);

    public static FocusListener remove(FocusListener l, FocusListener oldl);

    public static KeyListener remove(KeyListener l, KeyListener oldl);

    public static MouseListener remove(MouseListener l, MouseListener oldl);

    public static MouseMotionListener remove(MouseMotionListener l, MouseMotionListener oldl);

    public static WindowListener remove(WindowListener l, WindowListener oldl);

    @SuppressWarnings("overloads")
    public static WindowStateListener remove(WindowStateListener l, WindowStateListener oldl);

    public static WindowFocusListener remove(WindowFocusListener l, WindowFocusListener oldl);

    @SuppressWarnings("overloads")
    public static ActionListener remove(ActionListener l, ActionListener oldl);

    @SuppressWarnings("overloads")
    public static ItemListener remove(ItemListener l, ItemListener oldl);

    @SuppressWarnings("overloads")
    public static AdjustmentListener remove(AdjustmentListener l, AdjustmentListener oldl);

    @SuppressWarnings("overloads")
    public static TextListener remove(TextListener l, TextListener oldl);

    public static InputMethodListener remove(InputMethodListener l, InputMethodListener oldl);

    @SuppressWarnings("overloads")
    public static HierarchyListener remove(HierarchyListener l, HierarchyListener oldl);

    public static HierarchyBoundsListener remove(HierarchyBoundsListener l, HierarchyBoundsListener oldl);

    @SuppressWarnings("overloads")
    public static MouseWheelListener remove(MouseWheelListener l, MouseWheelListener oldl);

    protected static EventListener addInternal(EventListener a, EventListener b);

    protected static EventListener removeInternal(EventListener l, EventListener oldl);

    protected void saveInternal(ObjectOutputStream s, String k) throws IOException;

    protected static void save(ObjectOutputStream s, String k, EventListener l) throws IOException;

    @SuppressWarnings("unchecked")
    public static <T extends EventListener> T[] getListeners(EventListener l, Class<T> listenerType);
}
