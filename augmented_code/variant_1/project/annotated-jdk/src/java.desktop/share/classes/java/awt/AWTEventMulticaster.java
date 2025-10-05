/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2015, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.*;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.EventListener;
    @Positive
import java.io.Serializable;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.util.EventListener;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class AWTEventMulticaster implements ComponentListener, ContainerListener, FocusListener, KeyListener, MouseListener, MouseMotionListener, WindowListener, WindowFocusListener, WindowStateListener, ActionListener, ItemListener, AdjustmentListener, TextListener, InputMethodListener, HierarchyListener, HierarchyBoundsListener, MouseWheelListener {

    @Positive
    protected final EventListener a;

    @Positive
    protected final EventListener b;

    @Positive
    protected AWTEventMulticaster(EventListener a, EventListener b) {
    @Positive
    }

    @Positive
    protected EventListener remove(EventListener oldl);

    @Positive
    public void componentResized(ComponentEvent e);

    @Positive
    public void componentMoved(ComponentEvent e);

    @Positive
    public void componentShown(ComponentEvent e);

    @Positive
    public void componentHidden(ComponentEvent e);

    @Positive
    public void componentAdded(ContainerEvent e);

    @Positive
    public void componentRemoved(ContainerEvent e);

    @Positive
    public void focusGained(FocusEvent e);

    @Positive
    public void focusLost(FocusEvent e);

    @Positive
    public void keyTyped(KeyEvent e);

    @Positive
    public void keyPressed(KeyEvent e);

    @Positive
    public void keyReleased(KeyEvent e);

    @Positive
    public void mouseClicked(MouseEvent e);

    @Positive
    public void mousePressed(MouseEvent e);

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
    public void windowOpened(WindowEvent e);

    @Positive
    public void windowClosing(WindowEvent e);

    @Positive
    public void windowClosed(WindowEvent e);

    @Positive
    public void windowIconified(WindowEvent e);

    @Positive
    public void windowDeiconified(WindowEvent e);

    @Positive
    public void windowActivated(WindowEvent e);

    @Positive
    public void windowDeactivated(WindowEvent e);

    @Positive
    public void windowStateChanged(WindowEvent e);

    @Positive
    public void windowGainedFocus(WindowEvent e);

    @Positive
    public void windowLostFocus(WindowEvent e);

    @Positive
    public void actionPerformed(ActionEvent e);

    @Positive
    public void itemStateChanged(ItemEvent e);

    @Positive
    public void adjustmentValueChanged(AdjustmentEvent e);

    @Positive
    public void textValueChanged(TextEvent e);

    @Positive
    public void inputMethodTextChanged(InputMethodEvent e);

    @Positive
    public void caretPositionChanged(InputMethodEvent e);

    @Positive
    public void hierarchyChanged(HierarchyEvent e);

    @Positive
    public void ancestorMoved(HierarchyEvent e);

    @Positive
    public void ancestorResized(HierarchyEvent e);

    @Positive
    public void mouseWheelMoved(MouseWheelEvent e);

    @Positive
    public static ComponentListener add(ComponentListener a, ComponentListener b);

    @Positive
    public static ContainerListener add(ContainerListener a, ContainerListener b);

    @Positive
    public static FocusListener add(FocusListener a, FocusListener b);

    @Positive
    public static KeyListener add(KeyListener a, KeyListener b);

    @Positive
    public static MouseListener add(MouseListener a, MouseListener b);

    @Positive
    public static MouseMotionListener add(MouseMotionListener a, MouseMotionListener b);

    @Positive
    public static WindowListener add(WindowListener a, WindowListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static WindowStateListener add(WindowStateListener a, WindowStateListener b);

    @Positive
    public static WindowFocusListener add(WindowFocusListener a, WindowFocusListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static ActionListener add(ActionListener a, ActionListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static ItemListener add(ItemListener a, ItemListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static AdjustmentListener add(AdjustmentListener a, AdjustmentListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static TextListener add(TextListener a, TextListener b);

    @Positive
    public static InputMethodListener add(InputMethodListener a, InputMethodListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static HierarchyListener add(HierarchyListener a, HierarchyListener b);

    @Positive
    public static HierarchyBoundsListener add(HierarchyBoundsListener a, HierarchyBoundsListener b);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static MouseWheelListener add(MouseWheelListener a, MouseWheelListener b);

    @Positive
    public static ComponentListener remove(ComponentListener l, ComponentListener oldl);

    @Positive
    public static ContainerListener remove(ContainerListener l, ContainerListener oldl);

    @Positive
    public static FocusListener remove(FocusListener l, FocusListener oldl);

    @Positive
    public static KeyListener remove(KeyListener l, KeyListener oldl);

    @Positive
    public static MouseListener remove(MouseListener l, MouseListener oldl);

    @Positive
    public static MouseMotionListener remove(MouseMotionListener l, MouseMotionListener oldl);

    @Positive
    public static WindowListener remove(WindowListener l, WindowListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static WindowStateListener remove(WindowStateListener l, WindowStateListener oldl);

    @Positive
    public static WindowFocusListener remove(WindowFocusListener l, WindowFocusListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static ActionListener remove(ActionListener l, ActionListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static ItemListener remove(ItemListener l, ItemListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static AdjustmentListener remove(AdjustmentListener l, AdjustmentListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static TextListener remove(TextListener l, TextListener oldl);

    @Positive
    public static InputMethodListener remove(InputMethodListener l, InputMethodListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static HierarchyListener remove(HierarchyListener l, HierarchyListener oldl);

    @Positive
    public static HierarchyBoundsListener remove(HierarchyBoundsListener l, HierarchyBoundsListener oldl);

    @Positive
    @SuppressWarnings("overloads")
    @Positive
    public static MouseWheelListener remove(MouseWheelListener l, MouseWheelListener oldl);

    @Positive
    protected static EventListener addInternal(EventListener a, EventListener b);

    @Positive
    protected static EventListener removeInternal(EventListener l, EventListener oldl);

    @Positive
    protected void saveInternal(ObjectOutputStream s, String k) throws IOException;

    @Positive
    protected static void save(ObjectOutputStream s, String k, EventListener l) throws IOException;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T extends EventListener> T[] getListeners(EventListener l, Class<T> listenerType);
    @Positive
}
