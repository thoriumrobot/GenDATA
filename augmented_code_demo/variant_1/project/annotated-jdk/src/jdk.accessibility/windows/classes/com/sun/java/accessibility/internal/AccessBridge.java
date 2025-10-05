/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.java.accessibility.internal;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.util.*;
    @Positive
import java.lang.*;
    @Positive
import java.lang.reflect.*;
    @Positive
import java.beans.*;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.text.*;
    @Positive
import javax.swing.tree.*;
    @Positive
import javax.swing.table.*;
    @Positive
import javax.swing.plaf.TreeUI;
    @Positive
import javax.accessibility.*;
    @Positive
import com.sun.java.accessibility.util.*;
    @Positive
import java.awt.geom.Rectangle2D;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import java.util.concurrent.Callable;
    @Positive
import java.util.concurrent.ConcurrentHashMap;

    @Positive
final public class AccessBridge {

    @Positive
    public AccessBridge() {
    @Positive
    }

    @Positive
    private class dllRunner implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    private class shutdownHook implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    private interface NativeWindowHandler {

    @Positive
        public Accessible getAccessibleFromNativeWindowHandle(int nativeHandle);
    @Positive
    }

    @Positive
    private class DefaultNativeWindowHandler implements NativeWindowHandler {

    @Positive
        public Accessible getAccessibleFromNativeWindowHandle(int nativeHandle);
    @Positive
    }

    @Positive
    private class ObjectReferences {

    @Positive
        private class Reference {

    @Positive
            public String toString();
    @Positive
        }

    @Positive
        String dump();

    @Positive
        void increment(Object o);

    @Positive
        void decrement(Object o);
    @Positive
    }

    @Positive
    private class EventHandler implements PropertyChangeListener, FocusListener, CaretListener, MenuListener, PopupMenuListener, MouseListener, WindowListener, ChangeListener {

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
        void addJavaEventNotification(long type);

    @Positive
        void removeJavaEventNotification(long type);

    @Positive
        void addAccessibilityEventNotification(long type);

    @Positive
        void removeAccessibilityEventNotification(long type);

    @Positive
        public void propertyChange(PropertyChangeEvent e);

    @Positive
        public void focusGained(FocusEvent e);

    @Positive
        public void stateChanged(ChangeEvent e);

    @Positive
        public void focusLost(FocusEvent e);

    @Positive
        public void caretUpdate(CaretEvent e);

    @Positive
        public void mouseClicked(MouseEvent e);

    @Positive
        public void mouseEntered(MouseEvent e);

    @Positive
        public void mouseExited(MouseEvent e);

    @Positive
        public void mousePressed(MouseEvent e);

    @Positive
        public void mouseReleased(MouseEvent e);

    @Positive
        public void menuCanceled(MenuEvent e);

    @Positive
        public void menuDeselected(MenuEvent e);

    @Positive
        public void menuSelected(MenuEvent e);

    @Positive
        public void popupMenuCanceled(PopupMenuEvent e);

    @Positive
        public void popupMenuWillBecomeInvisible(PopupMenuEvent e);

    @Positive
        public void popupMenuWillBecomeVisible(PopupMenuEvent e);
    @Positive
    }

    @Positive
    private class AccessibleJTreeNode extends AccessibleContext implements Accessible, AccessibleComponent, AccessibleSelection, AccessibleAction {

    @Positive
        public AccessibleContext getAccessibleContext();

    @Positive
        public String getAccessibleName();

    @Positive
        public void setAccessibleName(String s);

    @Positive
        public String getAccessibleDescription();

    @Positive
        public void setAccessibleDescription(String s);

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();

    @Positive
        public Accessible getAccessibleParent();

    @Positive
        public int getAccessibleIndexInParent();

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);

    @Positive
        public Locale getLocale();

    @Positive
        public void addPropertyChangeListener(PropertyChangeListener l);

    @Positive
        public void removePropertyChangeListener(PropertyChangeListener l);

    @Positive
        public AccessibleAction getAccessibleAction();

    @Positive
        public AccessibleComponent getAccessibleComponent();

    @Positive
        public AccessibleSelection getAccessibleSelection();

    @Positive
        public AccessibleText getAccessibleText();

    @Positive
        public AccessibleValue getAccessibleValue();

    @Positive
        public Color getBackground();

    @Positive
        public void setBackground(Color c);

    @Positive
        public Color getForeground();

    @Positive
        public void setForeground(Color c);

    @Positive
        public Cursor getCursor();

    @Positive
        public void setCursor(Cursor c);

    @Positive
        public Font getFont();

    @Positive
        public void setFont(Font f);

    @Positive
        public FontMetrics getFontMetrics(Font f);

    @Positive
        public boolean isEnabled();

    @Positive
        public void setEnabled(boolean b);

    @Positive
        public boolean isVisible();

    @Positive
        public void setVisible(boolean b);

    @Positive
        public boolean isShowing();

    @Positive
        @Pure
    @Positive
        public boolean contains(Point p);

    @Positive
        public Point getLocationOnScreen();

    @Positive
        public Point getLocation();

    @Positive
        public void setLocation(Point p);

    @Positive
        public Rectangle getBounds();

    @Positive
        public void setBounds(Rectangle r);

    @Positive
        public Dimension getSize();

    @Positive
        public void setSize(Dimension d);

    @Positive
        public Accessible getAccessibleAt(Point p);

    @Positive
        public boolean isFocusTraversable();

    @Positive
        public void requestFocus();

    @Positive
        public void addFocusListener(FocusListener l);

    @Positive
        public void removeFocusListener(FocusListener l);

    @Positive
        public int getAccessibleSelectionCount();

    @Positive
        public Accessible getAccessibleSelection(int i);

    @Positive
        public boolean isAccessibleChildSelected(int i);

    @Positive
        public void addAccessibleSelection(int i);

    @Positive
        public void removeAccessibleSelection(int i);

    @Positive
        public void clearAccessibleSelection();

    @Positive
        public void selectAllAccessibleSelection();

    @Positive
        public int getAccessibleActionCount();

    @Positive
        public String getAccessibleActionDescription(int i);

    @Positive
        public boolean doAccessibleAction(int i);
    @Positive
    }

    @Positive
    private static class InvocationUtils {

    @Positive
        public static <T> T invokeAndWait(final Callable<T> callable, final AccessibleExtendedTable accessibleTable);

    @Positive
        public static <T> T invokeAndWait(final Callable<T> callable, final Accessible accessible);

    @Positive
        public static <T> T invokeAndWait(final Callable<T> callable, final Component component);

    @Positive
        public static <T> T invokeAndWait(final Callable<T> callable, final AccessibleContext accessibleContext);

    @Positive
        public static void registerAccessibleContext(final AccessibleContext accessibleContext, final AppContext targetContext);

    @Positive
        private static class CallableWrapper<T> implements Runnable {

    @Positive
            public void run();

    @Positive
            T getResult() throws Exception;
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
