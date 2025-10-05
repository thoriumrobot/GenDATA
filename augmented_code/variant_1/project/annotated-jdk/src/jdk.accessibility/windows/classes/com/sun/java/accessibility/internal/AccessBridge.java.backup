/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.java.accessibility.internal;

import org.checkerframework.dataflow.qual.Pure;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.lang.*;
import java.lang.reflect.*;
import java.beans.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.text.*;
import javax.swing.tree.*;
import javax.swing.table.*;
import javax.swing.plaf.TreeUI;
import javax.accessibility.*;
import com.sun.java.accessibility.util.*;
import java.awt.geom.Rectangle2D;
import sun.awt.AWTAccessor;
import sun.awt.AppContext;
import sun.awt.SunToolkit;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

final public class AccessBridge {

    public AccessBridge() {
    }

    private class dllRunner implements Runnable {

        public void run();
    }

    private class shutdownHook implements Runnable {

        public void run();
    }

    private interface NativeWindowHandler {

        public Accessible getAccessibleFromNativeWindowHandle(int nativeHandle);
    }

    private class DefaultNativeWindowHandler implements NativeWindowHandler {

        public Accessible getAccessibleFromNativeWindowHandle(int nativeHandle);
    }

    private class ObjectReferences {

        private class Reference {

            public String toString();
        }

        String dump();

        void increment(Object o);

        void decrement(Object o);
    }

    private class EventHandler implements PropertyChangeListener, FocusListener, CaretListener, MenuListener, PopupMenuListener, MouseListener, WindowListener, ChangeListener {

        public void windowOpened(WindowEvent e);

        public void windowClosing(WindowEvent e);

        public void windowClosed(WindowEvent e);

        public void windowIconified(WindowEvent e);

        public void windowDeiconified(WindowEvent e);

        public void windowActivated(WindowEvent e);

        public void windowDeactivated(WindowEvent e);

        void addJavaEventNotification(long type);

        void removeJavaEventNotification(long type);

        void addAccessibilityEventNotification(long type);

        void removeAccessibilityEventNotification(long type);

        public void propertyChange(PropertyChangeEvent e);

        public void focusGained(FocusEvent e);

        public void stateChanged(ChangeEvent e);

        public void focusLost(FocusEvent e);

        public void caretUpdate(CaretEvent e);

        public void mouseClicked(MouseEvent e);

        public void mouseEntered(MouseEvent e);

        public void mouseExited(MouseEvent e);

        public void mousePressed(MouseEvent e);

        public void mouseReleased(MouseEvent e);

        public void menuCanceled(MenuEvent e);

        public void menuDeselected(MenuEvent e);

        public void menuSelected(MenuEvent e);

        public void popupMenuCanceled(PopupMenuEvent e);

        public void popupMenuWillBecomeInvisible(PopupMenuEvent e);

        public void popupMenuWillBecomeVisible(PopupMenuEvent e);
    }

    private class AccessibleJTreeNode extends AccessibleContext implements Accessible, AccessibleComponent, AccessibleSelection, AccessibleAction {

        public AccessibleContext getAccessibleContext();

        public String getAccessibleName();

        public void setAccessibleName(String s);

        public String getAccessibleDescription();

        public void setAccessibleDescription(String s);

        public AccessibleRole getAccessibleRole();

        public AccessibleStateSet getAccessibleStateSet();

        public Accessible getAccessibleParent();

        public int getAccessibleIndexInParent();

        public int getAccessibleChildrenCount();

        public Accessible getAccessibleChild(int i);

        public Locale getLocale();

        public void addPropertyChangeListener(PropertyChangeListener l);

        public void removePropertyChangeListener(PropertyChangeListener l);

        public AccessibleAction getAccessibleAction();

        public AccessibleComponent getAccessibleComponent();

        public AccessibleSelection getAccessibleSelection();

        public AccessibleText getAccessibleText();

        public AccessibleValue getAccessibleValue();

        public Color getBackground();

        public void setBackground(Color c);

        public Color getForeground();

        public void setForeground(Color c);

        public Cursor getCursor();

        public void setCursor(Cursor c);

        public Font getFont();

        public void setFont(Font f);

        public FontMetrics getFontMetrics(Font f);

        public boolean isEnabled();

        public void setEnabled(boolean b);

        public boolean isVisible();

        public void setVisible(boolean b);

        public boolean isShowing();

        @Pure
        public boolean contains(Point p);

        public Point getLocationOnScreen();

        public Point getLocation();

        public void setLocation(Point p);

        public Rectangle getBounds();

        public void setBounds(Rectangle r);

        public Dimension getSize();

        public void setSize(Dimension d);

        public Accessible getAccessibleAt(Point p);

        public boolean isFocusTraversable();

        public void requestFocus();

        public void addFocusListener(FocusListener l);

        public void removeFocusListener(FocusListener l);

        public int getAccessibleSelectionCount();

        public Accessible getAccessibleSelection(int i);

        public boolean isAccessibleChildSelected(int i);

        public void addAccessibleSelection(int i);

        public void removeAccessibleSelection(int i);

        public void clearAccessibleSelection();

        public void selectAllAccessibleSelection();

        public int getAccessibleActionCount();

        public String getAccessibleActionDescription(int i);

        public boolean doAccessibleAction(int i);
    }

    private static class InvocationUtils {

        public static <T> T invokeAndWait(final Callable<T> callable, final AccessibleExtendedTable accessibleTable);

        public static <T> T invokeAndWait(final Callable<T> callable, final Accessible accessible);

        public static <T> T invokeAndWait(final Callable<T> callable, final Component component);

        public static <T> T invokeAndWait(final Callable<T> callable, final AccessibleContext accessibleContext);

        public static void registerAccessibleContext(final AccessibleContext accessibleContext, final AppContext targetContext);

        private static class CallableWrapper<T> implements Runnable {

            public void run();

            T getResult() throws Exception;
        }
    }
}
