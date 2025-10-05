/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.event.ActionEvent;
    @Positive
import java.awt.peer.MenuComponentPeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleComponent;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleSelection;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.ComponentFactory;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class MenuComponent implements java.io.Serializable {

    @Positive
    @SuppressWarnings("removal")
    @Positive
    final AccessControlContext getAccessControlContext();

    @Positive
    public MenuComponent() throws HeadlessException {
    @Positive
    }

    @Positive
    String constructComponentName();

    @Positive
    final ComponentFactory getComponentFactory();

    @Positive
    public String getName();

    @Positive
    public void setName(String name);

    @Positive
    public MenuContainer getParent();

    @Positive
    final MenuContainer getParent_NoClientCode();

    @Positive
    public Font getFont();

    @Positive
    final Font getFont_NoClientCode();

    @Positive
    public void setFont(Font f);

    @Positive
    public void removeNotify();

    @Positive
    @Deprecated
    @Positive
    public boolean postEvent(Event evt);

    @Positive
    public final void dispatchEvent(AWTEvent e);

    @Positive
    void dispatchEventImpl(AWTEvent e);

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected String paramString();

    @Positive
    public String toString();

    @Positive
    protected final Object getTreeLock();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected abstract class AccessibleAWTMenuComponent extends AccessibleContext implements java.io.Serializable, AccessibleComponent, AccessibleSelection {

    @Positive
        protected AccessibleAWTMenuComponent() {
    @Positive
        }

    @Positive
        public AccessibleSelection getAccessibleSelection();

    @Positive
        public String getAccessibleName();

    @Positive
        public String getAccessibleDescription();

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
        public java.util.Locale getLocale();

    @Positive
        public AccessibleComponent getAccessibleComponent();

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
        public void setCursor(Cursor cursor);

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
        public void addFocusListener(java.awt.event.FocusListener l);

    @Positive
        public void removeFocusListener(java.awt.event.FocusListener l);

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
    }

    @Positive
    int getAccessibleIndexInParent();

    @Positive
    int getAccessibleChildIndex(MenuComponent child);

    @Positive
    AccessibleStateSet getAccessibleStateSet();
    @Positive
}

// CFWR semantic augmentation - variant 0
