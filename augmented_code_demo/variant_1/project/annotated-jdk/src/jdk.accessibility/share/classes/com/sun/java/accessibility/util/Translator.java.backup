/*
 * Copyright (c) 2002, 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.java.accessibility.util;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.java.accessibility.util.internal.*;
import java.beans.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.accessibility.*;

public class Translator extends AccessibleContext implements Accessible, AccessibleComponent {

    protected Object source;

    protected static Class<?> getTranslatorClass(Class<?> c);

    public static Accessible getAccessible(Object o);

    public Translator() {
    }

    public Translator(Object o) {
    }

    public Object getSource();

    public void setSource(Object o);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();

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

    public Locale getLocale() throws IllegalComponentStateException;

    public void addPropertyChangeListener(PropertyChangeListener l);

    public void removePropertyChangeListener(PropertyChangeListener l);

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

    @SuppressWarnings("deprecation")
    public boolean isFocusTraversable();

    public void requestFocus();

    public synchronized void addFocusListener(FocusListener l);

    public synchronized void removeFocusListener(FocusListener l);
}
