/*
    @Positive
 * Copyright (c) 2002, 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.java.accessibility.util;

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
import com.sun.java.accessibility.util.internal.*;
    @Positive
import java.beans.*;
    @Positive
import java.util.*;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import javax.accessibility.*;

    @Positive
public class Translator extends AccessibleContext implements Accessible, AccessibleComponent {

    @Positive
    protected Object source;

    @Positive
    protected static Class<?> getTranslatorClass(Class<?> c);

    @Positive
    public static Accessible getAccessible(Object o);

    @Positive
    public Translator() {
    @Positive
    }

    @Positive
    public Translator(Object o) {
    @Positive
    }

    @Positive
    public Object getSource();

    @Positive
    public void setSource(Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

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
    public Locale getLocale() throws IllegalComponentStateException;

    @Positive
    public void addPropertyChangeListener(PropertyChangeListener l);

    @Positive
    public void removePropertyChangeListener(PropertyChangeListener l);

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
    @SuppressWarnings("deprecation")
    @Positive
    public boolean isFocusTraversable();

    @Positive
    public void requestFocus();

    @Positive
    public synchronized void addFocusListener(FocusListener l);

    @Positive
    public synchronized void removeFocusListener(FocusListener l);
    @Positive
}

// CFWR semantic augmentation - variant 1
