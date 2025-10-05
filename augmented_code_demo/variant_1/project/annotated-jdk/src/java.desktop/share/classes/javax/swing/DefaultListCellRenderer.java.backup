/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.border.*;
import java.awt.Component;
import java.awt.Color;
import java.awt.Rectangle;
import java.io.Serializable;
import sun.swing.DefaultLookup;
import sun.swing.SwingUtilities2;

@AnnotatedFor({ "interning", "nullness" })
@SuppressWarnings("serial")
public class DefaultListCellRenderer extends JLabel implements ListCellRenderer<Object>, Serializable {

    protected static Border noFocusBorder;

    public DefaultListCellRenderer() {
    }

    public Component getListCellRendererComponent(JList<?> list, @Nullable Object value, int index, boolean isSelected, boolean cellHasFocus);

    @Override
    public boolean isOpaque();

    @Override
    public void validate();

    @Override
    public void invalidate();

    @Override
    public void repaint();

    @Override
    public void revalidate();

    @Override
    public void repaint(long tm, int x, int y, int width, int height);

    @Override
    public void repaint(Rectangle r);

    @Override
    protected void firePropertyChange(@Interned String propertyName, Object oldValue, Object newValue);

    @Override
    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    @Override
    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    @Override
    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    @Override
    public void firePropertyChange(String propertyName, int oldValue, int newValue);

    @Override
    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    @Override
    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    @Override
    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    @Override
    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    @SuppressWarnings("serial")
    public static class UIResource extends DefaultListCellRenderer implements javax.swing.plaf.UIResource {

        public UIResource() {
        }
    }
}
