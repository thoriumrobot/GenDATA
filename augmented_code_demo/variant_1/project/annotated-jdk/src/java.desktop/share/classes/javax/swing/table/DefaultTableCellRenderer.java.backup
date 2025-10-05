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
package javax.swing.table;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.Component;
import java.awt.Color;
import java.awt.Rectangle;
import java.io.Serializable;
import sun.swing.DefaultLookup;
import sun.swing.SwingUtilities2;

@AnnotatedFor({ "index", "interning", "nullness" })
@SuppressWarnings("serial")
public class DefaultTableCellRenderer extends JLabel implements TableCellRenderer, Serializable {

    protected static Border noFocusBorder;

    public DefaultTableCellRenderer() {
    }

    public void setForeground(Color c);

    public void setBackground(Color c);

    public void updateUI();

    public Component getTableCellRendererComponent(JTable table, @Nullable Object value, boolean isSelected, boolean hasFocus, @NonNegative int row, @NonNegative int column);

    public boolean isOpaque();

    public void invalidate();

    public void validate();

    public void revalidate();

    public void repaint(long tm, int x, int y, int width, int height);

    public void repaint(Rectangle r);

    public void repaint();

    protected void firePropertyChange(@Interned String propertyName, @Nullable Object oldValue, @Nullable Object newValue);

    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    protected void setValue(@Nullable Object value);

    @SuppressWarnings("serial")
    public static class UIResource extends DefaultTableCellRenderer implements javax.swing.plaf.UIResource {

        public UIResource() {
        }
    }
}
