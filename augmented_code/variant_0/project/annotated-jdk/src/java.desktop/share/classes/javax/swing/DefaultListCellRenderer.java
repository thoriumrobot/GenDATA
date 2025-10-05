/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.border.*;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.io.Serializable;
    @Positive
import sun.swing.DefaultLookup;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class DefaultListCellRenderer extends JLabel implements ListCellRenderer<Object>, Serializable {

    @Positive
    protected static Border noFocusBorder;

    @Positive
    public DefaultListCellRenderer() {
    @Positive
    }

    @Positive
    public Component getListCellRendererComponent(JList<?> list, @Nullable Object value, int index, boolean isSelected, boolean cellHasFocus);

    @Positive
    @Override
    @Positive
    public boolean isOpaque();

    @Positive
    @Override
    @Positive
    public void validate();

    @Positive
    @Override
    @Positive
    public void invalidate();

    @Positive
    @Override
    @Positive
    public void repaint();

    @Positive
    @Override
    @Positive
    public void revalidate();

    @Positive
    @Override
    @Positive
    public void repaint(long tm, int x, int y, int width, int height);

    @Positive
    @Override
    @Positive
    public void repaint(Rectangle r);

    @Positive
    @Override
    @Positive
    protected void firePropertyChange(@Interned String propertyName, Object oldValue, Object newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, int oldValue, int newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    @Positive
    @Override
    @Positive
    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class UIResource extends DefaultListCellRenderer implements javax.swing.plaf.UIResource {

    @Positive
        public UIResource() {
    @Positive
        }
    @Positive
    }
    @Positive
}
