/*
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.tree;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Insets;
import java.awt.Rectangle;
import javax.swing.plaf.BorderUIResource.EmptyBorderUIResource;
import javax.swing.plaf.ColorUIResource;
import javax.swing.plaf.FontUIResource;
import javax.swing.plaf.UIResource;
import javax.swing.plaf.basic.BasicGraphicsUtils;
import javax.swing.Icon;
import javax.swing.JLabel;
import javax.swing.JTree;
import javax.swing.LookAndFeel;
import javax.swing.UIManager;
import javax.swing.border.EmptyBorder;
import sun.swing.DefaultLookup;
import sun.swing.SwingUtilities2;

@AnnotatedFor({ "interning", "nullness" })
@SuppressWarnings("serial")
public class DefaultTreeCellRenderer extends JLabel implements TreeCellRenderer {

    protected boolean selected;

    protected boolean hasFocus;

    @Nullable
    protected transient Icon closedIcon;

    @Nullable
    protected transient Icon leafIcon;

    @Nullable
    protected transient Icon openIcon;

    @Nullable
    protected Color textSelectionColor;

    @Nullable
    protected Color textNonSelectionColor;

    @Nullable
    protected Color backgroundSelectionColor;

    @Nullable
    protected Color backgroundNonSelectionColor;

    @Nullable
    protected Color borderSelectionColor;

    public DefaultTreeCellRenderer() {
    }

    public void updateUI();

    @Nullable
    public Icon getDefaultOpenIcon();

    @Nullable
    public Icon getDefaultClosedIcon();

    @Nullable
    public Icon getDefaultLeafIcon();

    public void setOpenIcon(@Nullable Icon newIcon);

    @Nullable
    public Icon getOpenIcon();

    public void setClosedIcon(@Nullable Icon newIcon);

    @Nullable
    public Icon getClosedIcon();

    public void setLeafIcon(@Nullable Icon newIcon);

    @Nullable
    public Icon getLeafIcon();

    public void setTextSelectionColor(@Nullable Color newColor);

    @Nullable
    public Color getTextSelectionColor();

    public void setTextNonSelectionColor(@Nullable Color newColor);

    @Nullable
    public Color getTextNonSelectionColor();

    public void setBackgroundSelectionColor(@Nullable Color newColor);

    @Nullable
    public Color getBackgroundSelectionColor();

    public void setBackgroundNonSelectionColor(@Nullable Color newColor);

    @Nullable
    public Color getBackgroundNonSelectionColor();

    public void setBorderSelectionColor(@Nullable Color newColor);

    @Nullable
    public Color getBorderSelectionColor();

    public void setFont(@Nullable Font font);

    @Nullable
    public Font getFont();

    public void setBackground(@Nullable Color color);

    public Component getTreeCellRendererComponent(JTree tree, @Nullable Object value, boolean sel, boolean expanded, boolean leaf, int row, boolean hasFocus);

    public void paint(Graphics g);

    public Dimension getPreferredSize();

    public void validate();

    public void invalidate();

    public void revalidate();

    public void repaint(long tm, int x, int y, int width, int height);

    public void repaint(Rectangle r);

    public void repaint();

    protected void firePropertyChange(@Interned String propertyName, @Nullable Object oldValue, @Nullable Object newValue);

    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    public void firePropertyChange(String propertyName, int oldValue, int newValue);

    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);
}
