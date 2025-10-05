/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.tree;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Rectangle;
    @Positive
import javax.swing.plaf.BorderUIResource.EmptyBorderUIResource;
    @Positive
import javax.swing.plaf.ColorUIResource;
    @Positive
import javax.swing.plaf.FontUIResource;
    @Positive
import javax.swing.plaf.UIResource;
    @Positive
import javax.swing.plaf.basic.BasicGraphicsUtils;
    @Positive
import javax.swing.Icon;
    @Positive
import javax.swing.JLabel;
    @Positive
import javax.swing.JTree;
    @Positive
import javax.swing.LookAndFeel;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.border.EmptyBorder;
    @Positive
import sun.swing.DefaultLookup;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class DefaultTreeCellRenderer extends JLabel implements TreeCellRenderer {

    @Positive
    protected boolean selected;

    @Positive
    protected boolean hasFocus;

    @Positive
    @Nullable
    @Positive
    protected transient Icon closedIcon;

    @Positive
    @Nullable
    @Positive
    protected transient Icon leafIcon;

    @Positive
    @Nullable
    @Positive
    protected transient Icon openIcon;

    @Positive
    @Nullable
    @Positive
    protected Color textSelectionColor;

    @Positive
    @Nullable
    @Positive
    protected Color textNonSelectionColor;

    @Positive
    @Nullable
    @Positive
    protected Color backgroundSelectionColor;

    @Positive
    @Nullable
    @Positive
    protected Color backgroundNonSelectionColor;

    @Positive
    @Nullable
    @Positive
    protected Color borderSelectionColor;

    @Positive
    public DefaultTreeCellRenderer() {
    @Positive
    }

    @Positive
    public void updateUI();

    @Positive
    @Nullable
    @Positive
    public Icon getDefaultOpenIcon();

    @Positive
    @Nullable
    @Positive
    public Icon getDefaultClosedIcon();

    @Positive
    @Nullable
    @Positive
    public Icon getDefaultLeafIcon();

    @Positive
    public void setOpenIcon(@Nullable Icon newIcon);

    @Positive
    @Nullable
    @Positive
    public Icon getOpenIcon();

    @Positive
    public void setClosedIcon(@Nullable Icon newIcon);

    @Positive
    @Nullable
    @Positive
    public Icon getClosedIcon();

    @Positive
    public void setLeafIcon(@Nullable Icon newIcon);

    @Positive
    @Nullable
    @Positive
    public Icon getLeafIcon();

    @Positive
    public void setTextSelectionColor(@Nullable Color newColor);

    @Positive
    @Nullable
    @Positive
    public Color getTextSelectionColor();

    @Positive
    public void setTextNonSelectionColor(@Nullable Color newColor);

    @Positive
    @Nullable
    @Positive
    public Color getTextNonSelectionColor();

    @Positive
    public void setBackgroundSelectionColor(@Nullable Color newColor);

    @Positive
    @Nullable
    @Positive
    public Color getBackgroundSelectionColor();

    @Positive
    public void setBackgroundNonSelectionColor(@Nullable Color newColor);

    @Positive
    @Nullable
    @Positive
    public Color getBackgroundNonSelectionColor();

    @Positive
    public void setBorderSelectionColor(@Nullable Color newColor);

    @Positive
    @Nullable
    @Positive
    public Color getBorderSelectionColor();

    @Positive
    public void setFont(@Nullable Font font);

    @Positive
    @Nullable
    @Positive
    public Font getFont();

    @Positive
    public void setBackground(@Nullable Color color);

    @Positive
    public Component getTreeCellRendererComponent(JTree tree, @Nullable Object value, boolean sel, boolean expanded, boolean leaf, int row, boolean hasFocus);

    @Positive
    public void paint(Graphics g);

    @Positive
    public Dimension getPreferredSize();

    @Positive
    public void validate();

    @Positive
    public void invalidate();

    @Positive
    public void revalidate();

    @Positive
    public void repaint(long tm, int x, int y, int width, int height);

    @Positive
    public void repaint(Rectangle r);

    @Positive
    public void repaint();

    @Positive
    protected void firePropertyChange(@Interned String propertyName, @Nullable Object oldValue, @Nullable Object newValue);

    @Positive
    public void firePropertyChange(String propertyName, byte oldValue, byte newValue);

    @Positive
    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    @Positive
    public void firePropertyChange(String propertyName, short oldValue, short newValue);

    @Positive
    public void firePropertyChange(String propertyName, int oldValue, int newValue);

    @Positive
    public void firePropertyChange(String propertyName, long oldValue, long newValue);

    @Positive
    public void firePropertyChange(String propertyName, float oldValue, float newValue);

    @Positive
    public void firePropertyChange(String propertyName, double oldValue, double newValue);

    @Positive
    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);
    @Positive
}
