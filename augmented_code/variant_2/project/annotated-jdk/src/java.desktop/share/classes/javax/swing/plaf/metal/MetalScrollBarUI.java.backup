/*
 * Copyright (c) 1998, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.metal;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JScrollBar;
import javax.swing.UIManager;
import javax.swing.plaf.ComponentUI;
import javax.swing.plaf.basic.BasicScrollBarUI;
import static sun.swing.SwingUtilities2.drawHLine;
import static sun.swing.SwingUtilities2.drawRect;
import static sun.swing.SwingUtilities2.drawVLine;

@AnnotatedFor({ "interning" })
public class MetalScrollBarUI extends BasicScrollBarUI {

    protected MetalScrollButton increaseButton;

    protected MetalScrollButton decreaseButton;

    protected int scrollBarWidth;

    @Interned
    public static final String FREE_STANDING_PROP;

    protected boolean isFreeStanding;

    public MetalScrollBarUI() {
    }

    public static ComponentUI createUI(JComponent c);

    protected void installDefaults();

    protected void installListeners();

    protected PropertyChangeListener createPropertyChangeListener();

    protected void configureScrollBarColors();

    public Dimension getPreferredSize(JComponent c);

    protected JButton createDecreaseButton(int orientation);

    protected JButton createIncreaseButton(int orientation);

    protected void paintTrack(Graphics g, JComponent c, Rectangle trackBounds);

    protected void paintThumb(Graphics g, JComponent c, Rectangle thumbBounds);

    protected Dimension getMinimumThumbSize();

    protected void setThumbBounds(int x, int y, int width, int height);

    class ScrollBarListener extends BasicScrollBarUI.PropertyChangeHandler {

        public void propertyChange(PropertyChangeEvent e);

        public void handlePropertyChange(Object newValue);

        protected void toFlush();

        protected void toFreeStanding();
    }
}
