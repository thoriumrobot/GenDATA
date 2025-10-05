/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1998, 2014, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.metal;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import javax.swing.JButton;
    @Positive
import javax.swing.JComponent;
    @Positive
import javax.swing.JScrollBar;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import javax.swing.plaf.basic.BasicScrollBarUI;
    @Positive
import static sun.swing.SwingUtilities2.drawHLine;
    @Positive
import static sun.swing.SwingUtilities2.drawRect;
    @Positive
import static sun.swing.SwingUtilities2.drawVLine;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class MetalScrollBarUI extends BasicScrollBarUI {

    @Positive
    protected MetalScrollButton increaseButton;

    @Positive
    protected MetalScrollButton decreaseButton;

    @Positive
    protected int scrollBarWidth;

    @Positive
    @Interned
    @Positive
    public static final String FREE_STANDING_PROP;

    @Positive
    protected boolean isFreeStanding;

    @Positive
    public MetalScrollBarUI() {
    @Positive
    }

    @Positive
    public static ComponentUI createUI(JComponent c);

    @Positive
    protected void installDefaults();

    @Positive
    protected void installListeners();

    @Positive
    protected PropertyChangeListener createPropertyChangeListener();

    @Positive
    protected void configureScrollBarColors();

    @Positive
    public Dimension getPreferredSize(JComponent c);

    @Positive
    protected JButton createDecreaseButton(int orientation);

    @Positive
    protected JButton createIncreaseButton(int orientation);

    @Positive
    protected void paintTrack(Graphics g, JComponent c, Rectangle trackBounds);

    @Positive
    protected void paintThumb(Graphics g, JComponent c, Rectangle thumbBounds);

    @Positive
    protected Dimension getMinimumThumbSize();

    @Positive
    protected void setThumbBounds(int x, int y, int width, int height);

    @Positive
    class ScrollBarListener extends BasicScrollBarUI.PropertyChangeHandler {

    @Positive
        public void propertyChange(PropertyChangeEvent e);

    @Positive
        public void handlePropertyChange(Object newValue);

    @Positive
        protected void toFlush();

    @Positive
        protected void toFreeStanding();
    @Positive
    }
    @Positive
}
