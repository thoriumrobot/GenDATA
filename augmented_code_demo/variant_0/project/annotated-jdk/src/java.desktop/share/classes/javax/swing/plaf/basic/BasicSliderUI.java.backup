/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.IllegalComponentStateException;
import java.awt.Insets;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.Dictionary;
import java.util.Enumeration;
import javax.swing.AbstractAction;
import javax.swing.BoundedRangeModel;
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.InputMap;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JSlider;
import javax.swing.LookAndFeel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.UIManager;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.MouseInputAdapter;
import javax.swing.plaf.ComponentUI;
import javax.swing.plaf.InsetsUIResource;
import javax.swing.plaf.SliderUI;
import sun.swing.DefaultLookup;
import sun.swing.SwingUtilities2;
import sun.swing.UIAction;

@AnnotatedFor({ "interning" })
public class BasicSliderUI extends SliderUI {

    public static final int POSITIVE_SCROLL;

    public static final int NEGATIVE_SCROLL;

    public static final int MIN_SCROLL;

    public static final int MAX_SCROLL;

    protected Timer scrollTimer;

    protected JSlider slider;

    protected Insets focusInsets;

    protected Insets insetCache;

    protected boolean leftToRightCache;

    protected Rectangle focusRect;

    protected Rectangle contentRect;

    protected Rectangle labelRect;

    protected Rectangle tickRect;

    protected Rectangle trackRect;

    protected Rectangle thumbRect;

    protected int trackBuffer;

    protected TrackListener trackListener;

    protected ChangeListener changeListener;

    protected ComponentListener componentListener;

    protected FocusListener focusListener;

    protected ScrollListener scrollListener;

    protected PropertyChangeListener propertyChangeListener;

    public BasicSliderUI() {
    }

    protected Color getShadowColor();

    protected Color getHighlightColor();

    protected Color getFocusColor();

    protected boolean isDragging();

    public static ComponentUI createUI(JComponent b);

    public BasicSliderUI(JSlider b) {
    }

    public void installUI(JComponent c);

    public void uninstallUI(JComponent c);

    protected void installDefaults(JSlider slider);

    protected void uninstallDefaults(JSlider slider);

    protected TrackListener createTrackListener(JSlider slider);

    protected ChangeListener createChangeListener(JSlider slider);

    protected ComponentListener createComponentListener(JSlider slider);

    protected FocusListener createFocusListener(JSlider slider);

    protected ScrollListener createScrollListener(JSlider slider);

    protected PropertyChangeListener createPropertyChangeListener(JSlider slider);

    protected void installListeners(JSlider slider);

    protected void uninstallListeners(JSlider slider);

    protected void installKeyboardActions(JSlider slider);

    InputMap getInputMap(int condition, JSlider slider);

    static void loadActionMap(LazyActionMap map);

    protected void uninstallKeyboardActions(JSlider slider);

    public int getBaseline(JComponent c, int width, int height);

    public Component.BaselineResizeBehavior getBaselineResizeBehavior(JComponent c);

    protected boolean labelsHaveSameBaselines();

    public Dimension getPreferredHorizontalSize();

    public Dimension getPreferredVerticalSize();

    public Dimension getMinimumHorizontalSize();

    public Dimension getMinimumVerticalSize();

    public Dimension getPreferredSize(JComponent c);

    public Dimension getMinimumSize(JComponent c);

    public Dimension getMaximumSize(JComponent c);

    protected void calculateGeometry();

    protected void calculateFocusRect();

    protected void calculateThumbSize();

    protected void calculateContentRect();

    protected void calculateThumbLocation();

    protected void calculateTrackBuffer();

    protected void calculateTrackRect();

    protected int getTickLength();

    protected void calculateTickRect();

    protected void calculateLabelRect();

    protected Dimension getThumbSize();

    public class PropertyChangeHandler implements PropertyChangeListener {

        public PropertyChangeHandler() {
        }

        public void propertyChange(PropertyChangeEvent e);
    }

    protected int getWidthOfWidestLabel();

    protected int getHeightOfTallestLabel();

    protected int getWidthOfHighValueLabel();

    protected int getWidthOfLowValueLabel();

    protected int getHeightOfHighValueLabel();

    protected int getHeightOfLowValueLabel();

    protected boolean drawInverted();

    protected Integer getHighestValue();

    protected Integer getLowestValue();

    protected Component getLowestValueLabel();

    protected Component getHighestValueLabel();

    public void paint(Graphics g, JComponent c);

    protected void recalculateIfInsetsChanged();

    protected void recalculateIfOrientationChanged();

    public void paintFocus(Graphics g);

    public void paintTrack(Graphics g);

    public void paintTicks(Graphics g);

    protected void paintMinorTickForHorizSlider(Graphics g, Rectangle tickBounds, int x);

    protected void paintMajorTickForHorizSlider(Graphics g, Rectangle tickBounds, int x);

    protected void paintMinorTickForVertSlider(Graphics g, Rectangle tickBounds, int y);

    protected void paintMajorTickForVertSlider(Graphics g, Rectangle tickBounds, int y);

    public void paintLabels(Graphics g);

    protected void paintHorizontalLabel(Graphics g, int value, Component label);

    protected void paintVerticalLabel(Graphics g, int value, Component label);

    public void paintThumb(Graphics g);

    public void setThumbLocation(int x, int y);

    public void scrollByBlock(int direction);

    public void scrollByUnit(int direction);

    protected void scrollDueToClickInTrack(int dir);

    protected int xPositionForValue(int value);

    protected int yPositionForValue(int value);

    protected int yPositionForValue(int value, int trackY, int trackHeight);

    public int valueForYPosition(int yPos);

    public int valueForXPosition(int xPos);

    private class Handler implements ChangeListener, ComponentListener, FocusListener, PropertyChangeListener {

        public void stateChanged(ChangeEvent e);

        public void componentHidden(ComponentEvent e);

        public void componentMoved(ComponentEvent e);

        public void componentResized(ComponentEvent e);

        public void componentShown(ComponentEvent e);

        public void focusGained(FocusEvent e);

        public void focusLost(FocusEvent e);

        public void propertyChange(PropertyChangeEvent e);
    }

    public class ChangeHandler implements ChangeListener {

        public ChangeHandler() {
        }

        public void stateChanged(ChangeEvent e);
    }

    public class TrackListener extends MouseInputAdapter {

        protected transient int offset;

        protected transient int currentMouseX;

        protected transient int currentMouseY;

        public TrackListener() {
        }

        public void mouseReleased(MouseEvent e);

        public void mousePressed(MouseEvent e);

        public boolean shouldScroll(int direction);

        public void mouseDragged(MouseEvent e);

        public void mouseMoved(MouseEvent e);
    }

    public class ScrollListener implements ActionListener {

        public ScrollListener() {
        }

        public ScrollListener(int dir, boolean block) {
        }

        public void setDirection(int direction);

        public void setScrollByBlock(boolean block);

        public void actionPerformed(ActionEvent e);
    }

    public class ComponentHandler extends ComponentAdapter {

        public ComponentHandler() {
        }

        public void componentResized(ComponentEvent e);
    }

    public class FocusHandler implements FocusListener {

        public FocusHandler() {
        }

        public void focusGained(FocusEvent e);

        public void focusLost(FocusEvent e);
    }

    @SuppressWarnings("serial")
    public class ActionScroller extends AbstractAction {

        public ActionScroller(JSlider slider, int dir, boolean block) {
        }

        public void actionPerformed(ActionEvent e);

        public boolean isEnabled();
    }

    @SuppressWarnings("serial")
    static class SharedActionScroller extends AbstractAction {

        public SharedActionScroller(int dir, boolean block) {
        }

        public void actionPerformed(ActionEvent evt);
    }

    private static class Actions extends UIAction {

        public static final String POSITIVE_UNIT_INCREMENT;

        public static final String POSITIVE_BLOCK_INCREMENT;

        public static final String NEGATIVE_UNIT_INCREMENT;

        public static final String NEGATIVE_BLOCK_INCREMENT;

        @Interned
        public static final String MIN_SCROLL_INCREMENT;

        @Interned
        public static final String MAX_SCROLL_INCREMENT;

        public Actions(String name) {
        }

        public void actionPerformed(ActionEvent evt);
    }
}
