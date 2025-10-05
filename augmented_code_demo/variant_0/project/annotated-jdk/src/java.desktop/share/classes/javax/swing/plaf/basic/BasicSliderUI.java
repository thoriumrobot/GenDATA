/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.FontMetrics;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.IllegalComponentStateException;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Polygon;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.ComponentAdapter;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.ComponentListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.FocusListener;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.util.Dictionary;
    @Positive
import java.util.Enumeration;
    @Positive
import javax.swing.AbstractAction;
    @Positive
import javax.swing.BoundedRangeModel;
    @Positive
import javax.swing.Icon;
    @Positive
import javax.swing.ImageIcon;
    @Positive
import javax.swing.InputMap;
    @Positive
import javax.swing.JComponent;
    @Positive
import javax.swing.JLabel;
    @Positive
import javax.swing.JSlider;
    @Positive
import javax.swing.LookAndFeel;
    @Positive
import javax.swing.SwingUtilities;
    @Positive
import javax.swing.Timer;
    @Positive
import javax.swing.UIManager;
    @Positive
import javax.swing.event.ChangeEvent;
    @Positive
import javax.swing.event.ChangeListener;
    @Positive
import javax.swing.event.MouseInputAdapter;
    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import javax.swing.plaf.InsetsUIResource;
    @Positive
import javax.swing.plaf.SliderUI;
    @Positive
import sun.swing.DefaultLookup;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import sun.swing.UIAction;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class BasicSliderUI extends SliderUI {

    @Positive
    public static final int POSITIVE_SCROLL;

    @Positive
    public static final int NEGATIVE_SCROLL;

    @Positive
    public static final int MIN_SCROLL;

    @Positive
    public static final int MAX_SCROLL;

    @Positive
    protected Timer scrollTimer;

    @Positive
    protected JSlider slider;

    @Positive
    protected Insets focusInsets;

    @Positive
    protected Insets insetCache;

    @Positive
    protected boolean leftToRightCache;

    @Positive
    protected Rectangle focusRect;

    @Positive
    protected Rectangle contentRect;

    @Positive
    protected Rectangle labelRect;

    @Positive
    protected Rectangle tickRect;

    @Positive
    protected Rectangle trackRect;

    @Positive
    protected Rectangle thumbRect;

    @Positive
    protected int trackBuffer;

    @Positive
    protected TrackListener trackListener;

    @Positive
    protected ChangeListener changeListener;

    @Positive
    protected ComponentListener componentListener;

    @Positive
    protected FocusListener focusListener;

    @Positive
    protected ScrollListener scrollListener;

    @Positive
    protected PropertyChangeListener propertyChangeListener;

    @Positive
    public BasicSliderUI() {
    @Positive
    }

    @Positive
    protected Color getShadowColor();

    @Positive
    protected Color getHighlightColor();

    @Positive
    protected Color getFocusColor();

    @Positive
    protected boolean isDragging();

    @Positive
    public static ComponentUI createUI(JComponent b);

    @Positive
    public BasicSliderUI(JSlider b) {
    @Positive
    }

    @Positive
    public void installUI(JComponent c);

    @Positive
    public void uninstallUI(JComponent c);

    @Positive
    protected void installDefaults(JSlider slider);

    @Positive
    protected void uninstallDefaults(JSlider slider);

    @Positive
    protected TrackListener createTrackListener(JSlider slider);

    @Positive
    protected ChangeListener createChangeListener(JSlider slider);

    @Positive
    protected ComponentListener createComponentListener(JSlider slider);

    @Positive
    protected FocusListener createFocusListener(JSlider slider);

    @Positive
    protected ScrollListener createScrollListener(JSlider slider);

    @Positive
    protected PropertyChangeListener createPropertyChangeListener(JSlider slider);

    @Positive
    protected void installListeners(JSlider slider);

    @Positive
    protected void uninstallListeners(JSlider slider);

    @Positive
    protected void installKeyboardActions(JSlider slider);

    @Positive
    InputMap getInputMap(int condition, JSlider slider);

    @Positive
    static void loadActionMap(LazyActionMap map);

    @Positive
    protected void uninstallKeyboardActions(JSlider slider);

    @Positive
    public int getBaseline(JComponent c, int width, int height);

    @Positive
    public Component.BaselineResizeBehavior getBaselineResizeBehavior(JComponent c);

    @Positive
    protected boolean labelsHaveSameBaselines();

    @Positive
    public Dimension getPreferredHorizontalSize();

    @Positive
    public Dimension getPreferredVerticalSize();

    @Positive
    public Dimension getMinimumHorizontalSize();

    @Positive
    public Dimension getMinimumVerticalSize();

    @Positive
    public Dimension getPreferredSize(JComponent c);

    @Positive
    public Dimension getMinimumSize(JComponent c);

    @Positive
    public Dimension getMaximumSize(JComponent c);

    @Positive
    protected void calculateGeometry();

    @Positive
    protected void calculateFocusRect();

    @Positive
    protected void calculateThumbSize();

    @Positive
    protected void calculateContentRect();

    @Positive
    protected void calculateThumbLocation();

    @Positive
    protected void calculateTrackBuffer();

    @Positive
    protected void calculateTrackRect();

    @Positive
    protected int getTickLength();

    @Positive
    protected void calculateTickRect();

    @Positive
    protected void calculateLabelRect();

    @Positive
    protected Dimension getThumbSize();

    @Positive
    public class PropertyChangeHandler implements PropertyChangeListener {

    @Positive
        public PropertyChangeHandler() {
    @Positive
        }

    @Positive
        public void propertyChange(PropertyChangeEvent e);
    @Positive
    }

    @Positive
    protected int getWidthOfWidestLabel();

    @Positive
    protected int getHeightOfTallestLabel();

    @Positive
    protected int getWidthOfHighValueLabel();

    @Positive
    protected int getWidthOfLowValueLabel();

    @Positive
    protected int getHeightOfHighValueLabel();

    @Positive
    protected int getHeightOfLowValueLabel();

    @Positive
    protected boolean drawInverted();

    @Positive
    protected Integer getHighestValue();

    @Positive
    protected Integer getLowestValue();

    @Positive
    protected Component getLowestValueLabel();

    @Positive
    protected Component getHighestValueLabel();

    @Positive
    public void paint(Graphics g, JComponent c);

    @Positive
    protected void recalculateIfInsetsChanged();

    @Positive
    protected void recalculateIfOrientationChanged();

    @Positive
    public void paintFocus(Graphics g);

    @Positive
    public void paintTrack(Graphics g);

    @Positive
    public void paintTicks(Graphics g);

    @Positive
    protected void paintMinorTickForHorizSlider(Graphics g, Rectangle tickBounds, int x);

    @Positive
    protected void paintMajorTickForHorizSlider(Graphics g, Rectangle tickBounds, int x);

    @Positive
    protected void paintMinorTickForVertSlider(Graphics g, Rectangle tickBounds, int y);

    @Positive
    protected void paintMajorTickForVertSlider(Graphics g, Rectangle tickBounds, int y);

    @Positive
    public void paintLabels(Graphics g);

    @Positive
    protected void paintHorizontalLabel(Graphics g, int value, Component label);

    @Positive
    protected void paintVerticalLabel(Graphics g, int value, Component label);

    @Positive
    public void paintThumb(Graphics g);

    @Positive
    public void setThumbLocation(int x, int y);

    @Positive
    public void scrollByBlock(int direction);

    @Positive
    public void scrollByUnit(int direction);

    @Positive
    protected void scrollDueToClickInTrack(int dir);

    @Positive
    protected int xPositionForValue(int value);

    @Positive
    protected int yPositionForValue(int value);

    @Positive
    protected int yPositionForValue(int value, int trackY, int trackHeight);

    @Positive
    public int valueForYPosition(int yPos);

    @Positive
    public int valueForXPosition(int xPos);

    @Positive
    private class Handler implements ChangeListener, ComponentListener, FocusListener, PropertyChangeListener {

    @Positive
        public void stateChanged(ChangeEvent e);

    @Positive
        public void componentHidden(ComponentEvent e);

    @Positive
        public void componentMoved(ComponentEvent e);

    @Positive
        public void componentResized(ComponentEvent e);

    @Positive
        public void componentShown(ComponentEvent e);

    @Positive
        public void focusGained(FocusEvent e);

    @Positive
        public void focusLost(FocusEvent e);

    @Positive
        public void propertyChange(PropertyChangeEvent e);
    @Positive
    }

    @Positive
    public class ChangeHandler implements ChangeListener {

    @Positive
        public ChangeHandler() {
    @Positive
        }

    @Positive
        public void stateChanged(ChangeEvent e);
    @Positive
    }

    @Positive
    public class TrackListener extends MouseInputAdapter {

    @Positive
        protected transient int offset;

    @Positive
        protected transient int currentMouseX;

    @Positive
        protected transient int currentMouseY;

    @Positive
        public TrackListener() {
    @Positive
        }

    @Positive
        public void mouseReleased(MouseEvent e);

    @Positive
        public void mousePressed(MouseEvent e);

    @Positive
        public boolean shouldScroll(int direction);

    @Positive
        public void mouseDragged(MouseEvent e);

    @Positive
        public void mouseMoved(MouseEvent e);
    @Positive
    }

    @Positive
    public class ScrollListener implements ActionListener {

    @Positive
        public ScrollListener() {
    @Positive
        }

    @Positive
        public ScrollListener(int dir, boolean block) {
    @Positive
        }

    @Positive
        public void setDirection(int direction);

    @Positive
        public void setScrollByBlock(boolean block);

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    public class ComponentHandler extends ComponentAdapter {

    @Positive
        public ComponentHandler() {
    @Positive
        }

    @Positive
        public void componentResized(ComponentEvent e);
    @Positive
    }

    @Positive
    public class FocusHandler implements FocusListener {

    @Positive
        public FocusHandler() {
    @Positive
        }

    @Positive
        public void focusGained(FocusEvent e);

    @Positive
        public void focusLost(FocusEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public class ActionScroller extends AbstractAction {

    @Positive
        public ActionScroller(JSlider slider, int dir, boolean block) {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);

    @Positive
        public boolean isEnabled();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SharedActionScroller extends AbstractAction {

    @Positive
        public SharedActionScroller(int dir, boolean block) {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent evt);
    @Positive
    }

    @Positive
    private static class Actions extends UIAction {

    @Positive
        public static final String POSITIVE_UNIT_INCREMENT;

    @Positive
        public static final String POSITIVE_BLOCK_INCREMENT;

    @Positive
        public static final String NEGATIVE_UNIT_INCREMENT;

    @Positive
        public static final String NEGATIVE_BLOCK_INCREMENT;

    @Positive
        @Interned
    @Positive
        public static final String MIN_SCROLL_INCREMENT;

    @Positive
        @Interned
    @Positive
        public static final String MAX_SCROLL_INCREMENT;

    @Positive
        public Actions(String name) {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent evt);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
