/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.applet.Applet;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.beans.*;
    @Positive
import java.security.AccessController;
    @Positive
import javax.accessibility.*;
    @Positive
import javax.swing.plaf.RootPaneUI;
    @Positive
import java.util.Vector;
    @Positive
import java.io.Serializable;
    @Positive
import javax.swing.border.*;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.security.action.GetBooleanAction;
    @Positive
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
@SuppressWarnings({ "removal", "serial" })
    @Positive
public class JRootPane extends JComponent implements Accessible {

    @Positive
    public static final int NONE;

    @Positive
    public static final int FRAME;

    @Positive
    public static final int PLAIN_DIALOG;

    @Positive
    public static final int INFORMATION_DIALOG;

    @Positive
    public static final int ERROR_DIALOG;

    @Positive
    public static final int COLOR_CHOOSER_DIALOG;

    @Positive
    public static final int FILE_CHOOSER_DIALOG;

    @Positive
    public static final int QUESTION_DIALOG;

    @Positive
    public static final int WARNING_DIALOG;

    @Positive
    @MonotonicNonNull
    @Positive
    protected JMenuBar menuBar;

    @Positive
    protected Container contentPane;

    @Positive
    protected JLayeredPane layeredPane;

    @Positive
    protected Component glassPane;

    @Positive
    @Nullable
    @Positive
    protected JButton defaultButton;

    @Positive
    public JRootPane() {
    @Positive
    }

    @Positive
    public void setDoubleBuffered(boolean aFlag);

    @Positive
    public int getWindowDecorationStyle();

    @Positive
    @BeanProperty(expert = true, visualUpdate = true, enumerationValues = { "JRootPane.NONE", "JRootPane.FRAME", "JRootPane.PLAIN_DIALOG", "JRootPane.INFORMATION_DIALOG", "JRootPane.ERROR_DIALOG", "JRootPane.COLOR_CHOOSER_DIALOG", "JRootPane.FILE_CHOOSER_DIALOG", "JRootPane.QUESTION_DIALOG", "JRootPane.WARNING_DIALOG" }, description = "Identifies the type of Window decorations to provide")
    @Positive
    public void setWindowDecorationStyle(int windowDecorationStyle);

    @Positive
    public RootPaneUI getUI();

    @Positive
    @BeanProperty(expert = true, hidden = true, visualUpdate = true, description = "The UI object that implements the Component's LookAndFeel.")
    @Positive
    public void setUI(RootPaneUI ui);

    @Positive
    public void updateUI();

    @Positive
    public String getUIClassID();

    @Positive
    protected JLayeredPane createLayeredPane();

    @Positive
    protected Container createContentPane();

    @Positive
    protected Component createGlassPane();

    @Positive
    protected LayoutManager createRootLayout();

    @Positive
    public void setJMenuBar(@Nullable JMenuBar menu);

    @Positive
    @Deprecated
    @Positive
    public void setMenuBar(@Nullable JMenuBar menu);

    @Positive
    @Nullable
    @Positive
    public JMenuBar getJMenuBar();

    @Positive
    @Deprecated
    @Positive
    @Nullable
    @Positive
    public JMenuBar getMenuBar();

    @Positive
    public void setContentPane(Container content);

    @Positive
    public Container getContentPane();

    @Positive
    public void setLayeredPane(JLayeredPane layered);

    @Positive
    public JLayeredPane getLayeredPane();

    @Positive
    public void setGlassPane(Component glass);

    @Positive
    public Component getGlassPane();

    @Positive
    @Override
    @Positive
    public boolean isValidateRoot();

    @Positive
    public boolean isOptimizedDrawingEnabled();

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    @BeanProperty(description = "The button activated by default in this root pane")
    @Positive
    public void setDefaultButton(@Nullable JButton defaultButton);

    @Positive
    @Nullable
    @Positive
    public JButton getDefaultButton();

    @Positive
    final void setUseTrueDoubleBuffering(boolean useTrueDoubleBuffering);

    @Positive
    final boolean getUseTrueDoubleBuffering();

    @Positive
    final void disableTrueDoubleBuffering();

    @Positive
    protected void addImpl(Component comp, @Nullable Object constraints, int index);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class RootLayout implements LayoutManager2, Serializable {

    @Positive
        protected RootLayout() {
    @Positive
        }

    @Positive
        public Dimension preferredLayoutSize(Container parent);

    @Positive
        public Dimension minimumLayoutSize(Container parent);

    @Positive
        public Dimension maximumLayoutSize(Container target);

    @Positive
        public void layoutContainer(Container parent);

    @Positive
        public void addLayoutComponent(String name, Component comp);

    @Positive
        public void removeLayoutComponent(Component comp);

    @Positive
        public void addLayoutComponent(Component comp, @Nullable Object constraints);

    @Positive
        public float getLayoutAlignmentX(Container target);

    @Positive
        public float getLayoutAlignmentY(Container target);

    @Positive
        public void invalidateLayout(Container target);
    @Positive
    }

    @Positive
    protected String paramString();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJRootPane extends AccessibleJComponent {

    @Positive
        protected AccessibleJRootPane() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
