/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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

import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;
import java.beans.*;
import java.security.AccessController;
import javax.accessibility.*;
import javax.swing.plaf.RootPaneUI;
import java.util.Vector;
import java.io.Serializable;
import javax.swing.border.*;
import sun.awt.AWTAccessor;
import sun.security.action.GetBooleanAction;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;

@AnnotatedFor({ "nullness" })
@SuppressWarnings({ "removal", "serial" })
public class JRootPane extends JComponent implements Accessible {

    public static final int NONE;

    public static final int FRAME;

    public static final int PLAIN_DIALOG;

    public static final int INFORMATION_DIALOG;

    public static final int ERROR_DIALOG;

    public static final int COLOR_CHOOSER_DIALOG;

    public static final int FILE_CHOOSER_DIALOG;

    public static final int QUESTION_DIALOG;

    public static final int WARNING_DIALOG;

    @MonotonicNonNull
    protected JMenuBar menuBar;

    protected Container contentPane;

    protected JLayeredPane layeredPane;

    protected Component glassPane;

    @Nullable
    protected JButton defaultButton;

    public JRootPane() {
    }

    public void setDoubleBuffered(boolean aFlag);

    public int getWindowDecorationStyle();

    @BeanProperty(expert = true, visualUpdate = true, enumerationValues = { "JRootPane.NONE", "JRootPane.FRAME", "JRootPane.PLAIN_DIALOG", "JRootPane.INFORMATION_DIALOG", "JRootPane.ERROR_DIALOG", "JRootPane.COLOR_CHOOSER_DIALOG", "JRootPane.FILE_CHOOSER_DIALOG", "JRootPane.QUESTION_DIALOG", "JRootPane.WARNING_DIALOG" }, description = "Identifies the type of Window decorations to provide")
    public void setWindowDecorationStyle(int windowDecorationStyle);

    public RootPaneUI getUI();

    @BeanProperty(expert = true, hidden = true, visualUpdate = true, description = "The UI object that implements the Component's LookAndFeel.")
    public void setUI(RootPaneUI ui);

    public void updateUI();

    public String getUIClassID();

    protected JLayeredPane createLayeredPane();

    protected Container createContentPane();

    protected Component createGlassPane();

    protected LayoutManager createRootLayout();

    public void setJMenuBar(@Nullable JMenuBar menu);

    @Deprecated
    public void setMenuBar(@Nullable JMenuBar menu);

    @Nullable
    public JMenuBar getJMenuBar();

    @Deprecated
    @Nullable
    public JMenuBar getMenuBar();

    public void setContentPane(Container content);

    public Container getContentPane();

    public void setLayeredPane(JLayeredPane layered);

    public JLayeredPane getLayeredPane();

    public void setGlassPane(Component glass);

    public Component getGlassPane();

    @Override
    public boolean isValidateRoot();

    public boolean isOptimizedDrawingEnabled();

    public void addNotify();

    public void removeNotify();

    @BeanProperty(description = "The button activated by default in this root pane")
    public void setDefaultButton(@Nullable JButton defaultButton);

    @Nullable
    public JButton getDefaultButton();

    final void setUseTrueDoubleBuffering(boolean useTrueDoubleBuffering);

    final boolean getUseTrueDoubleBuffering();

    final void disableTrueDoubleBuffering();

    protected void addImpl(Component comp, @Nullable Object constraints, int index);

    @SuppressWarnings("serial")
    protected class RootLayout implements LayoutManager2, Serializable {

        protected RootLayout() {
        }

        public Dimension preferredLayoutSize(Container parent);

        public Dimension minimumLayoutSize(Container parent);

        public Dimension maximumLayoutSize(Container target);

        public void layoutContainer(Container parent);

        public void addLayoutComponent(String name, Component comp);

        public void removeLayoutComponent(Component comp);

        public void addLayoutComponent(Component comp, @Nullable Object constraints);

        public float getLayoutAlignmentX(Container target);

        public float getLayoutAlignmentY(Container target);

        public void invalidateLayout(Container target);
    }

    protected String paramString();

    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJRootPane extends AccessibleJComponent {

        protected AccessibleJRootPane() {
        }

        public AccessibleRole getAccessibleRole();

        public int getAccessibleChildrenCount();

        public Accessible getAccessibleChild(int i);
    }
}
