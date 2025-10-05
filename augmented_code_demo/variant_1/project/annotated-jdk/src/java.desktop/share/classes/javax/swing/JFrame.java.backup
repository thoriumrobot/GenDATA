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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.AWTEvent;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.GraphicsConfiguration;
import java.awt.HeadlessException;
import java.awt.Image;
import java.awt.LayoutManager;
import java.awt.event.WindowEvent;
import java.beans.JavaBean;
import java.beans.BeanProperty;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;

@AnnotatedFor({ "nullness" })
@JavaBean(defaultProperty = "JMenuBar", description = "A toplevel window which can be minimized to an icon.")
@SwingContainer(delegate = "getContentPane")
@SuppressWarnings("serial")
public class JFrame extends Frame implements WindowConstants, Accessible, RootPaneContainer, TransferHandler.HasGetTransferHandler {

    protected JRootPane rootPane;

    protected boolean rootPaneCheckingEnabled;

    public JFrame() throws HeadlessException {
    }

    public JFrame(GraphicsConfiguration gc) {
    }

    public JFrame(@Nullable String title) throws HeadlessException {
    }

    public JFrame(String title, GraphicsConfiguration gc) {
    }

    protected void frameInit();

    protected JRootPane createRootPane();

    protected void processWindowEvent(final WindowEvent e);

    @BeanProperty(preferred = true, enumerationValues = { "WindowConstants.DO_NOTHING_ON_CLOSE", "WindowConstants.HIDE_ON_CLOSE", "WindowConstants.DISPOSE_ON_CLOSE", "WindowConstants.EXIT_ON_CLOSE" }, description = "The frame's default close operation.")
    public void setDefaultCloseOperation(int operation);

    public int getDefaultCloseOperation();

    @BeanProperty(hidden = true, description = "Mechanism for transfer of data into the component")
    public void setTransferHandler(@Nullable TransferHandler newHandler);

    @Nullable
    public TransferHandler getTransferHandler();

    public void update(Graphics g);

    @BeanProperty(bound = false, hidden = true, description = "The menubar for accessing pulldown menus from this frame.")
    public void setJMenuBar(@Nullable final JMenuBar menubar);

    @Nullable
    public JMenuBar getJMenuBar();

    protected boolean isRootPaneCheckingEnabled();

    @BeanProperty(hidden = true, description = "Whether the add and setLayout methods are forwarded")
    protected void setRootPaneCheckingEnabled(boolean enabled);

    protected void addImpl(Component comp, Object constraints, int index);

    public void remove(Component comp);

    public void setLayout(@Nullable LayoutManager manager);

    @BeanProperty(bound = false, hidden = true, description = "the RootPane object for this frame.")
    public JRootPane getRootPane();

    protected void setRootPane(JRootPane root);

    public void setIconImage(@Nullable Image image);

    public Container getContentPane();

    @BeanProperty(bound = false, hidden = true, description = "The client area of the frame where child components are normally inserted.")
    public void setContentPane(Container contentPane);

    public JLayeredPane getLayeredPane();

    @BeanProperty(bound = false, hidden = true, description = "The pane that holds the various frame layers.")
    public void setLayeredPane(JLayeredPane layeredPane);

    public Component getGlassPane();

    @BeanProperty(bound = false, hidden = true, description = "A transparent pane used for menu rendering.")
    public void setGlassPane(Component glassPane);

    @BeanProperty(bound = false)
    public Graphics getGraphics();

    public void repaint(long time, int x, int y, int width, int height);

    public static void setDefaultLookAndFeelDecorated(boolean defaultLookAndFeelDecorated);

    public static boolean isDefaultLookAndFeelDecorated();

    protected String paramString();

    protected AccessibleContext accessibleContext;

    public AccessibleContext getAccessibleContext();

    protected class AccessibleJFrame extends AccessibleAWTFrame {

        protected AccessibleJFrame() {
        }

        public String getAccessibleName();

        public AccessibleStateSet getAccessibleStateSet();
    }
}
