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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.AWTEvent;
    @Positive
import java.awt.BorderLayout;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Frame;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.GraphicsConfiguration;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Image;
    @Positive
import java.awt.LayoutManager;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.BeanProperty;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
@JavaBean(defaultProperty = "JMenuBar", description = "A toplevel window which can be minimized to an icon.")
    @Positive
@SwingContainer(delegate = "getContentPane")
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JFrame extends Frame implements WindowConstants, Accessible, RootPaneContainer, TransferHandler.HasGetTransferHandler {

    @Positive
    protected JRootPane rootPane;

    @Positive
    protected boolean rootPaneCheckingEnabled;

    @Positive
    public JFrame() throws HeadlessException {
    @Positive
    }

    @Positive
    public JFrame(GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    public JFrame(@Nullable String title) throws HeadlessException {
    @Positive
    }

    @Positive
    public JFrame(String title, GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    protected void frameInit();

    @Positive
    protected JRootPane createRootPane();

    @Positive
    protected void processWindowEvent(final WindowEvent e);

    @Positive
    @BeanProperty(preferred = true, enumerationValues = { "WindowConstants.DO_NOTHING_ON_CLOSE", "WindowConstants.HIDE_ON_CLOSE", "WindowConstants.DISPOSE_ON_CLOSE", "WindowConstants.EXIT_ON_CLOSE" }, description = "The frame's default close operation.")
    @Positive
    public void setDefaultCloseOperation(int operation);

    @Positive
    public int getDefaultCloseOperation();

    @Positive
    @BeanProperty(hidden = true, description = "Mechanism for transfer of data into the component")
    @Positive
    public void setTransferHandler(@Nullable TransferHandler newHandler);

    @Positive
    @Nullable
    @Positive
    public TransferHandler getTransferHandler();

    @Positive
    public void update(Graphics g);

    @Positive
    @BeanProperty(bound = false, hidden = true, description = "The menubar for accessing pulldown menus from this frame.")
    @Positive
    public void setJMenuBar(@Nullable final JMenuBar menubar);

    @Positive
    @Nullable
    @Positive
    public JMenuBar getJMenuBar();

    @Positive
    protected boolean isRootPaneCheckingEnabled();

    @Positive
    @BeanProperty(hidden = true, description = "Whether the add and setLayout methods are forwarded")
    @Positive
    protected void setRootPaneCheckingEnabled(boolean enabled);

    @Positive
    protected void addImpl(Component comp, Object constraints, int index);

    @Positive
    public void remove(Component comp);

    @Positive
    public void setLayout(@Nullable LayoutManager manager);

    @Positive
    @BeanProperty(bound = false, hidden = true, description = "the RootPane object for this frame.")
    @Positive
    public JRootPane getRootPane();

    @Positive
    protected void setRootPane(JRootPane root);

    @Positive
    public void setIconImage(@Nullable Image image);

    @Positive
    public Container getContentPane();

    @Positive
    @BeanProperty(bound = false, hidden = true, description = "The client area of the frame where child components are normally inserted.")
    @Positive
    public void setContentPane(Container contentPane);

    @Positive
    public JLayeredPane getLayeredPane();

    @Positive
    @BeanProperty(bound = false, hidden = true, description = "The pane that holds the various frame layers.")
    @Positive
    public void setLayeredPane(JLayeredPane layeredPane);

    @Positive
    public Component getGlassPane();

    @Positive
    @BeanProperty(bound = false, hidden = true, description = "A transparent pane used for menu rendering.")
    @Positive
    public void setGlassPane(Component glassPane);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Graphics getGraphics();

    @Positive
    public void repaint(long time, int x, int y, int width, int height);

    @Positive
    public static void setDefaultLookAndFeelDecorated(boolean defaultLookAndFeelDecorated);

    @Positive
    public static boolean isDefaultLookAndFeelDecorated();

    @Positive
    protected String paramString();

    @Positive
    protected AccessibleContext accessibleContext;

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleJFrame extends AccessibleAWTFrame {

    @Positive
        protected AccessibleJFrame() {
    @Positive
        }

    @Positive
        public String getAccessibleName();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
