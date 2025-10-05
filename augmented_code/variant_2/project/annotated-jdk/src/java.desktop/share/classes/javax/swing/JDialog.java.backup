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
package javax.swing;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.*;
import java.awt.event.*;
import java.beans.JavaBean;
import java.beans.BeanProperty;
import javax.accessibility.*;

@AnnotatedFor({ "nullness" })
@JavaBean(defaultProperty = "JMenuBar", description = "A toplevel window for creating dialog boxes.")
@SwingContainer(delegate = "getContentPane")
@SuppressWarnings("serial")
public class JDialog extends Dialog implements WindowConstants, Accessible, RootPaneContainer, TransferHandler.HasGetTransferHandler {

    protected JRootPane rootPane;

    protected boolean rootPaneCheckingEnabled;

    public JDialog() {
    }

    public JDialog(Frame owner) {
    }

    public JDialog(Frame owner, boolean modal) {
    }

    public JDialog(Frame owner, @Nullable String title) {
    }

    public JDialog(@Nullable Frame owner, @Nullable String title, boolean modal) {
    }

    public JDialog(@Nullable Frame owner, @Nullable String title, boolean modal, GraphicsConfiguration gc) {
    }

    public JDialog(@Nullable Dialog owner) {
    }

    public JDialog(@Nullable Dialog owner, boolean modal) {
    }

    public JDialog(@Nullable Dialog owner, @Nullable String title) {
    }

    public JDialog(@Nullable Dialog owner, @Nullable String title, boolean modal) {
    }

    public JDialog(@Nullable Dialog owner, @Nullable String title, boolean modal, GraphicsConfiguration gc) {
    }

    public JDialog(@Nullable Window owner) {
    }

    public JDialog(@Nullable Window owner, @Nullable ModalityType modalityType) {
    }

    public JDialog(@Nullable Window owner, @Nullable String title) {
    }

    public JDialog(@Nullable Window owner, @Nullable String title, Dialog.ModalityType modalityType) {
    }

    public JDialog(@Nullable Window owner, @Nullable String title, Dialog.ModalityType modalityType, @Nullable GraphicsConfiguration gc) {
    }

    protected void dialogInit();

    protected JRootPane createRootPane();

    protected void processWindowEvent(WindowEvent e);

    @BeanProperty(preferred = true, enumerationValues = { "WindowConstants.DO_NOTHING_ON_CLOSE", "WindowConstants.HIDE_ON_CLOSE", "WindowConstants.DISPOSE_ON_CLOSE" }, description = "The dialog's default close operation.")
    public void setDefaultCloseOperation(int operation);

    public int getDefaultCloseOperation();

    @BeanProperty(hidden = true, description = "Mechanism for transfer of data into the component")
    public void setTransferHandler(@Nullable TransferHandler newHandler);

    @Nullable
    public TransferHandler getTransferHandler();

    public void update(Graphics g);

    @BeanProperty(bound = false, hidden = true, description = "The menubar for accessing pulldown menus from this dialog.")
    public void setJMenuBar(final JMenuBar menu);

    public JMenuBar getJMenuBar();

    protected boolean isRootPaneCheckingEnabled();

    @BeanProperty(hidden = true, description = "Whether the add and setLayout methods are forwarded")
    protected void setRootPaneCheckingEnabled(boolean enabled);

    protected void addImpl(Component comp, @Nullable Object constraints, int index);

    public void remove(Component comp);

    public void setLayout(@Nullable LayoutManager manager);

    @BeanProperty(bound = false, hidden = true, description = "the RootPane object for this dialog.")
    public JRootPane getRootPane();

    protected void setRootPane(JRootPane root);

    public Container getContentPane();

    @BeanProperty(bound = false, hidden = true, description = "The client area of the dialog where child components are normally inserted.")
    public void setContentPane(Container contentPane);

    public JLayeredPane getLayeredPane();

    @BeanProperty(bound = false, hidden = true, description = "The pane which holds the various dialog layers.")
    public void setLayeredPane(JLayeredPane layeredPane);

    public Component getGlassPane();

    @BeanProperty(bound = false, hidden = true, description = "A transparent pane used for menu rendering.")
    public void setGlassPane(Component glassPane);

    @BeanProperty(bound = false)
    @Nullable
    public Graphics getGraphics();

    public void repaint(long time, int x, int y, int width, int height);

    public static void setDefaultLookAndFeelDecorated(boolean defaultLookAndFeelDecorated);

    public static boolean isDefaultLookAndFeelDecorated();

    protected String paramString();

    protected AccessibleContext accessibleContext;

    public AccessibleContext getAccessibleContext();

    protected class AccessibleJDialog extends AccessibleAWTDialog {

        protected AccessibleJDialog() {
        }

        public String getAccessibleName();

        public AccessibleStateSet getAccessibleStateSet();
    }
}
