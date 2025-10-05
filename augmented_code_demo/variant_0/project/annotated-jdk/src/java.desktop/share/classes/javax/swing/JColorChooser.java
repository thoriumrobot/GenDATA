/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.BorderLayout;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dialog;
    @Positive
import java.awt.FlowLayout;
    @Positive
import java.awt.Frame;
    @Positive
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.ComponentAdapter;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.WindowAdapter;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Locale;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.swing.colorchooser.AbstractColorChooserPanel;
    @Positive
import javax.swing.colorchooser.ColorChooserComponentFactory;
    @Positive
import javax.swing.colorchooser.ColorSelectionModel;
    @Positive
import javax.swing.colorchooser.DefaultColorSelectionModel;
    @Positive
import javax.swing.plaf.ColorChooserUI;
    @Positive
import sun.swing.SwingUtilities2;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(defaultProperty = "UI", description = "A component that supports selecting a Color.")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JColorChooser extends JComponent implements Accessible {

    @Positive
    @Interned
    @Positive
    public static final String SELECTION_MODEL_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String PREVIEW_PANEL_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CHOOSER_PANELS_PROPERTY;

    @Positive
    public static Color showDialog(Component component, String title, Color initialColor) throws HeadlessException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static Color showDialog(Component component, String title, Color initialColor, boolean colorTransparencySelectionEnabled) throws HeadlessException;

    @Positive
    public static JDialog createDialog(Component c, String title, boolean modal, JColorChooser chooserPane, ActionListener okListener, ActionListener cancelListener) throws HeadlessException;

    @Positive
    public JColorChooser() {
    @Positive
    }

    @Positive
    public JColorChooser(Color initialColor) {
    @Positive
    }

    @Positive
    public JColorChooser(ColorSelectionModel model) {
    @Positive
    }

    @Positive
    public ColorChooserUI getUI();

    @Positive
    @BeanProperty(hidden = true, description = "The UI object that implements the color chooser's LookAndFeel.")
    @Positive
    public void setUI(ColorChooserUI ui);

    @Positive
    public void updateUI();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public String getUIClassID();

    @Positive
    public Color getColor();

    @Positive
    @BeanProperty(bound = false, description = "The current color the chooser is to display.")
    @Positive
    public void setColor(Color color);

    @Positive
    public void setColor(int r, int g, int b);

    @Positive
    public void setColor(int c);

    @Positive
    @BeanProperty(bound = false, description = "Determines whether automatic drag handling is enabled.")
    @Positive
    public void setDragEnabled(boolean b);

    @Positive
    public boolean getDragEnabled();

    @Positive
    @BeanProperty(hidden = true, description = "The UI component which displays the current color.")
    @Positive
    public void setPreviewPanel(JComponent preview);

    @Positive
    public JComponent getPreviewPanel();

    @Positive
    public void addChooserPanel(AbstractColorChooserPanel panel);

    @Positive
    public AbstractColorChooserPanel removeChooserPanel(AbstractColorChooserPanel panel);

    @Positive
    @BeanProperty(hidden = true, description = "An array of different chooser types.")
    @Positive
    public void setChooserPanels(AbstractColorChooserPanel[] panels);

    @Positive
    public AbstractColorChooserPanel[] getChooserPanels();

    @Positive
    public ColorSelectionModel getSelectionModel();

    @Positive
    @BeanProperty(hidden = true, description = "The model which contains the currently selected color.")
    @Positive
    public void setSelectionModel(ColorSelectionModel newModel);

    @Positive
    protected String paramString();

    @Positive
    protected AccessibleContext accessibleContext;

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleJColorChooser extends AccessibleJComponent {

    @Positive
        protected AccessibleJColorChooser() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();
    @Positive
    }
    @Positive
}

    @Positive
@SuppressWarnings("serial")
    @Positive
class ColorChooserDialog extends JDialog {

    @Positive
    public ColorChooserDialog(Dialog owner, String title, boolean modal, Component c, JColorChooser chooserPane, ActionListener okListener, ActionListener cancelListener) throws HeadlessException {
    @Positive
    }

    @Positive
    public ColorChooserDialog(Frame owner, String title, boolean modal, Component c, JColorChooser chooserPane, ActionListener okListener, ActionListener cancelListener) throws HeadlessException {
    @Positive
    }

    @Positive
    protected void initColorChooserDialog(Component c, JColorChooser chooserPane, ActionListener okListener, ActionListener cancelListener);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public void show();

    @Positive
    public void reset();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    class Closer extends WindowAdapter implements Serializable {

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void windowClosing(WindowEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class DisposeOnClose extends ComponentAdapter implements Serializable {

    @Positive
        public void componentHidden(ComponentEvent e);
    @Positive
    }
    @Positive
}

    @Positive
@SuppressWarnings("serial")
    @Positive
class ColorTracker implements ActionListener, Serializable {

    @Positive
    public ColorTracker(JColorChooser c) {
    @Positive
    }

    @Positive
    public void actionPerformed(ActionEvent e);

    @Positive
    public Color getColor();
    @Positive
}

// CFWR semantic augmentation - variant 0
