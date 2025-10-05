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

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dialog;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.HeadlessException;
import java.awt.KeyboardFocusManager;
import java.awt.Point;
import java.awt.Window;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Vector;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.swing.event.InternalFrameAdapter;
import javax.swing.event.InternalFrameEvent;
import javax.swing.plaf.OptionPaneUI;
import sun.awt.AWTAccessor;
import static javax.swing.ClientPropertyKey.PopupFactory_FORCE_HEAVYWEIGHT_POPUP;

@AnnotatedFor({ "interning", "nullness" })
@JavaBean(defaultProperty = "UI", description = "A component which implements standard dialog box controls.")
@SwingContainer
@SuppressWarnings("serial")
public class JOptionPane extends JComponent implements Accessible {

    public static final Object UNINITIALIZED_VALUE;

    public static final int DEFAULT_OPTION;

    public static final int YES_NO_OPTION;

    public static final int YES_NO_CANCEL_OPTION;

    public static final int OK_CANCEL_OPTION;

    public static final int YES_OPTION;

    public static final int NO_OPTION;

    public static final int CANCEL_OPTION;

    public static final int OK_OPTION;

    public static final int CLOSED_OPTION;

    public static final int ERROR_MESSAGE;

    public static final int INFORMATION_MESSAGE;

    public static final int WARNING_MESSAGE;

    public static final int QUESTION_MESSAGE;

    public static final int PLAIN_MESSAGE;

    @Interned
    public static final String ICON_PROPERTY;

    @Interned
    public static final String MESSAGE_PROPERTY;

    @Interned
    public static final String VALUE_PROPERTY;

    @Interned
    public static final String OPTIONS_PROPERTY;

    @Interned
    public static final String INITIAL_VALUE_PROPERTY;

    @Interned
    public static final String MESSAGE_TYPE_PROPERTY;

    @Interned
    public static final String OPTION_TYPE_PROPERTY;

    @Interned
    public static final String SELECTION_VALUES_PROPERTY;

    @Interned
    public static final String INITIAL_SELECTION_VALUE_PROPERTY;

    @Interned
    public static final String INPUT_VALUE_PROPERTY;

    @Interned
    public static final String WANTS_INPUT_PROPERTY;

    @Nullable
    protected transient Icon icon;

    @Nullable
    protected transient Object message;

    @Nullable
    protected transient Object[] options;

    @Nullable
    protected transient Object initialValue;

    protected int messageType;

    protected int optionType;

    @Nullable
    protected transient Object value;

    @Nullable
    protected transient Object[] selectionValues;

    @Nullable
    protected transient Object inputValue;

    @Nullable
    protected transient Object initialSelectionValue;

    protected boolean wantsInput;

    public static String showInputDialog(@Nullable Object message) throws HeadlessException;

    public static String showInputDialog(@Nullable Object message, @Nullable Object initialSelectionValue);

    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable Object initialSelectionValue);

    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType) throws HeadlessException;

    @SuppressWarnings("deprecation")
    public static Object showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon, @Nullable Object[] selectionValues, @Nullable Object initialSelectionValue) throws HeadlessException;

    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType) throws HeadlessException;

    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon) throws HeadlessException;

    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType) throws HeadlessException;

    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType) throws HeadlessException;

    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon) throws HeadlessException;

    @SuppressWarnings("deprecation")
    public static int showOptionDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon, @Nullable Object[] options, @Nullable Object initialValue) throws HeadlessException;

    public JDialog createDialog(@Nullable Component parentComponent, @Nullable String title) throws HeadlessException;

    public JDialog createDialog(@Nullable String title) throws HeadlessException;

    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message);

    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType);

    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon);

    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message);

    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType);

    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType);

    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon);

    public static int showInternalOptionDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, Icon icon, @Nullable Object[] options, @Nullable Object initialValue);

    public static String showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message);

    public static String showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType);

    public static Object showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon, @Nullable Object[] selectionValues, @Nullable Object initialSelectionValue);

    public JInternalFrame createInternalFrame(@Nullable Component parentComponent, @Nullable String title);

    public static Frame getFrameForComponent(@Nullable Component parentComponent) throws HeadlessException;

    static Window getWindowForComponent(@Nullable Component parentComponent) throws HeadlessException;

    public static JDesktopPane getDesktopPaneForComponent(@Nullable Component parentComponent);

    public static void setRootFrame(@Nullable Frame newRootFrame);

    public static Frame getRootFrame() throws HeadlessException;

    public JOptionPane() {
    }

    public JOptionPane(@Nullable Object message) {
    }

    public JOptionPane(Object message, int messageType) {
    }

    public JOptionPane(@Nullable Object message, int messageType, int optionType) {
    }

    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon) {
    }

    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon, @Nullable Object[] options) {
    }

    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon, @Nullable Object[] options, @Nullable Object initialValue) {
    }

    @BeanProperty(hidden = true, description = "The UI object that implements the optionpane's LookAndFeel")
    public void setUI(@Nullable OptionPaneUI ui);

    @Nullable
    public OptionPaneUI getUI();

    public void updateUI();

    @BeanProperty(bound = false)
    public String getUIClassID();

    @BeanProperty(preferred = true, description = "The optionpane's message object.")
    public void setMessage(@Nullable Object newMessage);

    @Nullable
    public Object getMessage();

    @BeanProperty(preferred = true, description = "The option pane's type icon.")
    public void setIcon(@Nullable Icon newIcon);

    @Nullable
    public Icon getIcon();

    @BeanProperty(preferred = true, description = "The option pane's value object.")
    public void setValue(@Nullable Object newValue);

    @Nullable
    public Object getValue();

    @BeanProperty(description = "The option pane's options objects.")
    public void setOptions(@Nullable Object[] newOptions);

    @Nullable
    public Object[] getOptions();

    @BeanProperty(preferred = true, description = "The option pane's initial value object.")
    public void setInitialValue(@Nullable Object newInitialValue);

    @Nullable
    public Object getInitialValue();

    @BeanProperty(preferred = true, description = "The option pane's message type.")
    public void setMessageType(int newType);

    public int getMessageType();

    @BeanProperty(preferred = true, description = "The option pane's option type.")
    public void setOptionType(int newType);

    public int getOptionType();

    @BeanProperty(description = "The option pane's selection values.")
    public void setSelectionValues(@Nullable Object[] newValues);

    @Nullable
    public Object[] getSelectionValues();

    @BeanProperty(description = "The option pane's initial selection value object.")
    public void setInitialSelectionValue(@Nullable Object newValue);

    @Nullable
    public Object getInitialSelectionValue();

    @BeanProperty(preferred = true, description = "The option pane's input value object.")
    public void setInputValue(@Nullable Object newValue);

    @Nullable
    public Object getInputValue();

    @BeanProperty(bound = false)
    public int getMaxCharactersPerLineCount();

    @BeanProperty(preferred = true, description = "Flag which allows the user to input a value.")
    public void setWantsInput(boolean newValue);

    public boolean getWantsInput();

    public void selectInitialValue();

    protected String paramString();

    @BeanProperty(bound = false, expert = true, description = "The AccessibleContext associated with this option pane")
    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJOptionPane extends AccessibleJComponent {

        protected AccessibleJOptionPane() {
        }

        public AccessibleRole getAccessibleRole();
    }
}
