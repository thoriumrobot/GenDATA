/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.awt.BorderLayout;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dialog;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.Frame;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.KeyboardFocusManager;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.event.ComponentAdapter;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.WindowAdapter;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.event.WindowListener;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Vector;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.swing.event.InternalFrameAdapter;
    @Positive
import javax.swing.event.InternalFrameEvent;
    @Positive
import javax.swing.plaf.OptionPaneUI;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import static javax.swing.ClientPropertyKey.PopupFactory_FORCE_HEAVYWEIGHT_POPUP;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@JavaBean(defaultProperty = "UI", description = "A component which implements standard dialog box controls.")
    @Positive
@SwingContainer
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JOptionPane extends JComponent implements Accessible {

    @Positive
    public static final Object UNINITIALIZED_VALUE;

    @Positive
    public static final int DEFAULT_OPTION;

    @Positive
    public static final int YES_NO_OPTION;

    @Positive
    public static final int YES_NO_CANCEL_OPTION;

    @Positive
    public static final int OK_CANCEL_OPTION;

    @Positive
    public static final int YES_OPTION;

    @Positive
    public static final int NO_OPTION;

    @Positive
    public static final int CANCEL_OPTION;

    @Positive
    public static final int OK_OPTION;

    @Positive
    public static final int CLOSED_OPTION;

    @Positive
    public static final int ERROR_MESSAGE;

    @Positive
    public static final int INFORMATION_MESSAGE;

    @Positive
    public static final int WARNING_MESSAGE;

    @Positive
    public static final int QUESTION_MESSAGE;

    @Positive
    public static final int PLAIN_MESSAGE;

    @Positive
    @Interned
    @Positive
    public static final String ICON_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MESSAGE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String VALUE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String OPTIONS_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String INITIAL_VALUE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MESSAGE_TYPE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String OPTION_TYPE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String SELECTION_VALUES_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String INITIAL_SELECTION_VALUE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String INPUT_VALUE_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String WANTS_INPUT_PROPERTY;

    @Positive
    @Nullable
    @Positive
    protected transient Icon icon;

    @Positive
    @Nullable
    @Positive
    protected transient Object message;

    @Positive
    @Nullable
    @Positive
    protected transient Object[] options;

    @Positive
    @Nullable
    @Positive
    protected transient Object initialValue;

    @Positive
    protected int messageType;

    @Positive
    protected int optionType;

    @Positive
    @Nullable
    @Positive
    protected transient Object value;

    @Positive
    @Nullable
    @Positive
    protected transient Object[] selectionValues;

    @Positive
    @Nullable
    @Positive
    protected transient Object inputValue;

    @Positive
    @Nullable
    @Positive
    protected transient Object initialSelectionValue;

    @Positive
    protected boolean wantsInput;

    @Positive
    public static String showInputDialog(@Nullable Object message) throws HeadlessException;

    @Positive
    public static String showInputDialog(@Nullable Object message, @Nullable Object initialSelectionValue);

    @Positive
    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    @Positive
    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable Object initialSelectionValue);

    @Positive
    public static String showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType) throws HeadlessException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static Object showInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon, @Nullable Object[] selectionValues, @Nullable Object initialSelectionValue) throws HeadlessException;

    @Positive
    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    @Positive
    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType) throws HeadlessException;

    @Positive
    public static void showMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon) throws HeadlessException;

    @Positive
    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message) throws HeadlessException;

    @Positive
    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType) throws HeadlessException;

    @Positive
    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType) throws HeadlessException;

    @Positive
    public static int showConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon) throws HeadlessException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static int showOptionDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon, @Nullable Object[] options, @Nullable Object initialValue) throws HeadlessException;

    @Positive
    public JDialog createDialog(@Nullable Component parentComponent, @Nullable String title) throws HeadlessException;

    @Positive
    public JDialog createDialog(@Nullable String title) throws HeadlessException;

    @Positive
    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message);

    @Positive
    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType);

    @Positive
    public static void showInternalMessageDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon);

    @Positive
    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message);

    @Positive
    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType);

    @Positive
    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType);

    @Positive
    public static int showInternalConfirmDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, @Nullable Icon icon);

    @Positive
    public static int showInternalOptionDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int optionType, int messageType, Icon icon, @Nullable Object[] options, @Nullable Object initialValue);

    @Positive
    public static String showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message);

    @Positive
    public static String showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType);

    @Positive
    public static Object showInternalInputDialog(@Nullable Component parentComponent, @Nullable Object message, @Nullable String title, int messageType, @Nullable Icon icon, @Nullable Object[] selectionValues, @Nullable Object initialSelectionValue);

    @Positive
    public JInternalFrame createInternalFrame(@Nullable Component parentComponent, @Nullable String title);

    @Positive
    public static Frame getFrameForComponent(@Nullable Component parentComponent) throws HeadlessException;

    @Positive
    static Window getWindowForComponent(@Nullable Component parentComponent) throws HeadlessException;

    @Positive
    public static JDesktopPane getDesktopPaneForComponent(@Nullable Component parentComponent);

    @Positive
    public static void setRootFrame(@Nullable Frame newRootFrame);

    @Positive
    public static Frame getRootFrame() throws HeadlessException;

    @Positive
    public JOptionPane() {
    @Positive
    }

    @Positive
    public JOptionPane(@Nullable Object message) {
    @Positive
    }

    @Positive
    public JOptionPane(Object message, int messageType) {
    @Positive
    }

    @Positive
    public JOptionPane(@Nullable Object message, int messageType, int optionType) {
    @Positive
    }

    @Positive
    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon) {
    @Positive
    }

    @Positive
    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon, @Nullable Object[] options) {
    @Positive
    }

    @Positive
    public JOptionPane(@Nullable Object message, int messageType, int optionType, @Nullable Icon icon, @Nullable Object[] options, @Nullable Object initialValue) {
    @Positive
    }

    @Positive
    @BeanProperty(hidden = true, description = "The UI object that implements the optionpane's LookAndFeel")
    @Positive
    public void setUI(@Nullable OptionPaneUI ui);

    @Positive
    @Nullable
    @Positive
    public OptionPaneUI getUI();

    @Positive
    public void updateUI();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public String getUIClassID();

    @Positive
    @BeanProperty(preferred = true, description = "The optionpane's message object.")
    @Positive
    public void setMessage(@Nullable Object newMessage);

    @Positive
    @Nullable
    @Positive
    public Object getMessage();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's type icon.")
    @Positive
    public void setIcon(@Nullable Icon newIcon);

    @Positive
    @Nullable
    @Positive
    public Icon getIcon();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's value object.")
    @Positive
    public void setValue(@Nullable Object newValue);

    @Positive
    @Nullable
    @Positive
    public Object getValue();

    @Positive
    @BeanProperty(description = "The option pane's options objects.")
    @Positive
    public void setOptions(@Nullable Object[] newOptions);

    @Positive
    @Nullable
    @Positive
    public Object[] getOptions();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's initial value object.")
    @Positive
    public void setInitialValue(@Nullable Object newInitialValue);

    @Positive
    @Nullable
    @Positive
    public Object getInitialValue();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's message type.")
    @Positive
    public void setMessageType(int newType);

    @Positive
    public int getMessageType();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's option type.")
    @Positive
    public void setOptionType(int newType);

    @Positive
    public int getOptionType();

    @Positive
    @BeanProperty(description = "The option pane's selection values.")
    @Positive
    public void setSelectionValues(@Nullable Object[] newValues);

    @Positive
    @Nullable
    @Positive
    public Object[] getSelectionValues();

    @Positive
    @BeanProperty(description = "The option pane's initial selection value object.")
    @Positive
    public void setInitialSelectionValue(@Nullable Object newValue);

    @Positive
    @Nullable
    @Positive
    public Object getInitialSelectionValue();

    @Positive
    @BeanProperty(preferred = true, description = "The option pane's input value object.")
    @Positive
    public void setInputValue(@Nullable Object newValue);

    @Positive
    @Nullable
    @Positive
    public Object getInputValue();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public int getMaxCharactersPerLineCount();

    @Positive
    @BeanProperty(preferred = true, description = "Flag which allows the user to input a value.")
    @Positive
    public void setWantsInput(boolean newValue);

    @Positive
    public boolean getWantsInput();

    @Positive
    public void selectInitialValue();

    @Positive
    protected String paramString();

    @Positive
    @BeanProperty(bound = false, expert = true, description = "The AccessibleContext associated with this option pane")
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJOptionPane extends AccessibleJComponent {

    @Positive
        protected AccessibleJOptionPane() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();
    @Positive
    }
    @Positive
}
