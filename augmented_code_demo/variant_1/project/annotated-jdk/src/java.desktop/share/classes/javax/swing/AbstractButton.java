/*
    @Positive
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.text.*;
    @Positive
import java.awt.geom.*;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.Transient;
    @Positive
import java.util.Enumeration;
    @Positive
import java.io.Serializable;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.plaf.*;
    @Positive
import javax.accessibility.*;
    @Positive
import javax.swing.text.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(defaultProperty = "UI")
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class AbstractButton extends JComponent implements ItemSelectable, SwingConstants {

    @Positive
    @Interned
    @Positive
    public static final String MODEL_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String TEXT_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MNEMONIC_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String MARGIN_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String VERTICAL_ALIGNMENT_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String HORIZONTAL_ALIGNMENT_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String VERTICAL_TEXT_POSITION_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String HORIZONTAL_TEXT_POSITION_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String BORDER_PAINTED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String FOCUS_PAINTED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ROLLOVER_ENABLED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String CONTENT_AREA_FILLED_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String PRESSED_ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String SELECTED_ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ROLLOVER_ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String ROLLOVER_SELECTED_ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String DISABLED_ICON_CHANGED_PROPERTY;

    @Positive
    @Interned
    @Positive
    public static final String DISABLED_SELECTED_ICON_CHANGED_PROPERTY;

    @Positive
    protected ButtonModel model;

    @Positive
    protected ChangeListener changeListener;

    @Positive
    protected ActionListener actionListener;

    @Positive
    protected ItemListener itemListener;

    @Positive
    protected transient ChangeEvent changeEvent;

    @Positive
    protected AbstractButton() {
    @Positive
    }

    @Positive
    @BeanProperty(expert = true, description = "Whether the text of the button should come from the <code>Action</code>.")
    @Positive
    public void setHideActionText(boolean hideActionText);

    @Positive
    public boolean getHideActionText();

    @Positive
    public String getText();

    @Positive
    @BeanProperty(preferred = true, visualUpdate = true, description = "The button's text.")
    @Positive
    public void setText(String text);

    @Positive
    public boolean isSelected();

    @Positive
    public void setSelected(boolean b);

    @Positive
    public void doClick();

    @Positive
    public void doClick(int pressTime);

    @Positive
    @BeanProperty(visualUpdate = true, description = "The space between the button's border and the label.")
    @Positive
    public void setMargin(Insets m);

    @Positive
    public Insets getMargin();

    @Positive
    public Icon getIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The button's default icon")
    @Positive
    public void setIcon(Icon defaultIcon);

    @Positive
    public Icon getPressedIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The pressed icon for the button.")
    @Positive
    public void setPressedIcon(Icon pressedIcon);

    @Positive
    public Icon getSelectedIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The selected icon for the button.")
    @Positive
    public void setSelectedIcon(Icon selectedIcon);

    @Positive
    public Icon getRolloverIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The rollover icon for the button.")
    @Positive
    public void setRolloverIcon(Icon rolloverIcon);

    @Positive
    public Icon getRolloverSelectedIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The rollover selected icon for the button.")
    @Positive
    public void setRolloverSelectedIcon(Icon rolloverSelectedIcon);

    @Positive
    @Transient
    @Positive
    public Icon getDisabledIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The disabled icon for the button.")
    @Positive
    public void setDisabledIcon(Icon disabledIcon);

    @Positive
    public Icon getDisabledSelectedIcon();

    @Positive
    @BeanProperty(visualUpdate = true, description = "The disabled selection icon for the button.")
    @Positive
    public void setDisabledSelectedIcon(Icon disabledSelectedIcon);

    @Positive
    public int getVerticalAlignment();

    @Positive
    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.TOP", "SwingConstants.CENTER", "SwingConstants.BOTTOM" }, description = "The vertical alignment of the icon and text.")
    @Positive
    public void setVerticalAlignment(int alignment);

    @Positive
    public int getHorizontalAlignment();

    @Positive
    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.LEFT", "SwingConstants.CENTER", "SwingConstants.RIGHT", "SwingConstants.LEADING", "SwingConstants.TRAILING" }, description = "The horizontal alignment of the icon and text.")
    @Positive
    public void setHorizontalAlignment(int alignment);

    @Positive
    public int getVerticalTextPosition();

    @Positive
    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.TOP", "SwingConstants.CENTER", "SwingConstants.BOTTOM" }, description = "The vertical position of the text relative to the icon.")
    @Positive
    public void setVerticalTextPosition(int textPosition);

    @Positive
    public int getHorizontalTextPosition();

    @Positive
    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.LEFT", "SwingConstants.CENTER", "SwingConstants.RIGHT", "SwingConstants.LEADING", "SwingConstants.TRAILING" }, description = "The horizontal position of the text relative to the icon.")
    @Positive
    public void setHorizontalTextPosition(int textPosition);

    @Positive
    public int getIconTextGap();

    @Positive
    @BeanProperty(visualUpdate = true, description = "If both the icon and text properties are set, this property defines the space between them.")
    @Positive
    public void setIconTextGap(int iconTextGap);

    @Positive
    protected int checkHorizontalKey(int key, String exception);

    @Positive
    protected int checkVerticalKey(int key, String exception);

    @Positive
    public void removeNotify();

    @Positive
    public void setActionCommand(String actionCommand);

    @Positive
    public String getActionCommand();

    @Positive
    @BeanProperty(visualUpdate = true, description = "the Action instance connected with this ActionEvent source")
    @Positive
    public void setAction(Action a);

    @Positive
    public Action getAction();

    @Positive
    protected void configurePropertiesFromAction(Action a);

    @Positive
    void clientPropertyChanged(Object key, Object oldValue, Object newValue);

    @Positive
    boolean shouldUpdateSelectedStateFromAction();

    @Positive
    protected void actionPropertyChanged(Action action, String propertyName);

    @Positive
    void setIconFromAction(Action a);

    @Positive
    void smallIconChanged(Action a);

    @Positive
    void largeIconChanged(Action a);

    @Positive
    protected PropertyChangeListener createActionPropertyChangeListener(Action a);

    @Positive
    PropertyChangeListener createActionPropertyChangeListener0(Action a);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    private static class ButtonActionPropertyChangeListener extends ActionPropertyChangeListener<AbstractButton> {

    @Positive
        protected void actionPropertyChanged(AbstractButton button, Action action, PropertyChangeEvent e);
    @Positive
    }

    @Positive
    public boolean isBorderPainted();

    @Positive
    @BeanProperty(visualUpdate = true, description = "Whether the border should be painted.")
    @Positive
    public void setBorderPainted(boolean b);

    @Positive
    protected void paintBorder(Graphics g);

    @Positive
    public boolean isFocusPainted();

    @Positive
    @BeanProperty(visualUpdate = true, description = "Whether focus should be painted")
    @Positive
    public void setFocusPainted(boolean b);

    @Positive
    public boolean isContentAreaFilled();

    @Positive
    @BeanProperty(visualUpdate = true, description = "Whether the button should paint the content area or leave it transparent.")
    @Positive
    public void setContentAreaFilled(boolean b);

    @Positive
    public boolean isRolloverEnabled();

    @Positive
    @BeanProperty(visualUpdate = true, description = "Whether rollover effects should be enabled.")
    @Positive
    public void setRolloverEnabled(boolean b);

    @Positive
    public int getMnemonic();

    @Positive
    @BeanProperty(visualUpdate = true, description = "the keyboard character mnemonic")
    @Positive
    public void setMnemonic(int mnemonic);

    @Positive
    @BeanProperty(visualUpdate = true, description = "the keyboard character mnemonic")
    @Positive
    public void setMnemonic(char mnemonic);

    @Positive
    @BeanProperty(visualUpdate = true, description = "the index into the String to draw the keyboard character mnemonic at")
    @Positive
    public void setDisplayedMnemonicIndex(int index) throws IllegalArgumentException;

    @Positive
    public int getDisplayedMnemonicIndex();

    @Positive
    public void setMultiClickThreshhold(long threshhold);

    @Positive
    public long getMultiClickThreshhold();

    @Positive
    public ButtonModel getModel();

    @Positive
    @BeanProperty(description = "Model that the Button uses.")
    @Positive
    public void setModel(ButtonModel newModel);

    @Positive
    public ButtonUI getUI();

    @Positive
    @BeanProperty(hidden = true, visualUpdate = true, description = "The UI object that implements the LookAndFeel.")
    @Positive
    public void setUI(ButtonUI ui);

    @Positive
    public void updateUI();

    @Positive
    protected void addImpl(Component comp, Object constraints, int index);

    @Positive
    public void setLayout(LayoutManager mgr);

    @Positive
    public void addChangeListener(ChangeListener l);

    @Positive
    public void removeChangeListener(ChangeListener l);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public ChangeListener[] getChangeListeners();

    @Positive
    protected void fireStateChanged();

    @Positive
    public void addActionListener(ActionListener l);

    @Positive
    public void removeActionListener(ActionListener l);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public ActionListener[] getActionListeners();

    @Positive
    protected ChangeListener createChangeListener();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class ButtonChangeListener implements ChangeListener, Serializable {

    @Positive
        public void stateChanged(ChangeEvent e);
    @Positive
    }

    @Positive
    protected void fireActionPerformed(ActionEvent event);

    @Positive
    protected void fireItemStateChanged(ItemEvent event);

    @Positive
    protected ActionListener createActionListener();

    @Positive
    protected ItemListener createItemListener();

    @Positive
    public void setEnabled(boolean b);

    @Positive
    @Deprecated
    @Positive
    public String getLabel();

    @Positive
    @Deprecated
    @Positive
    @BeanProperty(description = "Replace by setText(text)")
    @Positive
    public void setLabel(String label);

    @Positive
    public void addItemListener(ItemListener l);

    @Positive
    public void removeItemListener(ItemListener l);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public ItemListener[] getItemListeners();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Object[] getSelectedObjects();

    @Positive
    protected void init(String text, Icon icon);

    @Positive
    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);

    @Positive
    void setUIProperty(String propertyName, Object value);

    @Positive
    protected String paramString();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    class Handler implements ActionListener, ChangeListener, ItemListener, Serializable {

    @Positive
        public void stateChanged(ChangeEvent e);

    @Positive
        public void actionPerformed(ActionEvent event);

    @Positive
        public void itemStateChanged(ItemEvent event);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected abstract class AccessibleAbstractButton extends AccessibleJComponent implements AccessibleAction, AccessibleValue, AccessibleText, AccessibleExtendedComponent {

    @Positive
        protected AccessibleAbstractButton() {
    @Positive
        }

    @Positive
        public String getAccessibleName();

    @Positive
        public AccessibleIcon[] getAccessibleIcon();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();

    @Positive
        public AccessibleRelationSet getAccessibleRelationSet();

    @Positive
        public AccessibleAction getAccessibleAction();

    @Positive
        public AccessibleValue getAccessibleValue();

    @Positive
        public int getAccessibleActionCount();

    @Positive
        public String getAccessibleActionDescription(int i);

    @Positive
        public boolean doAccessibleAction(int i);

    @Positive
        public Number getCurrentAccessibleValue();

    @Positive
        public boolean setCurrentAccessibleValue(Number n);

    @Positive
        public Number getMinimumAccessibleValue();

    @Positive
        public Number getMaximumAccessibleValue();

    @Positive
        public AccessibleText getAccessibleText();

    @Positive
        public int getIndexAtPoint(Point p);

    @Positive
        public Rectangle getCharacterBounds(int i);

    @Positive
        public int getCharCount();

    @Positive
        public int getCaretPosition();

    @Positive
        public String getAtIndex(int part, int index);

    @Positive
        public String getAfterIndex(int part, int index);

    @Positive
        public String getBeforeIndex(int part, int index);

    @Positive
        public AttributeSet getCharacterAttribute(int i);

    @Positive
        public int getSelectionStart();

    @Positive
        public int getSelectionEnd();

    @Positive
        public String getSelectedText();

    @Positive
        AccessibleExtendedComponent getAccessibleExtendedComponent();

    @Positive
        public String getToolTipText();

    @Positive
        public String getTitledBorderText();

    @Positive
        public AccessibleKeyBinding getAccessibleKeyBinding();

    @Positive
        class ButtonKeyBinding implements AccessibleKeyBinding {

    @Positive
            public int getAccessibleKeyBindingCount();

    @Positive
            public java.lang.Object getAccessibleKeyBinding(int i);
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
