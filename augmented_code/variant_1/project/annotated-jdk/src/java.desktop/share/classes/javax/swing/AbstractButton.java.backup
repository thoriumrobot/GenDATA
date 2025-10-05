/*
 * Copyright (c) 1997, 2018, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.*;
import java.awt.event.*;
import java.text.*;
import java.awt.geom.*;
import java.beans.JavaBean;
import java.beans.BeanProperty;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.Transient;
import java.util.Enumeration;
import java.io.Serializable;
import javax.swing.event.*;
import javax.swing.plaf.*;
import javax.accessibility.*;
import javax.swing.text.*;

@AnnotatedFor({ "interning" })
@JavaBean(defaultProperty = "UI")
@SuppressWarnings("serial")
public abstract class AbstractButton extends JComponent implements ItemSelectable, SwingConstants {

    @Interned
    public static final String MODEL_CHANGED_PROPERTY;

    @Interned
    public static final String TEXT_CHANGED_PROPERTY;

    @Interned
    public static final String MNEMONIC_CHANGED_PROPERTY;

    @Interned
    public static final String MARGIN_CHANGED_PROPERTY;

    @Interned
    public static final String VERTICAL_ALIGNMENT_CHANGED_PROPERTY;

    @Interned
    public static final String HORIZONTAL_ALIGNMENT_CHANGED_PROPERTY;

    @Interned
    public static final String VERTICAL_TEXT_POSITION_CHANGED_PROPERTY;

    @Interned
    public static final String HORIZONTAL_TEXT_POSITION_CHANGED_PROPERTY;

    @Interned
    public static final String BORDER_PAINTED_CHANGED_PROPERTY;

    @Interned
    public static final String FOCUS_PAINTED_CHANGED_PROPERTY;

    @Interned
    public static final String ROLLOVER_ENABLED_CHANGED_PROPERTY;

    @Interned
    public static final String CONTENT_AREA_FILLED_CHANGED_PROPERTY;

    @Interned
    public static final String ICON_CHANGED_PROPERTY;

    @Interned
    public static final String PRESSED_ICON_CHANGED_PROPERTY;

    @Interned
    public static final String SELECTED_ICON_CHANGED_PROPERTY;

    @Interned
    public static final String ROLLOVER_ICON_CHANGED_PROPERTY;

    @Interned
    public static final String ROLLOVER_SELECTED_ICON_CHANGED_PROPERTY;

    @Interned
    public static final String DISABLED_ICON_CHANGED_PROPERTY;

    @Interned
    public static final String DISABLED_SELECTED_ICON_CHANGED_PROPERTY;

    protected ButtonModel model;

    protected ChangeListener changeListener;

    protected ActionListener actionListener;

    protected ItemListener itemListener;

    protected transient ChangeEvent changeEvent;

    protected AbstractButton() {
    }

    @BeanProperty(expert = true, description = "Whether the text of the button should come from the <code>Action</code>.")
    public void setHideActionText(boolean hideActionText);

    public boolean getHideActionText();

    public String getText();

    @BeanProperty(preferred = true, visualUpdate = true, description = "The button's text.")
    public void setText(String text);

    public boolean isSelected();

    public void setSelected(boolean b);

    public void doClick();

    public void doClick(int pressTime);

    @BeanProperty(visualUpdate = true, description = "The space between the button's border and the label.")
    public void setMargin(Insets m);

    public Insets getMargin();

    public Icon getIcon();

    @BeanProperty(visualUpdate = true, description = "The button's default icon")
    public void setIcon(Icon defaultIcon);

    public Icon getPressedIcon();

    @BeanProperty(visualUpdate = true, description = "The pressed icon for the button.")
    public void setPressedIcon(Icon pressedIcon);

    public Icon getSelectedIcon();

    @BeanProperty(visualUpdate = true, description = "The selected icon for the button.")
    public void setSelectedIcon(Icon selectedIcon);

    public Icon getRolloverIcon();

    @BeanProperty(visualUpdate = true, description = "The rollover icon for the button.")
    public void setRolloverIcon(Icon rolloverIcon);

    public Icon getRolloverSelectedIcon();

    @BeanProperty(visualUpdate = true, description = "The rollover selected icon for the button.")
    public void setRolloverSelectedIcon(Icon rolloverSelectedIcon);

    @Transient
    public Icon getDisabledIcon();

    @BeanProperty(visualUpdate = true, description = "The disabled icon for the button.")
    public void setDisabledIcon(Icon disabledIcon);

    public Icon getDisabledSelectedIcon();

    @BeanProperty(visualUpdate = true, description = "The disabled selection icon for the button.")
    public void setDisabledSelectedIcon(Icon disabledSelectedIcon);

    public int getVerticalAlignment();

    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.TOP", "SwingConstants.CENTER", "SwingConstants.BOTTOM" }, description = "The vertical alignment of the icon and text.")
    public void setVerticalAlignment(int alignment);

    public int getHorizontalAlignment();

    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.LEFT", "SwingConstants.CENTER", "SwingConstants.RIGHT", "SwingConstants.LEADING", "SwingConstants.TRAILING" }, description = "The horizontal alignment of the icon and text.")
    public void setHorizontalAlignment(int alignment);

    public int getVerticalTextPosition();

    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.TOP", "SwingConstants.CENTER", "SwingConstants.BOTTOM" }, description = "The vertical position of the text relative to the icon.")
    public void setVerticalTextPosition(int textPosition);

    public int getHorizontalTextPosition();

    @BeanProperty(visualUpdate = true, enumerationValues = { "SwingConstants.LEFT", "SwingConstants.CENTER", "SwingConstants.RIGHT", "SwingConstants.LEADING", "SwingConstants.TRAILING" }, description = "The horizontal position of the text relative to the icon.")
    public void setHorizontalTextPosition(int textPosition);

    public int getIconTextGap();

    @BeanProperty(visualUpdate = true, description = "If both the icon and text properties are set, this property defines the space between them.")
    public void setIconTextGap(int iconTextGap);

    protected int checkHorizontalKey(int key, String exception);

    protected int checkVerticalKey(int key, String exception);

    public void removeNotify();

    public void setActionCommand(String actionCommand);

    public String getActionCommand();

    @BeanProperty(visualUpdate = true, description = "the Action instance connected with this ActionEvent source")
    public void setAction(Action a);

    public Action getAction();

    protected void configurePropertiesFromAction(Action a);

    void clientPropertyChanged(Object key, Object oldValue, Object newValue);

    boolean shouldUpdateSelectedStateFromAction();

    protected void actionPropertyChanged(Action action, String propertyName);

    void setIconFromAction(Action a);

    void smallIconChanged(Action a);

    void largeIconChanged(Action a);

    protected PropertyChangeListener createActionPropertyChangeListener(Action a);

    PropertyChangeListener createActionPropertyChangeListener0(Action a);

    @SuppressWarnings("serial")
    private static class ButtonActionPropertyChangeListener extends ActionPropertyChangeListener<AbstractButton> {

        protected void actionPropertyChanged(AbstractButton button, Action action, PropertyChangeEvent e);
    }

    public boolean isBorderPainted();

    @BeanProperty(visualUpdate = true, description = "Whether the border should be painted.")
    public void setBorderPainted(boolean b);

    protected void paintBorder(Graphics g);

    public boolean isFocusPainted();

    @BeanProperty(visualUpdate = true, description = "Whether focus should be painted")
    public void setFocusPainted(boolean b);

    public boolean isContentAreaFilled();

    @BeanProperty(visualUpdate = true, description = "Whether the button should paint the content area or leave it transparent.")
    public void setContentAreaFilled(boolean b);

    public boolean isRolloverEnabled();

    @BeanProperty(visualUpdate = true, description = "Whether rollover effects should be enabled.")
    public void setRolloverEnabled(boolean b);

    public int getMnemonic();

    @BeanProperty(visualUpdate = true, description = "the keyboard character mnemonic")
    public void setMnemonic(int mnemonic);

    @BeanProperty(visualUpdate = true, description = "the keyboard character mnemonic")
    public void setMnemonic(char mnemonic);

    @BeanProperty(visualUpdate = true, description = "the index into the String to draw the keyboard character mnemonic at")
    public void setDisplayedMnemonicIndex(int index) throws IllegalArgumentException;

    public int getDisplayedMnemonicIndex();

    public void setMultiClickThreshhold(long threshhold);

    public long getMultiClickThreshhold();

    public ButtonModel getModel();

    @BeanProperty(description = "Model that the Button uses.")
    public void setModel(ButtonModel newModel);

    public ButtonUI getUI();

    @BeanProperty(hidden = true, visualUpdate = true, description = "The UI object that implements the LookAndFeel.")
    public void setUI(ButtonUI ui);

    public void updateUI();

    protected void addImpl(Component comp, Object constraints, int index);

    public void setLayout(LayoutManager mgr);

    public void addChangeListener(ChangeListener l);

    public void removeChangeListener(ChangeListener l);

    @BeanProperty(bound = false)
    public ChangeListener[] getChangeListeners();

    protected void fireStateChanged();

    public void addActionListener(ActionListener l);

    public void removeActionListener(ActionListener l);

    @BeanProperty(bound = false)
    public ActionListener[] getActionListeners();

    protected ChangeListener createChangeListener();

    @SuppressWarnings("serial")
    protected class ButtonChangeListener implements ChangeListener, Serializable {

        public void stateChanged(ChangeEvent e);
    }

    protected void fireActionPerformed(ActionEvent event);

    protected void fireItemStateChanged(ItemEvent event);

    protected ActionListener createActionListener();

    protected ItemListener createItemListener();

    public void setEnabled(boolean b);

    @Deprecated
    public String getLabel();

    @Deprecated
    @BeanProperty(description = "Replace by setText(text)")
    public void setLabel(String label);

    public void addItemListener(ItemListener l);

    public void removeItemListener(ItemListener l);

    @BeanProperty(bound = false)
    public ItemListener[] getItemListeners();

    @BeanProperty(bound = false)
    public Object[] getSelectedObjects();

    protected void init(String text, Icon icon);

    public boolean imageUpdate(Image img, int infoflags, int x, int y, int w, int h);

    void setUIProperty(String propertyName, Object value);

    protected String paramString();

    @SuppressWarnings("serial")
    class Handler implements ActionListener, ChangeListener, ItemListener, Serializable {

        public void stateChanged(ChangeEvent e);

        public void actionPerformed(ActionEvent event);

        public void itemStateChanged(ItemEvent event);
    }

    @SuppressWarnings("serial")
    protected abstract class AccessibleAbstractButton extends AccessibleJComponent implements AccessibleAction, AccessibleValue, AccessibleText, AccessibleExtendedComponent {

        protected AccessibleAbstractButton() {
        }

        public String getAccessibleName();

        public AccessibleIcon[] getAccessibleIcon();

        public AccessibleStateSet getAccessibleStateSet();

        public AccessibleRelationSet getAccessibleRelationSet();

        public AccessibleAction getAccessibleAction();

        public AccessibleValue getAccessibleValue();

        public int getAccessibleActionCount();

        public String getAccessibleActionDescription(int i);

        public boolean doAccessibleAction(int i);

        public Number getCurrentAccessibleValue();

        public boolean setCurrentAccessibleValue(Number n);

        public Number getMinimumAccessibleValue();

        public Number getMaximumAccessibleValue();

        public AccessibleText getAccessibleText();

        public int getIndexAtPoint(Point p);

        public Rectangle getCharacterBounds(int i);

        public int getCharCount();

        public int getCaretPosition();

        public String getAtIndex(int part, int index);

        public String getAfterIndex(int part, int index);

        public String getBeforeIndex(int part, int index);

        public AttributeSet getCharacterAttribute(int i);

        public int getSelectionStart();

        public int getSelectionEnd();

        public String getSelectedText();

        AccessibleExtendedComponent getAccessibleExtendedComponent();

        public String getToolTipText();

        public String getTitledBorderText();

        public AccessibleKeyBinding getAccessibleKeyBinding();

        class ButtonKeyBinding implements AccessibleKeyBinding {

            public int getAccessibleKeyBindingCount();

            public java.lang.Object getAccessibleKeyBinding(int i);
        }
    }
}
