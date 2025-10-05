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
import java.awt.AWTEvent;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.EventQueue;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.FontMetrics;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.InputEvent;
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
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import javax.swing.event.ChangeEvent;
    @Positive
import javax.swing.event.ChangeListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.text.Document;
    @Positive
import javax.swing.text.JTextComponent;
    @Positive
import javax.swing.text.PlainDocument;
    @Positive
import javax.swing.text.TextAction;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(defaultProperty = "UIClassID", description = "A component which allows for the editing of a single line of text.")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JTextField extends JTextComponent implements SwingConstants {

    @Positive
    public JTextField() {
    @Positive
    }

    @Positive
    public JTextField(String text) {
    @Positive
    }

    @Positive
    public JTextField(int columns) {
    @Positive
    }

    @Positive
    public JTextField(String text, int columns) {
    @Positive
    }

    @Positive
    public JTextField(Document doc, String text, int columns) {
    @Positive
    }

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public String getUIClassID();

    @Positive
    @BeanProperty(expert = true, description = "the text document model")
    @Positive
    public void setDocument(Document doc);

    @Positive
    @Override
    @Positive
    public boolean isValidateRoot();

    @Positive
    public int getHorizontalAlignment();

    @Positive
    @BeanProperty(preferred = true, enumerationValues = { "JTextField.LEFT", "JTextField.CENTER", "JTextField.RIGHT", "JTextField.LEADING", "JTextField.TRAILING" }, description = "Set the field alignment to LEFT, CENTER, RIGHT, LEADING (the default) or TRAILING")
    @Positive
    public void setHorizontalAlignment(int alignment);

    @Positive
    protected Document createDefaultModel();

    @Positive
    public int getColumns();

    @Positive
    @BeanProperty(bound = false, description = "the number of columns preferred for display")
    @Positive
    public void setColumns(int columns);

    @Positive
    protected int getColumnWidth();

    @Positive
    public Dimension getPreferredSize();

    @Positive
    public void setFont(Font f);

    @Positive
    public synchronized void addActionListener(ActionListener l);

    @Positive
    public synchronized void removeActionListener(ActionListener l);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public synchronized ActionListener[] getActionListeners();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void fireActionPerformed();

    @Positive
    public void setActionCommand(String command);

    @Positive
    @BeanProperty(visualUpdate = true, description = "the Action instance connected with this ActionEvent source")
    @Positive
    public void setAction(Action a);

    @Positive
    public Action getAction();

    @Positive
    protected void configurePropertiesFromAction(Action a);

    @Positive
    protected void actionPropertyChanged(Action action, String propertyName);

    @Positive
    protected PropertyChangeListener createActionPropertyChangeListener(Action a);

    @Positive
    private static class TextFieldActionPropertyChangeListener extends ActionPropertyChangeListener<JTextField> {

    @Positive
        protected void actionPropertyChanged(JTextField textField, Action action, PropertyChangeEvent e);
    @Positive
    }

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Action[] getActions();

    @Positive
    public void postActionEvent();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public BoundedRangeModel getHorizontalVisibility();

    @Positive
    public int getScrollOffset();

    @Positive
    public void setScrollOffset(int scrollOffset);

    @Positive
    public void scrollRectToVisible(Rectangle r);

    @Positive
    boolean hasActionListener();

    @Positive
    @Interned
    @Positive
    public static final String notifyAction;

    @Positive
    static class NotifyAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);

    @Positive
        public boolean isEnabled();
    @Positive
    }

    @Positive
    class ScrollRepainter implements ChangeListener, Serializable {

    @Positive
        public void stateChanged(ChangeEvent e);
    @Positive
    }

    @Positive
    protected String paramString();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJTextField extends AccessibleJTextComponent {

    @Positive
        protected AccessibleJTextField() {
    @Positive
        }

    @Positive
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
