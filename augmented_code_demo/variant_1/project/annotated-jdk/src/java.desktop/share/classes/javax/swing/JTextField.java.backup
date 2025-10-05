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
import java.awt.AWTEvent;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Insets;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.EventListenerList;
import javax.swing.text.Document;
import javax.swing.text.JTextComponent;
import javax.swing.text.PlainDocument;
import javax.swing.text.TextAction;

@AnnotatedFor({ "interning" })
@JavaBean(defaultProperty = "UIClassID", description = "A component which allows for the editing of a single line of text.")
@SwingContainer(false)
@SuppressWarnings("serial")
public class JTextField extends JTextComponent implements SwingConstants {

    public JTextField() {
    }

    public JTextField(String text) {
    }

    public JTextField(int columns) {
    }

    public JTextField(String text, int columns) {
    }

    public JTextField(Document doc, String text, int columns) {
    }

    @BeanProperty(bound = false)
    public String getUIClassID();

    @BeanProperty(expert = true, description = "the text document model")
    public void setDocument(Document doc);

    @Override
    public boolean isValidateRoot();

    public int getHorizontalAlignment();

    @BeanProperty(preferred = true, enumerationValues = { "JTextField.LEFT", "JTextField.CENTER", "JTextField.RIGHT", "JTextField.LEADING", "JTextField.TRAILING" }, description = "Set the field alignment to LEFT, CENTER, RIGHT, LEADING (the default) or TRAILING")
    public void setHorizontalAlignment(int alignment);

    protected Document createDefaultModel();

    public int getColumns();

    @BeanProperty(bound = false, description = "the number of columns preferred for display")
    public void setColumns(int columns);

    protected int getColumnWidth();

    public Dimension getPreferredSize();

    public void setFont(Font f);

    public synchronized void addActionListener(ActionListener l);

    public synchronized void removeActionListener(ActionListener l);

    @BeanProperty(bound = false)
    public synchronized ActionListener[] getActionListeners();

    @SuppressWarnings("deprecation")
    protected void fireActionPerformed();

    public void setActionCommand(String command);

    @BeanProperty(visualUpdate = true, description = "the Action instance connected with this ActionEvent source")
    public void setAction(Action a);

    public Action getAction();

    protected void configurePropertiesFromAction(Action a);

    protected void actionPropertyChanged(Action action, String propertyName);

    protected PropertyChangeListener createActionPropertyChangeListener(Action a);

    private static class TextFieldActionPropertyChangeListener extends ActionPropertyChangeListener<JTextField> {

        protected void actionPropertyChanged(JTextField textField, Action action, PropertyChangeEvent e);
    }

    @BeanProperty(bound = false)
    public Action[] getActions();

    public void postActionEvent();

    @BeanProperty(bound = false)
    public BoundedRangeModel getHorizontalVisibility();

    public int getScrollOffset();

    public void setScrollOffset(int scrollOffset);

    public void scrollRectToVisible(Rectangle r);

    boolean hasActionListener();

    @Interned
    public static final String notifyAction;

    static class NotifyAction extends TextAction {

        public void actionPerformed(ActionEvent e);

        public boolean isEnabled();
    }

    class ScrollRepainter implements ChangeListener, Serializable {

        public void stateChanged(ChangeEvent e);
    }

    protected String paramString();

    @BeanProperty(bound = false)
    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJTextField extends AccessibleJTextComponent {

        protected AccessibleJTextField() {
        }

        public AccessibleStateSet getAccessibleStateSet();
    }
}
