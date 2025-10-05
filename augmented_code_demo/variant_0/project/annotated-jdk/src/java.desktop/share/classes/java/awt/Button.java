/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.i18n.qual.Localized;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.peer.ButtonPeer;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.util.EventListener;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleAction;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleValue;

    @Positive
@AnnotatedFor({ "i18n" })
    @Positive
public class Button extends Component implements Accessible {

    @Positive
    public Button() throws HeadlessException {
    @Positive
    }

    @Positive
    public Button(String label) throws HeadlessException {
    @Positive
    }

    @Positive
    String constructComponentName();

    @Positive
    public void addNotify();

    @Positive
    @Localized
    @Positive
    public String getLabel();

    @Positive
    public void setLabel(@Localized String label);

    @Positive
    public void setActionCommand(String command);

    @Positive
    public String getActionCommand();

    @Positive
    public synchronized void addActionListener(ActionListener l);

    @Positive
    public synchronized void removeActionListener(ActionListener l);

    @Positive
    public synchronized ActionListener[] getActionListeners();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected void processActionEvent(ActionEvent e);

    @Positive
    protected String paramString();

    @Positive
    @BeanProperty(expert = true, description = "The AccessibleContext associated with this Button.")
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleAWTButton extends AccessibleAWTComponent implements AccessibleAction, AccessibleValue {

    @Positive
        protected AccessibleAWTButton() {
    @Positive
        }

    @Positive
        public String getAccessibleName();

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
        public AccessibleRole getAccessibleRole();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
