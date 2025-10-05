/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.i18n.qual.Localized;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.peer.ButtonPeer;
import java.beans.BeanProperty;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.util.EventListener;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleAction;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleValue;

@AnnotatedFor({ "i18n" })
public class Button extends Component implements Accessible {

    public Button() throws HeadlessException {
    }

    public Button(String label) throws HeadlessException {
    }

    String constructComponentName();

    public void addNotify();

    @Localized
    public String getLabel();

    public void setLabel(@Localized String label);

    public void setActionCommand(String command);

    public String getActionCommand();

    public synchronized void addActionListener(ActionListener l);

    public synchronized void removeActionListener(ActionListener l);

    public synchronized ActionListener[] getActionListeners();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    boolean eventEnabled(AWTEvent e);

    protected void processEvent(AWTEvent e);

    protected void processActionEvent(ActionEvent e);

    protected String paramString();

    @BeanProperty(expert = true, description = "The AccessibleContext associated with this Button.")
    public AccessibleContext getAccessibleContext();

    protected class AccessibleAWTButton extends AccessibleAWTComponent implements AccessibleAction, AccessibleValue {

        protected AccessibleAWTButton() {
        }

        public String getAccessibleName();

        public AccessibleAction getAccessibleAction();

        public AccessibleValue getAccessibleValue();

        public int getAccessibleActionCount();

        public String getAccessibleActionDescription(int i);

        public boolean doAccessibleAction(int i);

        public Number getCurrentAccessibleValue();

        public boolean setCurrentAccessibleValue(Number n);

        public Number getMinimumAccessibleValue();

        public Number getMaximumAccessibleValue();

        public AccessibleRole getAccessibleRole();
    }
}
