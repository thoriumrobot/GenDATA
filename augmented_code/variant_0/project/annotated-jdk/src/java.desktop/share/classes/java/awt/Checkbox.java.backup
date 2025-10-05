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
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.peer.CheckboxPeer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.util.EventListener;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleAction;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import javax.accessibility.AccessibleValue;

@AnnotatedFor({ "i18n" })
public class Checkbox extends Component implements ItemSelectable, Accessible {

    void setStateInternal(boolean state);

    public Checkbox() throws HeadlessException {
    }

    public Checkbox(String label) throws HeadlessException {
    }

    public Checkbox(String label, boolean state) throws HeadlessException {
    }

    public Checkbox(String label, boolean state, CheckboxGroup group) throws HeadlessException {
    }

    public Checkbox(String label, CheckboxGroup group, boolean state) throws HeadlessException {
    }

    String constructComponentName();

    public void addNotify();

    @Localized
    public String getLabel();

    public void setLabel(@Localized String label);

    public boolean getState();

    public void setState(boolean state);

    public Object[] getSelectedObjects();

    public CheckboxGroup getCheckboxGroup();

    public void setCheckboxGroup(CheckboxGroup g);

    public synchronized void addItemListener(ItemListener l);

    public synchronized void removeItemListener(ItemListener l);

    public synchronized ItemListener[] getItemListeners();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    boolean eventEnabled(AWTEvent e);

    protected void processEvent(AWTEvent e);

    protected void processItemEvent(ItemEvent e);

    protected String paramString();

    public AccessibleContext getAccessibleContext();

    protected class AccessibleAWTCheckbox extends AccessibleAWTComponent implements ItemListener, AccessibleAction, AccessibleValue {

        public AccessibleAWTCheckbox() {
        }

        public void itemStateChanged(ItemEvent e);

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

        public AccessibleStateSet getAccessibleStateSet();
    }
}
