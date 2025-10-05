/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.awt;

    @Positive
import org.checkerframework.checker.i18n.qual.Localized;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.ItemEvent;
    @Positive
import java.awt.event.ItemListener;
    @Positive
import java.awt.peer.CheckboxPeer;
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
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import javax.accessibility.AccessibleValue;

    @Positive
@AnnotatedFor({ "i18n" })
    @Positive
public class Checkbox extends Component implements ItemSelectable, Accessible {

    @Positive
    void setStateInternal(boolean state);

    @Positive
    public Checkbox() throws HeadlessException {
    @Positive
    }

    @Positive
    public Checkbox(String label) throws HeadlessException {
    @Positive
    }

    @Positive
    public Checkbox(String label, boolean state) throws HeadlessException {
    @Positive
    }

    @Positive
    public Checkbox(String label, boolean state, CheckboxGroup group) throws HeadlessException {
    @Positive
    }

    @Positive
    public Checkbox(String label, CheckboxGroup group, boolean state) throws HeadlessException {
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
    public boolean getState();

    @Positive
    public void setState(boolean state);

    @Positive
    public Object[] getSelectedObjects();

    @Positive
    public CheckboxGroup getCheckboxGroup();

    @Positive
    public void setCheckboxGroup(CheckboxGroup g);

    @Positive
    public synchronized void addItemListener(ItemListener l);

    @Positive
    public synchronized void removeItemListener(ItemListener l);

    @Positive
    public synchronized ItemListener[] getItemListeners();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    boolean eventEnabled(AWTEvent e);

    @Positive
    protected void processEvent(AWTEvent e);

    @Positive
    protected void processItemEvent(ItemEvent e);

    @Positive
    protected String paramString();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleAWTCheckbox extends AccessibleAWTComponent implements ItemListener, AccessibleAction, AccessibleValue {

    @Positive
        public AccessibleAWTCheckbox() {
    @Positive
        }

    @Positive
        public void itemStateChanged(ItemEvent e);

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
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }
    @Positive
}
