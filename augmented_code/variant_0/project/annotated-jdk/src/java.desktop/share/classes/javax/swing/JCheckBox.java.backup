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
import java.beans.BeanProperty;
import java.beans.JavaBean;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.swing.plaf.ButtonUI;

@AnnotatedFor({ "interning" })
@JavaBean(description = "A component which can be selected or deselected.")
@SwingContainer(false)
@SuppressWarnings("serial")
public class JCheckBox extends JToggleButton implements Accessible {

    @Interned
    public static final String BORDER_PAINTED_FLAT_CHANGED_PROPERTY;

    public JCheckBox() {
    }

    public JCheckBox(Icon icon) {
    }

    public JCheckBox(Icon icon, boolean selected) {
    }

    public JCheckBox(String text) {
    }

    public JCheckBox(Action a) {
    }

    public JCheckBox(String text, boolean selected) {
    }

    public JCheckBox(String text, Icon icon) {
    }

    public JCheckBox(String text, Icon icon, boolean selected) {
    }

    @BeanProperty(visualUpdate = true, description = "Whether the border is painted flat.")
    public void setBorderPaintedFlat(boolean b);

    public boolean isBorderPaintedFlat();

    public void updateUI();

    @BeanProperty(bound = false, expert = true, description = "A string that specifies the name of the L&F class")
    public String getUIClassID();

    void setIconFromAction(Action a);

    protected String paramString();

    @BeanProperty(bound = false, expert = true, description = "The AccessibleContext associated with this CheckBox.")
    public AccessibleContext getAccessibleContext();

    @SuppressWarnings("serial")
    protected class AccessibleJCheckBox extends AccessibleJToggleButton {

        protected AccessibleJCheckBox() {
        }

        public AccessibleRole getAccessibleRole();
    }
}
