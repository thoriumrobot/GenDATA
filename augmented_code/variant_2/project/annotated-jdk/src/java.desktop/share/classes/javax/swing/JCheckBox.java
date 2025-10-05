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
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.swing.plaf.ButtonUI;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@JavaBean(description = "A component which can be selected or deselected.")
    @Positive
@SwingContainer(false)
    @Positive
@SuppressWarnings("serial")
    @Positive
public class JCheckBox extends JToggleButton implements Accessible {

    @Positive
    @Interned
    @Positive
    public static final String BORDER_PAINTED_FLAT_CHANGED_PROPERTY;

    @Positive
    public JCheckBox() {
    @Positive
    }

    @Positive
    public JCheckBox(Icon icon) {
    @Positive
    }

    @Positive
    public JCheckBox(Icon icon, boolean selected) {
    @Positive
    }

    @Positive
    public JCheckBox(String text) {
    @Positive
    }

    @Positive
    public JCheckBox(Action a) {
    @Positive
    }

    @Positive
    public JCheckBox(String text, boolean selected) {
    @Positive
    }

    @Positive
    public JCheckBox(String text, Icon icon) {
    @Positive
    }

    @Positive
    public JCheckBox(String text, Icon icon, boolean selected) {
    @Positive
    }

    @Positive
    @BeanProperty(visualUpdate = true, description = "Whether the border is painted flat.")
    @Positive
    public void setBorderPaintedFlat(boolean b);

    @Positive
    public boolean isBorderPaintedFlat();

    @Positive
    public void updateUI();

    @Positive
    @BeanProperty(bound = false, expert = true, description = "A string that specifies the name of the L&F class")
    @Positive
    public String getUIClassID();

    @Positive
    void setIconFromAction(Action a);

    @Positive
    protected String paramString();

    @Positive
    @BeanProperty(bound = false, expert = true, description = "The AccessibleContext associated with this CheckBox.")
    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected class AccessibleJCheckBox extends AccessibleJToggleButton {

    @Positive
        protected AccessibleJCheckBox() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();
    @Positive
    }
    @Positive
}
