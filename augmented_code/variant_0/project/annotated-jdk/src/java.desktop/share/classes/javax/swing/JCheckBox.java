/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
