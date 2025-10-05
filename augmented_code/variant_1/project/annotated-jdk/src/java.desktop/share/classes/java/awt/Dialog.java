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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.HierarchyEvent;
    @Positive
import java.awt.event.InvocationEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.peer.DialogPeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.security.AccessControlException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import sun.awt.AWTPermissions;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.util.IdentityArrayList;
    @Positive
import sun.awt.util.IdentityLinkedList;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class Dialog extends Window {

    @Positive
    public static enum ModalityType {

    @Positive
        MODELESS, DOCUMENT_MODAL, APPLICATION_MODAL, TOOLKIT_MODAL
    @Positive
    }

    @Positive
    public static final ModalityType DEFAULT_MODALITY_TYPE;

    @Positive
    public static enum ModalExclusionType {

    @Positive
        NO_EXCLUDE, APPLICATION_EXCLUDE, TOOLKIT_EXCLUDE
    @Positive
    }

    @Positive
    public Dialog(@Nullable Frame owner) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Frame owner, boolean modal) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Frame owner, @Nullable String title) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Frame owner, @Nullable String title, boolean modal) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Frame owner, @Nullable String title, boolean modal, @Nullable GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Dialog owner) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Dialog owner, @Nullable String title) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Dialog owner, @Nullable String title, boolean modal) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Dialog owner, @Nullable String title, boolean modal, GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Window owner) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Window owner, @Nullable String title) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Window owner, @Nullable ModalityType modalityType) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Window owner, @Nullable String title, @Nullable ModalityType modalityType) {
    @Positive
    }

    @Positive
    public Dialog(@Nullable Window owner, @Nullable String title, @Nullable ModalityType modalityType, @Nullable GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    String constructComponentName();

    @Positive
    public void addNotify();

    @Positive
    public boolean isModal();

    @Positive
    final boolean isModal_NoClientCode();

    @Positive
    public void setModal(boolean modal);

    @Positive
    public ModalityType getModalityType();

    @Positive
    public void setModalityType(@Nullable ModalityType type);

    @Positive
    @Nullable
    @Positive
    public String getTitle();

    @Positive
    public void setTitle(@Nullable String title);

    @Positive
    public void setVisible(boolean b);

    @Positive
    @Deprecated
    @Positive
    public void show();

    @Positive
    final void modalityPushed();

    @Positive
    final void modalityPopped();

    @Positive
    @Deprecated
    @Positive
    public void hide();

    @Positive
    void doDispose();

    @Positive
    public void toBack();

    @Positive
    public boolean isResizable();

    @Positive
    public void setResizable(boolean resizable);

    @Positive
    public void setUndecorated(boolean undecorated);

    @Positive
    public boolean isUndecorated();

    @Positive
    @Override
    @Positive
    public void setOpacity(float opacity);

    @Positive
    @Override
    @Positive
    public void setShape(@Nullable Shape shape);

    @Positive
    @Override
    @Positive
    public void setBackground(@Nullable Color bgColor);

    @Positive
    protected String paramString();

    @Positive
    void modalShow();

    @Positive
    void modalHide();

    @Positive
    boolean shouldBlock(Window w);

    @Positive
    void blockWindow(Window w);

    @Positive
    void blockWindows(java.util.List<Window> toBlock);

    @Positive
    void unblockWindow(Window w);

    @Positive
    static void checkShouldBeBlocked(Window w);

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleAWTDialog extends AccessibleAWTWindow {

    @Positive
        protected AccessibleAWTDialog() {
    @Positive
        }

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();
    @Positive
    }
    @Positive
}
