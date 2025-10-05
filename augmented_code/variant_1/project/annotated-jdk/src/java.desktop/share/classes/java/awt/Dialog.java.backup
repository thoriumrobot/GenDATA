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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.ComponentEvent;
import java.awt.event.HierarchyEvent;
import java.awt.event.InvocationEvent;
import java.awt.event.WindowEvent;
import java.awt.peer.DialogPeer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.security.AccessControlException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicLong;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import sun.awt.AWTPermissions;
import sun.awt.AppContext;
import sun.awt.SunToolkit;
import sun.awt.util.IdentityArrayList;
import sun.awt.util.IdentityLinkedList;

@AnnotatedFor({ "nullness" })
public class Dialog extends Window {

    public static enum ModalityType {

        MODELESS, DOCUMENT_MODAL, APPLICATION_MODAL, TOOLKIT_MODAL
    }

    public static final ModalityType DEFAULT_MODALITY_TYPE;

    public static enum ModalExclusionType {

        NO_EXCLUDE, APPLICATION_EXCLUDE, TOOLKIT_EXCLUDE
    }

    public Dialog(@Nullable Frame owner) {
    }

    public Dialog(@Nullable Frame owner, boolean modal) {
    }

    public Dialog(@Nullable Frame owner, @Nullable String title) {
    }

    public Dialog(@Nullable Frame owner, @Nullable String title, boolean modal) {
    }

    public Dialog(@Nullable Frame owner, @Nullable String title, boolean modal, @Nullable GraphicsConfiguration gc) {
    }

    public Dialog(@Nullable Dialog owner) {
    }

    public Dialog(@Nullable Dialog owner, @Nullable String title) {
    }

    public Dialog(@Nullable Dialog owner, @Nullable String title, boolean modal) {
    }

    public Dialog(@Nullable Dialog owner, @Nullable String title, boolean modal, GraphicsConfiguration gc) {
    }

    public Dialog(@Nullable Window owner) {
    }

    public Dialog(@Nullable Window owner, @Nullable String title) {
    }

    public Dialog(@Nullable Window owner, @Nullable ModalityType modalityType) {
    }

    public Dialog(@Nullable Window owner, @Nullable String title, @Nullable ModalityType modalityType) {
    }

    public Dialog(@Nullable Window owner, @Nullable String title, @Nullable ModalityType modalityType, @Nullable GraphicsConfiguration gc) {
    }

    String constructComponentName();

    public void addNotify();

    public boolean isModal();

    final boolean isModal_NoClientCode();

    public void setModal(boolean modal);

    public ModalityType getModalityType();

    public void setModalityType(@Nullable ModalityType type);

    @Nullable
    public String getTitle();

    public void setTitle(@Nullable String title);

    public void setVisible(boolean b);

    @Deprecated
    public void show();

    final void modalityPushed();

    final void modalityPopped();

    @Deprecated
    public void hide();

    void doDispose();

    public void toBack();

    public boolean isResizable();

    public void setResizable(boolean resizable);

    public void setUndecorated(boolean undecorated);

    public boolean isUndecorated();

    @Override
    public void setOpacity(float opacity);

    @Override
    public void setShape(@Nullable Shape shape);

    @Override
    public void setBackground(@Nullable Color bgColor);

    protected String paramString();

    void modalShow();

    void modalHide();

    boolean shouldBlock(Window w);

    void blockWindow(Window w);

    void blockWindows(java.util.List<Window> toBlock);

    void unblockWindow(Window w);

    static void checkShouldBeBlocked(Window w);

    public AccessibleContext getAccessibleContext();

    protected class AccessibleAWTDialog extends AccessibleAWTWindow {

        protected AccessibleAWTDialog() {
        }

        public AccessibleRole getAccessibleRole();

        public AccessibleStateSet getAccessibleStateSet();
    }
}
