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
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.peer.FramePeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Vector;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.SunToolkit;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class Frame extends Window implements MenuContainer {

    @Positive
    @Deprecated
    @Positive
    public static final int DEFAULT_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int CROSSHAIR_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int TEXT_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int WAIT_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int SW_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int SE_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int NW_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int NE_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int N_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int S_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int W_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int E_RESIZE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int HAND_CURSOR;

    @Positive
    @Deprecated
    @Positive
    public static final int MOVE_CURSOR;

    @Positive
    public static final int NORMAL;

    @Positive
    public static final int ICONIFIED;

    @Positive
    public static final int MAXIMIZED_HORIZ;

    @Positive
    public static final int MAXIMIZED_VERT;

    @Positive
    public static final int MAXIMIZED_BOTH;

    @Positive
    public Frame() throws HeadlessException {
    @Positive
    }

    @Positive
    public Frame(GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    public Frame(@Nullable String title) throws HeadlessException {
    @Positive
    }

    @Positive
    public Frame(@Nullable String title, GraphicsConfiguration gc) {
    @Positive
    }

    @Positive
    String constructComponentName();

    @Positive
    public void addNotify();

    @Positive
    public String getTitle();

    @Positive
    public void setTitle(@Nullable String title);

    @Positive
    @Nullable
    @Positive
    public Image getIconImage();

    @Positive
    public void setIconImage(@Nullable Image image);

    @Positive
    @Nullable
    @Positive
    public MenuBar getMenuBar();

    @Positive
    public void setMenuBar(@Nullable MenuBar mb);

    @Positive
    public boolean isResizable();

    @Positive
    public void setResizable(boolean resizable);

    @Positive
    public synchronized void setState(int state);

    @Positive
    public void setExtendedState(int state);

    @Positive
    public synchronized int getState();

    @Positive
    public int getExtendedState();

    @Positive
    public void setMaximizedBounds(@Nullable Rectangle bounds);

    @Positive
    @Nullable
    @Positive
    public Rectangle getMaximizedBounds();

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
    public void remove(@Nullable MenuComponent m);

    @Positive
    public void removeNotify();

    @Positive
    void postProcessKeyEvent(KeyEvent e);

    @Positive
    protected String paramString();

    @Positive
    @Deprecated
    @Positive
    public void setCursor(int cursorType);

    @Positive
    @Deprecated
    @Positive
    public int getCursorType();

    @Positive
    public static Frame[] getFrames();

    @Positive
    public AccessibleContext getAccessibleContext();

    @Positive
    protected class AccessibleAWTFrame extends AccessibleAWTWindow {

    @Positive
        protected AccessibleAWTFrame() {
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

// CFWR semantic augmentation - variant 0
