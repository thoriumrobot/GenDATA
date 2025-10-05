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

import java.awt.event.KeyEvent;
import java.awt.event.WindowEvent;
import java.awt.peer.FramePeer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Vector;
import javax.accessibility.AccessibleContext;
import javax.accessibility.AccessibleRole;
import javax.accessibility.AccessibleState;
import javax.accessibility.AccessibleStateSet;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import sun.awt.AWTAccessor;
import sun.awt.SunToolkit;

@AnnotatedFor({ "nullness" })
public class Frame extends Window implements MenuContainer {

    @Deprecated
    public static final int DEFAULT_CURSOR;

    @Deprecated
    public static final int CROSSHAIR_CURSOR;

    @Deprecated
    public static final int TEXT_CURSOR;

    @Deprecated
    public static final int WAIT_CURSOR;

    @Deprecated
    public static final int SW_RESIZE_CURSOR;

    @Deprecated
    public static final int SE_RESIZE_CURSOR;

    @Deprecated
    public static final int NW_RESIZE_CURSOR;

    @Deprecated
    public static final int NE_RESIZE_CURSOR;

    @Deprecated
    public static final int N_RESIZE_CURSOR;

    @Deprecated
    public static final int S_RESIZE_CURSOR;

    @Deprecated
    public static final int W_RESIZE_CURSOR;

    @Deprecated
    public static final int E_RESIZE_CURSOR;

    @Deprecated
    public static final int HAND_CURSOR;

    @Deprecated
    public static final int MOVE_CURSOR;

    public static final int NORMAL;

    public static final int ICONIFIED;

    public static final int MAXIMIZED_HORIZ;

    public static final int MAXIMIZED_VERT;

    public static final int MAXIMIZED_BOTH;

    public Frame() throws HeadlessException {
    }

    public Frame(GraphicsConfiguration gc) {
    }

    public Frame(@Nullable String title) throws HeadlessException {
    }

    public Frame(@Nullable String title, GraphicsConfiguration gc) {
    }

    String constructComponentName();

    public void addNotify();

    public String getTitle();

    public void setTitle(@Nullable String title);

    @Nullable
    public Image getIconImage();

    public void setIconImage(@Nullable Image image);

    @Nullable
    public MenuBar getMenuBar();

    public void setMenuBar(@Nullable MenuBar mb);

    public boolean isResizable();

    public void setResizable(boolean resizable);

    public synchronized void setState(int state);

    public void setExtendedState(int state);

    public synchronized int getState();

    public int getExtendedState();

    public void setMaximizedBounds(@Nullable Rectangle bounds);

    @Nullable
    public Rectangle getMaximizedBounds();

    public void setUndecorated(boolean undecorated);

    public boolean isUndecorated();

    @Override
    public void setOpacity(float opacity);

    @Override
    public void setShape(@Nullable Shape shape);

    @Override
    public void setBackground(@Nullable Color bgColor);

    public void remove(@Nullable MenuComponent m);

    public void removeNotify();

    void postProcessKeyEvent(KeyEvent e);

    protected String paramString();

    @Deprecated
    public void setCursor(int cursorType);

    @Deprecated
    public int getCursorType();

    public static Frame[] getFrames();

    public AccessibleContext getAccessibleContext();

    protected class AccessibleAWTFrame extends AccessibleAWTWindow {

        protected AccessibleAWTFrame() {
        }

        public AccessibleRole getAccessibleRole();

        public AccessibleStateSet getAccessibleStateSet();
    }
}
