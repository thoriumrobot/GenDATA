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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.KeyEvent;
import java.io.Serial;

@Deprecated()
@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Event implements java.io.Serializable {

    public static final int SHIFT_MASK;

    public static final int CTRL_MASK;

    public static final int META_MASK;

    public static final int ALT_MASK;

    public static final int HOME;

    public static final int END;

    public static final int PGUP;

    public static final int PGDN;

    public static final int UP;

    public static final int DOWN;

    public static final int LEFT;

    public static final int RIGHT;

    public static final int F1;

    public static final int F2;

    public static final int F3;

    public static final int F4;

    public static final int F5;

    public static final int F6;

    public static final int F7;

    public static final int F8;

    public static final int F9;

    public static final int F10;

    public static final int F11;

    public static final int F12;

    public static final int PRINT_SCREEN;

    public static final int SCROLL_LOCK;

    public static final int CAPS_LOCK;

    public static final int NUM_LOCK;

    public static final int PAUSE;

    public static final int INSERT;

    public static final int ENTER;

    public static final int BACK_SPACE;

    public static final int TAB;

    public static final int ESCAPE;

    public static final int DELETE;

    public static final int WINDOW_DESTROY;

    public static final int WINDOW_EXPOSE;

    public static final int WINDOW_ICONIFY;

    public static final int WINDOW_DEICONIFY;

    public static final int WINDOW_MOVED;

    public static final int KEY_PRESS;

    public static final int KEY_RELEASE;

    public static final int KEY_ACTION;

    public static final int KEY_ACTION_RELEASE;

    public static final int MOUSE_DOWN;

    public static final int MOUSE_UP;

    public static final int MOUSE_MOVE;

    public static final int MOUSE_ENTER;

    public static final int MOUSE_EXIT;

    public static final int MOUSE_DRAG;

    public static final int SCROLL_LINE_UP;

    public static final int SCROLL_LINE_DOWN;

    public static final int SCROLL_PAGE_UP;

    public static final int SCROLL_PAGE_DOWN;

    public static final int SCROLL_ABSOLUTE;

    public static final int SCROLL_BEGIN;

    public static final int SCROLL_END;

    public static final int LIST_SELECT;

    public static final int LIST_DESELECT;

    public static final int ACTION_EVENT;

    public static final int LOAD_FILE;

    public static final int SAVE_FILE;

    public static final int GOT_FOCUS;

    public static final int LOST_FOCUS;

    @SuppressWarnings("serial")
    public Object target;

    public long when;

    public int id;

    public int x;

    public int y;

    public int key;

    public int modifiers;

    public int clickCount;

    @SuppressWarnings("serial")
    public Object arg;

    public Event evt;

    public Event(Object target, long when, int id, int x, int y, int key, int modifiers, Object arg) {
    }

    public Event(Object target, long when, int id, int x, int y, int key, int modifiers) {
    }

    public Event(Object target, int id, Object arg) {
    }

    public void translate(int dx, int dy);

    public boolean shiftDown();

    public boolean controlDown();

    public boolean metaDown();

    void consume();

    boolean isConsumed();

    static int getOldEventKey(KeyEvent e);

    char getKeyEventChar();

    protected String paramString();

    public String toString();
}
