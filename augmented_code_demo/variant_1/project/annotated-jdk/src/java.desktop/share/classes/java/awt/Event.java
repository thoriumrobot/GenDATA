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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.io.Serial;

    @Positive
@Deprecated()
    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Event implements java.io.Serializable {

    @Positive
    public static final int SHIFT_MASK;

    @Positive
    public static final int CTRL_MASK;

    @Positive
    public static final int META_MASK;

    @Positive
    public static final int ALT_MASK;

    @Positive
    public static final int HOME;

    @Positive
    public static final int END;

    @Positive
    public static final int PGUP;

    @Positive
    public static final int PGDN;

    @Positive
    public static final int UP;

    @Positive
    public static final int DOWN;

    @Positive
    public static final int LEFT;

    @Positive
    public static final int RIGHT;

    @Positive
    public static final int F1;

    @Positive
    public static final int F2;

    @Positive
    public static final int F3;

    @Positive
    public static final int F4;

    @Positive
    public static final int F5;

    @Positive
    public static final int F6;

    @Positive
    public static final int F7;

    @Positive
    public static final int F8;

    @Positive
    public static final int F9;

    @Positive
    public static final int F10;

    @Positive
    public static final int F11;

    @Positive
    public static final int F12;

    @Positive
    public static final int PRINT_SCREEN;

    @Positive
    public static final int SCROLL_LOCK;

    @Positive
    public static final int CAPS_LOCK;

    @Positive
    public static final int NUM_LOCK;

    @Positive
    public static final int PAUSE;

    @Positive
    public static final int INSERT;

    @Positive
    public static final int ENTER;

    @Positive
    public static final int BACK_SPACE;

    @Positive
    public static final int TAB;

    @Positive
    public static final int ESCAPE;

    @Positive
    public static final int DELETE;

    @Positive
    public static final int WINDOW_DESTROY;

    @Positive
    public static final int WINDOW_EXPOSE;

    @Positive
    public static final int WINDOW_ICONIFY;

    @Positive
    public static final int WINDOW_DEICONIFY;

    @Positive
    public static final int WINDOW_MOVED;

    @Positive
    public static final int KEY_PRESS;

    @Positive
    public static final int KEY_RELEASE;

    @Positive
    public static final int KEY_ACTION;

    @Positive
    public static final int KEY_ACTION_RELEASE;

    @Positive
    public static final int MOUSE_DOWN;

    @Positive
    public static final int MOUSE_UP;

    @Positive
    public static final int MOUSE_MOVE;

    @Positive
    public static final int MOUSE_ENTER;

    @Positive
    public static final int MOUSE_EXIT;

    @Positive
    public static final int MOUSE_DRAG;

    @Positive
    public static final int SCROLL_LINE_UP;

    @Positive
    public static final int SCROLL_LINE_DOWN;

    @Positive
    public static final int SCROLL_PAGE_UP;

    @Positive
    public static final int SCROLL_PAGE_DOWN;

    @Positive
    public static final int SCROLL_ABSOLUTE;

    @Positive
    public static final int SCROLL_BEGIN;

    @Positive
    public static final int SCROLL_END;

    @Positive
    public static final int LIST_SELECT;

    @Positive
    public static final int LIST_DESELECT;

    @Positive
    public static final int ACTION_EVENT;

    @Positive
    public static final int LOAD_FILE;

    @Positive
    public static final int SAVE_FILE;

    @Positive
    public static final int GOT_FOCUS;

    @Positive
    public static final int LOST_FOCUS;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public Object target;

    @Positive
    public long when;

    @Positive
    public int id;

    @Positive
    public int x;

    @Positive
    public int y;

    @Positive
    public int key;

    @Positive
    public int modifiers;

    @Positive
    public int clickCount;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public Object arg;

    @Positive
    public Event evt;

    @Positive
    public Event(Object target, long when, int id, int x, int y, int key, int modifiers, Object arg) {
    @Positive
    }

    @Positive
    public Event(Object target, long when, int id, int x, int y, int key, int modifiers) {
    @Positive
    }

    @Positive
    public Event(Object target, int id, Object arg) {
    @Positive
    }

    @Positive
    public void translate(int dx, int dy);

    @Positive
    public boolean shiftDown();

    @Positive
    public boolean controlDown();

    @Positive
    public boolean metaDown();

    @Positive
    void consume();

    @Positive
    boolean isConsumed();

    @Positive
    static int getOldEventKey(KeyEvent e);

    @Positive
    char getKeyEventChar();

    @Positive
    protected String paramString();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
