/*
 * Copyright (c) 2002, 2014, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.X11;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import jdk.internal.misc.Unsafe;
import java.util.HashMap;

public final class XAtom {

    public static final long XA_PRIMARY;

    public static final long XA_SECONDARY;

    public static final long XA_ARC;

    public static final long XA_ATOM;

    public static final long XA_BITMAP;

    public static final long XA_CARDINAL;

    public static final long XA_COLORMAP;

    public static final long XA_CURSOR;

    public static final long XA_CUT_BUFFER0;

    public static final long XA_CUT_BUFFER1;

    public static final long XA_CUT_BUFFER2;

    public static final long XA_CUT_BUFFER3;

    public static final long XA_CUT_BUFFER4;

    public static final long XA_CUT_BUFFER5;

    public static final long XA_CUT_BUFFER6;

    public static final long XA_CUT_BUFFER7;

    public static final long XA_DRAWABLE;

    public static final long XA_FONT;

    public static final long XA_INTEGER;

    public static final long XA_PIXMAP;

    public static final long XA_POINT;

    public static final long XA_RECTANGLE;

    public static final long XA_RESOURCE_MANAGER;

    public static final long XA_RGB_COLOR_MAP;

    public static final long XA_RGB_BEST_MAP;

    public static final long XA_RGB_BLUE_MAP;

    public static final long XA_RGB_DEFAULT_MAP;

    public static final long XA_RGB_GRAY_MAP;

    public static final long XA_RGB_GREEN_MAP;

    public static final long XA_RGB_RED_MAP;

    public static final long XA_STRING;

    public static final long XA_VISUALID;

    public static final long XA_WINDOW;

    public static final long XA_WM_COMMAND;

    public static final long XA_WM_HINTS;

    public static final long XA_WM_CLIENT_MACHINE;

    public static final long XA_WM_ICON_NAME;

    public static final long XA_WM_ICON_SIZE;

    public static final long XA_WM_NAME;

    public static final long XA_WM_NORMAL_HINTS;

    public static final long XA_WM_SIZE_HINTS;

    public static final long XA_WM_ZOOM_HINTS;

    public static final long XA_MIN_SPACE;

    public static final long XA_NORM_SPACE;

    public static final long XA_MAX_SPACE;

    public static final long XA_END_SPACE;

    public static final long XA_SUPERSCRIPT_X;

    public static final long XA_SUPERSCRIPT_Y;

    public static final long XA_SUBSCRIPT_X;

    public static final long XA_SUBSCRIPT_Y;

    public static final long XA_UNDERLINE_POSITION;

    public static final long XA_UNDERLINE_THICKNESS;

    public static final long XA_STRIKEOUT_ASCENT;

    public static final long XA_STRIKEOUT_DESCENT;

    public static final long XA_ITALIC_ANGLE;

    public static final long XA_X_HEIGHT;

    public static final long XA_QUAD_WIDTH;

    public static final long XA_WEIGHT;

    public static final long XA_POINT_SIZE;

    public static final long XA_RESOLUTION;

    public static final long XA_COPYRIGHT;

    public static final long XA_NOTICE;

    public static final long XA_FONT_NAME;

    public static final long XA_FAMILY_NAME;

    public static final long XA_FULL_NAME;

    public static final long XA_CAP_HEIGHT;

    public static final long XA_WM_CLASS;

    public static final long XA_WM_TRANSIENT_FOR;

    public static final long XA_LAST_PREDEFINED;

    static void register(XAtom at);

    static XAtom lookup(long atom);

    static XAtom lookup(String name);

    static XAtom get(long atom);

    public static XAtom get(String name);

    public String getName();

    static String asString(long atom);

    void register();

    public String toString();

    public XAtom(String name, boolean autoIntern) {
    }

    public XAtom(long display, long atom) {
    }

    public XAtom() {
    }

    public void setProperty(long window, String str);

    public void setPropertyUTF8(long window, String str);

    public void setProperty8(long window, String str);

    public String getProperty(long window);

    public long get32Property(long window, long property_type);

    public long getCard32Property(XBaseWindow window);

    public void setCard32Property(long window, long value);

    public void setCard32Property(XBaseWindow window, long value);

    public boolean getAtomData(long window, long data_ptr, int length);

    public boolean getAtomData(long window, long type, long data_ptr, int length);

    public void setAtomData(long window, long data_ptr, int length);

    public void setAtomData(long window, long type, long data_ptr, int length);

    public void setAtomData8(long window, long type, long data_ptr, int length);

    public void DeleteProperty(long window);

    public void DeleteProperty(XBaseWindow window);

    public void setAtomData(long window, long property_type, byte[] data);

    public byte[] getByteArrayProperty(long window, long property_type);

    public void intern(boolean onlyIfExists);

    public boolean isInterned();

    public void setValues(long display, String name, long atom);

    static int getAtomSize();

    XAtom[] getAtomListProperty(long window);

    XAtomList getAtomListPropertyList(long window);

    XAtomList getAtomListPropertyList(XBaseWindow window);

    XAtom[] getAtomListProperty(XBaseWindow window);

    void setAtomListProperty(long window, XAtom[] atoms);

    void setAtomListProperty(long window, XAtomList atoms);

    public void setAtomListProperty(XBaseWindow window, XAtom[] atoms);

    public void setAtomListProperty(XBaseWindow window, XAtomList atoms);

    long getAtom();

    void putAtom(long ptr);

    static long getAtom(long ptr);

    static long toData(XAtom[] atoms);

    void checkWindow(long window);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();

    public void setWindowProperty(long window, long window_value);

    public void setWindowProperty(XBaseWindow window, XBaseWindow window_value);

    public long getWindowProperty(long window);
}
