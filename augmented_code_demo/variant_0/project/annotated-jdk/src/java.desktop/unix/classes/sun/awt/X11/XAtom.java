/*
    @Positive
 * Copyright (c) 2002, 2014, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.X11;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import java.util.HashMap;

    @Positive
public final class XAtom {

    @Positive
    public static final long XA_PRIMARY;

    @Positive
    public static final long XA_SECONDARY;

    @Positive
    public static final long XA_ARC;

    @Positive
    public static final long XA_ATOM;

    @Positive
    public static final long XA_BITMAP;

    @Positive
    public static final long XA_CARDINAL;

    @Positive
    public static final long XA_COLORMAP;

    @Positive
    public static final long XA_CURSOR;

    @Positive
    public static final long XA_CUT_BUFFER0;

    @Positive
    public static final long XA_CUT_BUFFER1;

    @Positive
    public static final long XA_CUT_BUFFER2;

    @Positive
    public static final long XA_CUT_BUFFER3;

    @Positive
    public static final long XA_CUT_BUFFER4;

    @Positive
    public static final long XA_CUT_BUFFER5;

    @Positive
    public static final long XA_CUT_BUFFER6;

    @Positive
    public static final long XA_CUT_BUFFER7;

    @Positive
    public static final long XA_DRAWABLE;

    @Positive
    public static final long XA_FONT;

    @Positive
    public static final long XA_INTEGER;

    @Positive
    public static final long XA_PIXMAP;

    @Positive
    public static final long XA_POINT;

    @Positive
    public static final long XA_RECTANGLE;

    @Positive
    public static final long XA_RESOURCE_MANAGER;

    @Positive
    public static final long XA_RGB_COLOR_MAP;

    @Positive
    public static final long XA_RGB_BEST_MAP;

    @Positive
    public static final long XA_RGB_BLUE_MAP;

    @Positive
    public static final long XA_RGB_DEFAULT_MAP;

    @Positive
    public static final long XA_RGB_GRAY_MAP;

    @Positive
    public static final long XA_RGB_GREEN_MAP;

    @Positive
    public static final long XA_RGB_RED_MAP;

    @Positive
    public static final long XA_STRING;

    @Positive
    public static final long XA_VISUALID;

    @Positive
    public static final long XA_WINDOW;

    @Positive
    public static final long XA_WM_COMMAND;

    @Positive
    public static final long XA_WM_HINTS;

    @Positive
    public static final long XA_WM_CLIENT_MACHINE;

    @Positive
    public static final long XA_WM_ICON_NAME;

    @Positive
    public static final long XA_WM_ICON_SIZE;

    @Positive
    public static final long XA_WM_NAME;

    @Positive
    public static final long XA_WM_NORMAL_HINTS;

    @Positive
    public static final long XA_WM_SIZE_HINTS;

    @Positive
    public static final long XA_WM_ZOOM_HINTS;

    @Positive
    public static final long XA_MIN_SPACE;

    @Positive
    public static final long XA_NORM_SPACE;

    @Positive
    public static final long XA_MAX_SPACE;

    @Positive
    public static final long XA_END_SPACE;

    @Positive
    public static final long XA_SUPERSCRIPT_X;

    @Positive
    public static final long XA_SUPERSCRIPT_Y;

    @Positive
    public static final long XA_SUBSCRIPT_X;

    @Positive
    public static final long XA_SUBSCRIPT_Y;

    @Positive
    public static final long XA_UNDERLINE_POSITION;

    @Positive
    public static final long XA_UNDERLINE_THICKNESS;

    @Positive
    public static final long XA_STRIKEOUT_ASCENT;

    @Positive
    public static final long XA_STRIKEOUT_DESCENT;

    @Positive
    public static final long XA_ITALIC_ANGLE;

    @Positive
    public static final long XA_X_HEIGHT;

    @Positive
    public static final long XA_QUAD_WIDTH;

    @Positive
    public static final long XA_WEIGHT;

    @Positive
    public static final long XA_POINT_SIZE;

    @Positive
    public static final long XA_RESOLUTION;

    @Positive
    public static final long XA_COPYRIGHT;

    @Positive
    public static final long XA_NOTICE;

    @Positive
    public static final long XA_FONT_NAME;

    @Positive
    public static final long XA_FAMILY_NAME;

    @Positive
    public static final long XA_FULL_NAME;

    @Positive
    public static final long XA_CAP_HEIGHT;

    @Positive
    public static final long XA_WM_CLASS;

    @Positive
    public static final long XA_WM_TRANSIENT_FOR;

    @Positive
    public static final long XA_LAST_PREDEFINED;

    @Positive
    static void register(XAtom at);

    @Positive
    static XAtom lookup(long atom);

    @Positive
    static XAtom lookup(String name);

    @Positive
    static XAtom get(long atom);

    @Positive
    public static XAtom get(String name);

    @Positive
    public String getName();

    @Positive
    static String asString(long atom);

    @Positive
    void register();

    @Positive
    public String toString();

    @Positive
    public XAtom(String name, boolean autoIntern) {
    @Positive
    }

    @Positive
    public XAtom(long display, long atom) {
    @Positive
    }

    @Positive
    public XAtom() {
    @Positive
    }

    @Positive
    public void setProperty(long window, String str);

    @Positive
    public void setPropertyUTF8(long window, String str);

    @Positive
    public void setProperty8(long window, String str);

    @Positive
    public String getProperty(long window);

    @Positive
    public long get32Property(long window, long property_type);

    @Positive
    public long getCard32Property(XBaseWindow window);

    @Positive
    public void setCard32Property(long window, long value);

    @Positive
    public void setCard32Property(XBaseWindow window, long value);

    @Positive
    public boolean getAtomData(long window, long data_ptr, int length);

    @Positive
    public boolean getAtomData(long window, long type, long data_ptr, int length);

    @Positive
    public void setAtomData(long window, long data_ptr, int length);

    @Positive
    public void setAtomData(long window, long type, long data_ptr, int length);

    @Positive
    public void setAtomData8(long window, long type, long data_ptr, int length);

    @Positive
    public void DeleteProperty(long window);

    @Positive
    public void DeleteProperty(XBaseWindow window);

    @Positive
    public void setAtomData(long window, long property_type, byte[] data);

    @Positive
    public byte[] getByteArrayProperty(long window, long property_type);

    @Positive
    public void intern(boolean onlyIfExists);

    @Positive
    public boolean isInterned();

    @Positive
    public void setValues(long display, String name, long atom);

    @Positive
    static int getAtomSize();

    @Positive
    XAtom[] getAtomListProperty(long window);

    @Positive
    XAtomList getAtomListPropertyList(long window);

    @Positive
    XAtomList getAtomListPropertyList(XBaseWindow window);

    @Positive
    XAtom[] getAtomListProperty(XBaseWindow window);

    @Positive
    void setAtomListProperty(long window, XAtom[] atoms);

    @Positive
    void setAtomListProperty(long window, XAtomList atoms);

    @Positive
    public void setAtomListProperty(XBaseWindow window, XAtom[] atoms);

    @Positive
    public void setAtomListProperty(XBaseWindow window, XAtomList atoms);

    @Positive
    long getAtom();

    @Positive
    void putAtom(long ptr);

    @Positive
    static long getAtom(long ptr);

    @Positive
    static long toData(XAtom[] atoms);

    @Positive
    void checkWindow(long window);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    public void setWindowProperty(long window, long window_value);

    @Positive
    public void setWindowProperty(XBaseWindow window, XBaseWindow window_value);

    @Positive
    public long getWindowProperty(long window);
    @Positive
}

// CFWR semantic augmentation - variant 0
