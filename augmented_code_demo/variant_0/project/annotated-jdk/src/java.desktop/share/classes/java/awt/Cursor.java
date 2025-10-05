/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.beans.ConstructorProperties;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Properties;
    @Positive
import java.util.StringTokenizer;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.util.logging.PlatformLogger;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Cursor implements java.io.Serializable {

    @Positive
    public static final int DEFAULT_CURSOR;

    @Positive
    public static final int CROSSHAIR_CURSOR;

    @Positive
    public static final int TEXT_CURSOR;

    @Positive
    public static final int WAIT_CURSOR;

    @Positive
    public static final int SW_RESIZE_CURSOR;

    @Positive
    public static final int SE_RESIZE_CURSOR;

    @Positive
    public static final int NW_RESIZE_CURSOR;

    @Positive
    public static final int NE_RESIZE_CURSOR;

    @Positive
    public static final int N_RESIZE_CURSOR;

    @Positive
    public static final int S_RESIZE_CURSOR;

    @Positive
    public static final int W_RESIZE_CURSOR;

    @Positive
    public static final int E_RESIZE_CURSOR;

    @Positive
    public static final int HAND_CURSOR;

    @Positive
    public static final int MOVE_CURSOR;

    @Positive
    @Deprecated
    @Positive
    protected static Cursor[] predefined;

    @Positive
    public static final int CUSTOM_CURSOR;

    @Positive
    static class CursorDisposer implements sun.java2d.DisposerRecord {

    @Positive
        public CursorDisposer(long pData) {
    @Positive
        }

    @Positive
        public void dispose();
    @Positive
    }

    @Positive
    protected String name;

    @Positive
    public static Cursor getPredefinedCursor(int type);

    @Positive
    public static Cursor getSystemCustomCursor(final String name) throws AWTException, HeadlessException;

    @Positive
    public static Cursor getDefaultCursor();

    @Positive
    @ConstructorProperties({ "type" })
    @Positive
    public Cursor(int type) {
    @Positive
    }

    @Positive
    protected Cursor(String name) {
    @Positive
    }

    @Positive
    public int getType();

    @Positive
    public String getName();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 0
