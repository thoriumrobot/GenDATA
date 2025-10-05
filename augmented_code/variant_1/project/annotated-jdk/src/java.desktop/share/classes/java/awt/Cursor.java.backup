/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.beans.ConstructorProperties;
import java.io.InputStream;
import java.io.Serial;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.PrivilegedExceptionAction;
import java.util.Hashtable;
import java.util.Properties;
import java.util.StringTokenizer;
import sun.awt.AWTAccessor;
import sun.util.logging.PlatformLogger;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Cursor implements java.io.Serializable {

    public static final int DEFAULT_CURSOR;

    public static final int CROSSHAIR_CURSOR;

    public static final int TEXT_CURSOR;

    public static final int WAIT_CURSOR;

    public static final int SW_RESIZE_CURSOR;

    public static final int SE_RESIZE_CURSOR;

    public static final int NW_RESIZE_CURSOR;

    public static final int NE_RESIZE_CURSOR;

    public static final int N_RESIZE_CURSOR;

    public static final int S_RESIZE_CURSOR;

    public static final int W_RESIZE_CURSOR;

    public static final int E_RESIZE_CURSOR;

    public static final int HAND_CURSOR;

    public static final int MOVE_CURSOR;

    @Deprecated
    protected static Cursor[] predefined;

    public static final int CUSTOM_CURSOR;

    static class CursorDisposer implements sun.java2d.DisposerRecord {

        public CursorDisposer(long pData) {
        }

        public void dispose();
    }

    protected String name;

    public static Cursor getPredefinedCursor(int type);

    public static Cursor getSystemCustomCursor(final String name) throws AWTException, HeadlessException;

    public static Cursor getDefaultCursor();

    @ConstructorProperties({ "type" })
    public Cursor(int type) {
    }

    protected Cursor(String name) {
    }

    public int getType();

    public String getName();

    public String toString();
}
