/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.prefs;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.security.AccessController;
import java.security.Permission;
import java.security.PrivilegedAction;
import java.util.Iterator;
import java.util.ServiceLoader;
import java.util.ServiceConfigurationError;
import java.lang.RuntimePermission;
import java.lang.Integer;
import java.lang.Long;
import java.lang.Float;
import java.lang.Double;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public abstract class Preferences {

    public static final int MAX_KEY_LENGTH;

    public static final int MAX_VALUE_LENGTH;

    public static final int MAX_NAME_LENGTH;

    public static Preferences userNodeForPackage(Class<?> c);

    public static Preferences systemNodeForPackage(Class<?> c);

    public static Preferences userRoot();

    public static Preferences systemRoot();

    protected Preferences() {
    }

    public abstract void put(String key, String value);

    @PolyNull
    public abstract String get(String key, @PolyNull String def);

    public abstract void remove(String key);

    public abstract void clear() throws BackingStoreException;

    public abstract void putInt(String key, int value);

    public abstract int getInt(String key, int def);

    public abstract void putLong(String key, long value);

    public abstract long getLong(String key, long def);

    public abstract void putBoolean(String key, boolean value);

    public abstract boolean getBoolean(String key, boolean def);

    public abstract void putFloat(String key, float value);

    public abstract float getFloat(String key, float def);

    public abstract void putDouble(String key, double value);

    public abstract double getDouble(String key, double def);

    public abstract void putByteArray(String key, byte[] value);

    public abstract byte @PolyNull [] getByteArray(String key, byte @PolyNull [] def);

    public abstract String[] keys() throws BackingStoreException;

    public abstract String[] childrenNames() throws BackingStoreException;

    @Nullable
    public abstract Preferences parent();

    public abstract Preferences node(String pathName);

    public abstract boolean nodeExists(String pathName) throws BackingStoreException;

    public abstract void removeNode() throws BackingStoreException;

    public abstract String name();

    public abstract String absolutePath();

    public abstract boolean isUserNode();

    public abstract String toString();

    public abstract void flush() throws BackingStoreException;

    public abstract void sync() throws BackingStoreException;

    public abstract void addPreferenceChangeListener(PreferenceChangeListener pcl);

    public abstract void removePreferenceChangeListener(PreferenceChangeListener pcl);

    public abstract void addNodeChangeListener(NodeChangeListener ncl);

    public abstract void removeNodeChangeListener(NodeChangeListener ncl);

    public abstract void exportNode(OutputStream os) throws IOException, BackingStoreException;

    public abstract void exportSubtree(OutputStream os) throws IOException, BackingStoreException;

    public static void importPreferences(InputStream is) throws IOException, InvalidPreferencesFormatException;
}
