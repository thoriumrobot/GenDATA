/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.prefs;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.ServiceConfigurationError;
    @Positive
import java.lang.RuntimePermission;
    @Positive
import java.lang.Integer;
    @Positive
import java.lang.Long;
    @Positive
import java.lang.Float;
    @Positive
import java.lang.Double;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Preferences {

    @Positive
    public static final int MAX_KEY_LENGTH;

    @Positive
    public static final int MAX_VALUE_LENGTH;

    @Positive
    public static final int MAX_NAME_LENGTH;

    @Positive
    public static Preferences userNodeForPackage(Class<?> c);

    @Positive
    public static Preferences systemNodeForPackage(Class<?> c);

    @Positive
    public static Preferences userRoot();

    @Positive
    public static Preferences systemRoot();

    @Positive
    protected Preferences() {
    @Positive
    }

    @Positive
    public abstract void put(String key, String value);

    @Positive
    @PolyNull
    @Positive
    public abstract String get(String key, @PolyNull String def);

    @Positive
    public abstract void remove(String key);

    @Positive
    public abstract void clear() throws BackingStoreException;

    @Positive
    public abstract void putInt(String key, int value);

    @Positive
    public abstract int getInt(String key, int def);

    @Positive
    public abstract void putLong(String key, long value);

    @Positive
    public abstract long getLong(String key, long def);

    @Positive
    public abstract void putBoolean(String key, boolean value);

    @Positive
    public abstract boolean getBoolean(String key, boolean def);

    @Positive
    public abstract void putFloat(String key, float value);

    @Positive
    public abstract float getFloat(String key, float def);

    @Positive
    public abstract void putDouble(String key, double value);

    @Positive
    public abstract double getDouble(String key, double def);

    @Positive
    public abstract void putByteArray(String key, byte[] value);

    @Positive
    public abstract byte @PolyNull [] getByteArray(String key, byte @PolyNull [] def);

    @Positive
    public abstract String[] keys() throws BackingStoreException;

    @Positive
    public abstract String[] childrenNames() throws BackingStoreException;

    @Positive
    @Nullable
    @Positive
    public abstract Preferences parent();

    @Positive
    public abstract Preferences node(String pathName);

    @Positive
    public abstract boolean nodeExists(String pathName) throws BackingStoreException;

    @Positive
    public abstract void removeNode() throws BackingStoreException;

    @Positive
    public abstract String name();

    @Positive
    public abstract String absolutePath();

    @Positive
    public abstract boolean isUserNode();

    @Positive
    public abstract String toString();

    @Positive
    public abstract void flush() throws BackingStoreException;

    @Positive
    public abstract void sync() throws BackingStoreException;

    @Positive
    public abstract void addPreferenceChangeListener(PreferenceChangeListener pcl);

    @Positive
    public abstract void removePreferenceChangeListener(PreferenceChangeListener pcl);

    @Positive
    public abstract void addNodeChangeListener(NodeChangeListener ncl);

    @Positive
    public abstract void removeNodeChangeListener(NodeChangeListener ncl);

    @Positive
    public abstract void exportNode(OutputStream os) throws IOException, BackingStoreException;

    @Positive
    public abstract void exportSubtree(OutputStream os) throws IOException, BackingStoreException;

    @Positive
    public static void importPreferences(InputStream is) throws IOException, InvalidPreferencesFormatException;
    @Positive
}

// CFWR semantic augmentation - variant 0
