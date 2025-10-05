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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.*;
import java.io.*;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.lang.Integer;
import java.lang.Long;
import java.lang.Float;
import java.lang.Double;

@AnnotatedFor({ "nullness" })
public abstract class AbstractPreferences extends Preferences {

    protected boolean newNode;

    protected final Object lock;

    protected AbstractPreferences(AbstractPreferences parent, String name) {
    }

    public void put(String key, String value);

    @PolyNull
    public String get(String key, @PolyNull String def);

    public void remove(String key);

    public void clear() throws BackingStoreException;

    public void putInt(String key, int value);

    public int getInt(String key, int def);

    public void putLong(String key, long value);

    public long getLong(String key, long def);

    public void putBoolean(String key, boolean value);

    public boolean getBoolean(String key, boolean def);

    public void putFloat(String key, float value);

    public float getFloat(String key, float def);

    public void putDouble(String key, double value);

    public double getDouble(String key, double def);

    public void putByteArray(String key, byte[] value);

    public byte @PolyNull [] getByteArray(String key, byte @PolyNull [] def);

    public String[] keys() throws BackingStoreException;

    public String[] childrenNames() throws BackingStoreException;

    protected final AbstractPreferences[] cachedChildren();

    @Nullable
    public Preferences parent();

    public Preferences node(String path);

    public boolean nodeExists(String path) throws BackingStoreException;

    public void removeNode() throws BackingStoreException;

    public String name();

    public String absolutePath();

    @SuppressWarnings("removal")
    public boolean isUserNode();

    public void addPreferenceChangeListener(PreferenceChangeListener pcl);

    public void removePreferenceChangeListener(PreferenceChangeListener pcl);

    public void addNodeChangeListener(NodeChangeListener ncl);

    public void removeNodeChangeListener(NodeChangeListener ncl);

    protected abstract void putSpi(String key, String value);

    protected abstract String getSpi(String key);

    protected abstract void removeSpi(String key);

    protected abstract void removeNodeSpi() throws BackingStoreException;

    protected abstract String[] keysSpi() throws BackingStoreException;

    protected abstract String[] childrenNamesSpi() throws BackingStoreException;

    @Nullable
    protected AbstractPreferences getChild(String nodeName) throws BackingStoreException;

    protected abstract AbstractPreferences childSpi(String name);

    public String toString();

    public void sync() throws BackingStoreException;

    protected abstract void syncSpi() throws BackingStoreException;

    public void flush() throws BackingStoreException;

    protected abstract void flushSpi() throws BackingStoreException;

    protected boolean isRemoved();

    private class NodeAddedEvent extends NodeChangeEvent {
    }

    private class NodeRemovedEvent extends NodeChangeEvent {
    }

    private static class EventDispatchThread extends Thread {

        public void run();
    }

    PreferenceChangeListener[] prefListeners();

    NodeChangeListener[] nodeListeners();

    public void exportNode(OutputStream os) throws IOException, BackingStoreException;

    public void exportSubtree(OutputStream os) throws IOException, BackingStoreException;
}
