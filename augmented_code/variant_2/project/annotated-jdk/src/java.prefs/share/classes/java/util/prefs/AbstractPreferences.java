/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.*;
    @Positive
import java.io.*;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.lang.Integer;
    @Positive
import java.lang.Long;
    @Positive
import java.lang.Float;
    @Positive
import java.lang.Double;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public abstract class AbstractPreferences extends Preferences {

    @Positive
    protected boolean newNode;

    @Positive
    protected final Object lock;

    @Positive
    protected AbstractPreferences(AbstractPreferences parent, String name) {
    @Positive
    }

    @Positive
    public void put(String key, String value);

    @Positive
    @PolyNull
    @Positive
    public String get(String key, @PolyNull String def);

    @Positive
    public void remove(String key);

    @Positive
    public void clear() throws BackingStoreException;

    @Positive
    public void putInt(String key, int value);

    @Positive
    public int getInt(String key, int def);

    @Positive
    public void putLong(String key, long value);

    @Positive
    public long getLong(String key, long def);

    @Positive
    public void putBoolean(String key, boolean value);

    @Positive
    public boolean getBoolean(String key, boolean def);

    @Positive
    public void putFloat(String key, float value);

    @Positive
    public float getFloat(String key, float def);

    @Positive
    public void putDouble(String key, double value);

    @Positive
    public double getDouble(String key, double def);

    @Positive
    public void putByteArray(String key, byte[] value);

    @Positive
    public byte @PolyNull [] getByteArray(String key, byte @PolyNull [] def);

    @Positive
    public String[] keys() throws BackingStoreException;

    @Positive
    public String[] childrenNames() throws BackingStoreException;

    @Positive
    protected final AbstractPreferences[] cachedChildren();

    @Positive
    @Nullable
    @Positive
    public Preferences parent();

    @Positive
    public Preferences node(String path);

    @Positive
    public boolean nodeExists(String path) throws BackingStoreException;

    @Positive
    public void removeNode() throws BackingStoreException;

    @Positive
    public String name();

    @Positive
    public String absolutePath();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public boolean isUserNode();

    @Positive
    public void addPreferenceChangeListener(PreferenceChangeListener pcl);

    @Positive
    public void removePreferenceChangeListener(PreferenceChangeListener pcl);

    @Positive
    public void addNodeChangeListener(NodeChangeListener ncl);

    @Positive
    public void removeNodeChangeListener(NodeChangeListener ncl);

    @Positive
    protected abstract void putSpi(String key, String value);

    @Positive
    protected abstract String getSpi(String key);

    @Positive
    protected abstract void removeSpi(String key);

    @Positive
    protected abstract void removeNodeSpi() throws BackingStoreException;

    @Positive
    protected abstract String[] keysSpi() throws BackingStoreException;

    @Positive
    protected abstract String[] childrenNamesSpi() throws BackingStoreException;

    @Positive
    @Nullable
    @Positive
    protected AbstractPreferences getChild(String nodeName) throws BackingStoreException;

    @Positive
    protected abstract AbstractPreferences childSpi(String name);

    @Positive
    public String toString();

    @Positive
    public void sync() throws BackingStoreException;

    @Positive
    protected abstract void syncSpi() throws BackingStoreException;

    @Positive
    public void flush() throws BackingStoreException;

    @Positive
    protected abstract void flushSpi() throws BackingStoreException;

    @Positive
    protected boolean isRemoved();

    @Positive
    private class NodeAddedEvent extends NodeChangeEvent {
    @Positive
    }

    @Positive
    private class NodeRemovedEvent extends NodeChangeEvent {
    @Positive
    }

    @Positive
    private static class EventDispatchThread extends Thread {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    PreferenceChangeListener[] prefListeners();

    @Positive
    NodeChangeListener[] nodeListeners();

    @Positive
    public void exportNode(OutputStream os) throws IOException, BackingStoreException;

    @Positive
    public void exportSubtree(OutputStream os) throws IOException, BackingStoreException;
    @Positive
}
