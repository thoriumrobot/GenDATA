/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.propkey.qual.PropertyKey;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.PrintWriter;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Writer;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.BufferedWriter;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.StreamCorruptedException;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.IllegalCharsetNameException;
    @Positive
import java.nio.charset.UnsupportedCharsetException;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Function;
    @Positive
import sun.nio.cs.ISO_8859_1;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.util.ArraysSupport;
    @Positive
import jdk.internal.util.xml.PropertiesDefaultHandler;

    @Positive
@AnnotatedFor({ "index", "lock", "nullness", "propkey" })
    @Positive
public class Properties extends Hashtable<Object, Object> {

    @Positive
    protected volatile Properties defaults;

    @Positive
    public Properties() {
    @Positive
    }

    @Positive
    public Properties(int initialCapacity) {
    @Positive
    }

    @Positive
    public Properties(Properties defaults) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public synchronized Object setProperty(@GuardSatisfied Properties this, @PropertyKey String key, String value);

    @Positive
    public synchronized void load(Reader reader) throws IOException;

    @Positive
    public synchronized void load(InputStream inStream) throws IOException;

    @Positive
    private static class LineReader {

    @Positive
        int readLine() throws IOException;
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public void save(OutputStream out, @Nullable String comments);

    @Positive
    public void store(Writer writer, @Nullable String comments) throws IOException;

    @Positive
    public void store(OutputStream out, @Nullable String comments) throws IOException;

    @Positive
    public synchronized void loadFromXML(InputStream in) throws IOException, InvalidPropertiesFormatException;

    @Positive
    public void storeToXML(OutputStream os, @Nullable String comment) throws IOException;

    @Positive
    public void storeToXML(OutputStream os, @Nullable String comment, String encoding) throws IOException;

    @Positive
    public void storeToXML(OutputStream os, String comment, Charset charset) throws IOException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getProperty(@GuardSatisfied Properties this, @PropertyKey String key);

    @Positive
    @Pure
    @Positive
    @PolyNull
    @Positive
    public String getProperty(@GuardSatisfied Properties this, @PropertyKey String key, @PolyNull String defaultValue);

    @Positive
    public Enumeration<?> propertyNames();

    @Positive
    public Set<String> stringPropertyNames();

    @Positive
    public void list(PrintStream out);

    @Positive
    public void list(PrintWriter out);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @Override
    @Positive
    public Enumeration<Object> keys();

    @Positive
    @Override
    @Positive
    public Enumeration<Object> elements();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public Object get(Object key);

    @Positive
    @Override
    @Positive
    public synchronized Object put(Object key, Object value);

    @Positive
    @Override
    @Positive
    public synchronized Object remove(@GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    @Override
    @Positive
    public synchronized void putAll(Map<?, ?> t);

    @Positive
    @Override
    @Positive
    public synchronized void clear();

    @Positive
    @Override
    @Positive
    public synchronized String toString();

    @Positive
    @Override
    @Positive
    public Set<@KeyFor("this") Object> keySet();

    @Positive
    @Override
    @Positive
    public Collection<Object> values();

    @Positive
    @Override
    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor("this") Object, Object>> entrySet();

    @Positive
    private static class EntrySet implements Set<Map.Entry<Object, Object>> {

    @Positive
        @Pure
    @Positive
        @Override
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        @Override
    @Positive
        public boolean isEmpty();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        @Override
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @Override
    @Positive
        public void clear();

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public boolean add(Map.Entry<Object, Object> e);

    @Positive
        @Override
    @Positive
        public boolean addAll(Collection<? extends Map.Entry<Object, Object>> c);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public Iterator<Map.Entry<Object, Object>> iterator();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public synchronized boolean equals(Object o);

    @Positive
    @Override
    @Positive
    public synchronized int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public Object getOrDefault(@GuardSatisfied @Nullable @UnknownSignedness Object key, Object defaultValue);

    @Positive
    @Override
    @Positive
    public synchronized void forEach(BiConsumer<? super Object, ? super Object> action);

    @Positive
    @Override
    @Positive
    public synchronized void replaceAll(BiFunction<? super Object, ? super Object, ?> function);

    @Positive
    @Override
    @Positive
    public synchronized Object putIfAbsent(Object key, Object value);

    @Positive
    @Override
    @Positive
    public synchronized boolean remove(@GuardSatisfied @Nullable @UnknownSignedness Object key, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    public synchronized boolean replace(Object key, Object oldValue, Object newValue);

    @Positive
    @Override
    @Positive
    public synchronized Object replace(Object key, Object value);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object computeIfAbsent(Object key, Function<? super Object, ? extends @PolyNull Object> mappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object computeIfPresent(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object compute(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public synchronized Object merge(Object key, Object value, BiFunction<? super Object, ? super Object, ?> remappingFunction);

    @Positive
    @Override
    @Positive
    protected void rehash();

    @Positive
    @Override
    @Positive
    public synchronized Object clone();

    @Positive
    @Override
    @Positive
    void writeHashtable(ObjectOutputStream s) throws IOException;

    @Positive
    @Override
    @Positive
    void readHashtable(ObjectInputStream s) throws IOException, ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 0
