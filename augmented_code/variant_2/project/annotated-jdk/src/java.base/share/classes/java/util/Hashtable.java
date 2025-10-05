/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.*;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.BiFunction;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@CFComment({ "lock: This collection can only contain nonnull values" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class Hashtable<K extends @NonNull Object, V extends @NonNull Object> extends Dictionary<K, V> implements Map<K, V>, Cloneable, java.io.Serializable {

    @Positive
    public Hashtable(@NonNegative int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public Hashtable(@NonNegative int initialCapacity) {
    @Positive
    }

    @Positive
    public Hashtable() {
    @Positive
    }

    @Positive
    public Hashtable(Map<? extends K, ? extends V> t) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public synchronized int size(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public synchronized boolean isEmpty(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    public synchronized Enumeration<@KeyFor({ "this" }) K> keys();

    @Positive
    public synchronized Enumeration<V> elements();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public synchronized boolean contains(@GuardSatisfied Hashtable<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied Hashtable<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public synchronized boolean containsKey(@GuardSatisfied Hashtable<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public synchronized V get(@GuardSatisfied Hashtable<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    protected void rehash();

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public synchronized V put(@GuardSatisfied Hashtable<K, V> this, K key, V value);

    @Positive
    @Nullable
    @Positive
    public synchronized V remove(@GuardSatisfied Hashtable<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    public synchronized void putAll(@GuardSatisfied Hashtable<K, V> this, Map<? extends K, ? extends V> t);

    @Positive
    public synchronized void clear(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public synchronized Object clone(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    final Hashtable<?, ?> cloneHashtable();

    @Positive
    public synchronized String toString(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    private class KeySet extends AbstractSet<K> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<K> iterator();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    private class EntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<Map.Entry<K, V>> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(Map.Entry<K, V> o);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Collection<V> values(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    private class ValueCollection extends AbstractCollection<V> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<V> iterator();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public synchronized boolean equals(@GuardSatisfied Hashtable<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
    @Pure
    @Positive
    public synchronized int hashCode(@GuardSatisfied Hashtable<K, V> this);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public synchronized V getOrDefault(@GuardSatisfied @UnknownSignedness Object key, V defaultValue);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public synchronized void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public synchronized void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Override
    @Positive
    public synchronized V putIfAbsent(K key, V value);

    @Positive
    @Override
    @Positive
    public synchronized boolean remove(@GuardSatisfied @UnknownSignedness Object key, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    public synchronized boolean replace(K key, V oldValue, V newValue);

    @Positive
    @Override
    @Positive
    public synchronized V replace(K key, V value);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    void writeHashtable(java.io.ObjectOutputStream s) throws IOException;

    @Positive
    final void defaultWriteHashtable(java.io.ObjectOutputStream s, int length, float loadFactor) throws IOException;

    @Positive
    void readHashtable(java.io.ObjectInputStream s) throws IOException, ClassNotFoundException;

    @Positive
    private static class Entry<K, V> implements Map.Entry<K, V> {

    @Positive
        protected Entry(int hash, K key, V value, Entry<K, V> next) {
    @Positive
        }

    @Positive
        @SideEffectFree
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        protected Object clone();

    @Positive
        public K getKey();

    @Positive
        public V getValue();

    @Positive
        public V setValue(V value);

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private class Enumerator<T> implements Enumeration<T>, Iterator<T> {

    @Positive
        protected int expectedModCount;

    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasMoreElements();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public T nextElement(@NonEmpty Enumerator<T> this);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public T next(@NonEmpty Enumerator<T> this);

    @Positive
        public void remove();
    @Positive
    }
    @Positive
}
