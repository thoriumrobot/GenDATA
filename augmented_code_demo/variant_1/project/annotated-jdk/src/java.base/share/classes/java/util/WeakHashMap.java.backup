/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
import org.checkerframework.checker.nullness.qual.KeyFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.lang.ref.WeakReference;
import java.lang.ref.ReferenceQueue;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;

@CFComment({ "lock: permits null keys and values" })
@AnnotatedFor({ "lock", "index", "nullness" })
public class WeakHashMap<K, V> extends AbstractMap<K, V> implements Map<K, V> {

    public WeakHashMap(@NonNegative int initialCapacity, float loadFactor) {
    }

    public WeakHashMap(@NonNegative int initialCapacity) {
    }

    public WeakHashMap() {
    }

    @PolyNonEmpty
    public WeakHashMap(@PolyNonEmpty Map<? extends K, ? extends V> m) {
    }

    static Object unmaskNull(Object key);

    final int hash(Object k);

    @Pure
    @NonNegative
    public int size(@GuardSatisfied WeakHashMap<K, V> this);

    @Pure
    @EnsuresNonEmptyIf(result = false, expression = "this")
    public boolean isEmpty(@GuardSatisfied WeakHashMap<K, V> this);

    @Pure
    @Nullable
    public V get(@GuardSatisfied WeakHashMap<K, V> this, @UnknownSignedness @GuardSatisfied @Nullable Object key);

    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Pure
    public boolean containsKey(@GuardSatisfied WeakHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    Entry<K, V> getEntry(Object key);

    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Nullable
    public V put(@GuardSatisfied WeakHashMap<K, V> this, K key, V value);

    void resize(int newCapacity);

    public void putAll(@GuardSatisfied WeakHashMap<K, V> this, Map<? extends K, ? extends V> m);

    @Nullable
    public V remove(@GuardSatisfied WeakHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    boolean removeMapping(Object o);

    public void clear(@GuardSatisfied WeakHashMap<K, V> this);

    @Pure
    public boolean containsValue(@GuardSatisfied WeakHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    private static class Entry<K, V> extends WeakReference<Object> implements Map.Entry<K, V> {

        @SuppressWarnings("unchecked")
        public K getKey();

        public V getValue();

        public V setValue(V newValue);

        public boolean equals(Object o);

        public int hashCode();

        public String toString();
    }

    private abstract class HashIterator<T> implements Iterator<T> {

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean hasNext();

        @SideEffectsOnly("this")
        protected Entry<K, V> nextEntry();

        public void remove();
    }

    private class ValueIterator extends HashIterator<V> {

        public V next(@NonEmpty ValueIterator this);
    }

    private class KeyIterator extends HashIterator<K> {

        public K next(@NonEmpty KeyIterator this);
    }

    private class EntryIterator extends HashIterator<Map.Entry<K, V>> {

        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);
    }

    @SideEffectFree
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied WeakHashMap<K, V> this);

    private class KeySet extends AbstractSet<K> {

        @SideEffectFree
        public Iterator<K> iterator();

        @Pure
        @NonNegative
        public int size();

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean contains(@Nullable @UnknownSignedness Object o);

        public boolean remove(@Nullable @UnknownSignedness Object o);

        public void clear();

        @SideEffectFree
        public Spliterator<K> spliterator();
    }

    @SideEffectFree
    public Collection<V> values(@GuardSatisfied WeakHashMap<K, V> this);

    private class Values extends AbstractCollection<V> {

        @SideEffectFree
        public Iterator<V> iterator();

        @Pure
        @NonNegative
        public int size();

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean contains(@Nullable @UnknownSignedness Object o);

        public void clear();

        @SideEffectFree
        public Spliterator<V> spliterator();
    }

    @SideEffectFree
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied WeakHashMap<K, V> this);

    private class EntrySet extends AbstractSet<Map.Entry<K, V>> {

        @SideEffectFree
        public Iterator<Map.Entry<K, V>> iterator();

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean contains(@Nullable @UnknownSignedness Object o);

        public boolean remove(@Nullable @UnknownSignedness Object o);

        @Pure
        @NonNegative
        public int size();

        public void clear();

        @SideEffectFree
        public Object[] toArray();

        @SideEffectFree
        @Nullable
        public <T> T[] toArray(@PolyNull T[] a);

        @SideEffectFree
        public Spliterator<Map.Entry<K, V>> spliterator();
    }

    @SuppressWarnings("unchecked")
    @Override
    public void forEach(BiConsumer<? super K, ? super V> action);

    @SuppressWarnings("unchecked")
    @Override
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    static class WeakHashMapSpliterator<K, V> {

        final int getFence();

        public final long estimateSize();
    }

    static final class KeySpliterator<K, V> extends WeakHashMapSpliterator<K, V> implements Spliterator<K> {

        public KeySpliterator<K, V> trySplit();

        public void forEachRemaining(Consumer<? super K> action);

        public boolean tryAdvance(Consumer<? super K> action);

        public int characteristics();
    }

    static final class ValueSpliterator<K, V> extends WeakHashMapSpliterator<K, V> implements Spliterator<V> {

        public ValueSpliterator<K, V> trySplit();

        public void forEachRemaining(Consumer<? super V> action);

        public boolean tryAdvance(Consumer<? super V> action);

        public int characteristics();
    }

    static final class EntrySpliterator<K, V> extends WeakHashMapSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

        public EntrySpliterator<K, V> trySplit();

        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

        public int characteristics();
    }
}
