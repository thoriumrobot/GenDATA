/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.reflect.Array;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import jdk.internal.access.SharedSecrets;

@CFComment({ "lock/nullness: This collection can only contain null values" })
@AnnotatedFor({ "lock", "nullness", "index" })
public class IdentityHashMap<K, V> extends AbstractMap<K, V> implements Map<K, V>, java.io.Serializable, Cloneable {

    static final Object unmaskNull(Object key);

    public IdentityHashMap() {
    }

    public IdentityHashMap(@NonNegative int expectedMaxSize) {
    }

    public IdentityHashMap(Map<? extends K, ? extends V> m) {
    }

    @Pure
    @NonNegative
    public int size(@GuardSatisfied IdentityHashMap<K, V> this);

    @Pure
    @EnsuresNonEmptyIf(result = false, expression = "this")
    public boolean isEmpty(@GuardSatisfied IdentityHashMap<K, V> this);

    @Pure
    @SuppressWarnings("unchecked")
    @Nullable
    public V get(@GuardSatisfied IdentityHashMap<K, V> this, @UnknownSignedness @GuardSatisfied @Nullable Object key);

    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Pure
    public boolean containsKey(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Pure
    public boolean containsValue(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Nullable
    public V put(@GuardSatisfied IdentityHashMap<K, V> this, K key, V value);

    public void putAll(@GuardSatisfied IdentityHashMap<K, V> this, Map<? extends K, ? extends V> m);

    @Nullable
    public V remove(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    public void clear(@GuardSatisfied IdentityHashMap<K, V> this);

    @Pure
    public boolean equals(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable Object o);

    @Pure
    public int hashCode(@GuardSatisfied IdentityHashMap<K, V> this);

    @SideEffectFree
    public Object clone(@GuardSatisfied IdentityHashMap<K, V> this);

    private abstract class IdentityHashMapIterator<T> implements Iterator<T> {

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean hasNext();

        @SideEffectsOnly("this")
        protected int nextIndex(@NonEmpty IdentityHashMapIterator<T> this);

        public void remove();
    }

    private class KeyIterator extends IdentityHashMapIterator<K> {

        @SuppressWarnings("unchecked")
        public K next(@NonEmpty KeyIterator this);
    }

    private class ValueIterator extends IdentityHashMapIterator<V> {

        @SuppressWarnings("unchecked")
        public V next(@NonEmpty ValueIterator this);
    }

    private class EntryIterator extends IdentityHashMapIterator<Map.Entry<K, V>> {

        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);

        public void remove();

        private class Entry implements Map.Entry<K, V> {

            @SuppressWarnings("unchecked")
            public K getKey();

            @SuppressWarnings("unchecked")
            public V getValue();

            @SuppressWarnings("unchecked")
            public V setValue(V value);

            public boolean equals(@Nullable Object o);

            public int hashCode();

            public String toString();
        }
    }

    @SideEffectFree
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied IdentityHashMap<K, V> this);

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

        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

        public void clear();

        public int hashCode();

        @SideEffectFree
        public Object[] toArray();

        @SuppressWarnings("unchecked")
        @SideEffectFree
        @Nullable
        public <T> T[] toArray(@PolyNull T[] a);

        @SideEffectFree
        public Spliterator<K> spliterator();
    }

    @SideEffectFree
    public Collection<V> values(@GuardSatisfied IdentityHashMap<K, V> this);

    private class Values extends AbstractCollection<V> {

        @SideEffectFree
        public Iterator<V> iterator();

        @Pure
        @NonNegative
        public int size();

        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean contains(@Nullable @UnknownSignedness Object o);

        public boolean remove(@Nullable @UnknownSignedness Object o);

        public void clear();

        @SideEffectFree
        public Object[] toArray();

        @SuppressWarnings("unchecked")
        @SideEffectFree
        @Nullable
        public <T> T[] toArray(@PolyNull T[] a);

        @SideEffectFree
        public Spliterator<V> spliterator();
    }

    @SideEffectFree
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied IdentityHashMap<K, V> this);

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

        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

        @SideEffectFree
        public Object[] toArray();

        @SuppressWarnings("unchecked")
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

    static class IdentityHashMapSpliterator<K, V> {

        final int getFence();

        public final long estimateSize();
    }

    static final class KeySpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<K> {

        public KeySpliterator<K, V> trySplit();

        @SuppressWarnings("unchecked")
        public void forEachRemaining(Consumer<? super K> action);

        @SuppressWarnings("unchecked")
        public boolean tryAdvance(Consumer<? super K> action);

        public int characteristics();
    }

    static final class ValueSpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<V> {

        public ValueSpliterator<K, V> trySplit();

        public void forEachRemaining(Consumer<? super V> action);

        public boolean tryAdvance(Consumer<? super V> action);

        public int characteristics();
    }

    static final class EntrySpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

        public EntrySpliterator<K, V> trySplit();

        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

        public int characteristics();
    }
}
