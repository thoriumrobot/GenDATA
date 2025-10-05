/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.reflect.Array;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@CFComment({ "lock/nullness: This collection can only contain null values" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class IdentityHashMap<K, V> extends AbstractMap<K, V> implements Map<K, V>, java.io.Serializable, Cloneable {

    @Positive
    static final Object unmaskNull(Object key);

    @Positive
    public IdentityHashMap() {
    @Positive
    }

    @Positive
    public IdentityHashMap(@NonNegative int expectedMaxSize) {
    @Positive
    }

    @Positive
    public IdentityHashMap(Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public V get(@GuardSatisfied IdentityHashMap<K, V> this, @UnknownSignedness @GuardSatisfied @Nullable Object key);

    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(@GuardSatisfied IdentityHashMap<K, V> this, K key, V value);

    @Positive
    public void putAll(@GuardSatisfied IdentityHashMap<K, V> this, Map<? extends K, ? extends V> m);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    public void clear(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied IdentityHashMap<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    private abstract class IdentityHashMapIterator<T> implements Iterator<T> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        protected int nextIndex(@NonEmpty IdentityHashMapIterator<T> this);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    private class KeyIterator extends IdentityHashMapIterator<K> {

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public K next(@NonEmpty KeyIterator this);
    @Positive
    }

    @Positive
    private class ValueIterator extends IdentityHashMapIterator<V> {

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public V next(@NonEmpty ValueIterator this);
    @Positive
    }

    @Positive
    private class EntryIterator extends IdentityHashMapIterator<Map.Entry<K, V>> {

    @Positive
        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);

    @Positive
        public void remove();

    @Positive
        private class Entry implements Map.Entry<K, V> {

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public K getKey();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public V getValue();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public V setValue(V value);

    @Positive
            public boolean equals(@Nullable Object o);

    @Positive
            public int hashCode();

    @Positive
            public String toString();
    @Positive
        }
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied IdentityHashMap<K, V> this);

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
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public void clear();

    @Positive
        public int hashCode();

    @Positive
        @SideEffectFree
    @Positive
        public Object[] toArray();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<K> spliterator();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Collection<V> values(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    private class Values extends AbstractCollection<V> {

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
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        public void clear();

    @Positive
        @SideEffectFree
    @Positive
        public Object[] toArray();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<V> spliterator();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied IdentityHashMap<K, V> this);

    @Positive
    private class EntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<Map.Entry<K, V>> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        public void clear();

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @SideEffectFree
    @Positive
        public Object[] toArray();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<Map.Entry<K, V>> spliterator();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    static class IdentityHashMapSpliterator<K, V> {

    @Positive
        final int getFence();

    @Positive
        public final long estimateSize();
    @Positive
    }

    @Positive
    static final class KeySpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<K> {

    @Positive
        public KeySpliterator<K, V> trySplit();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public void forEachRemaining(Consumer<? super K> action);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public boolean tryAdvance(Consumer<? super K> action);

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static final class ValueSpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<V> {

    @Positive
        public ValueSpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super V> action);

    @Positive
        public boolean tryAdvance(Consumer<? super V> action);

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static final class EntrySpliterator<K, V> extends IdentityHashMapSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

    @Positive
        public EntrySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public int characteristics();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
