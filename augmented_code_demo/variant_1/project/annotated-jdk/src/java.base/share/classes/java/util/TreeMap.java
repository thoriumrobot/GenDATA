/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
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
import java.io.Serializable;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;

    @Positive
@CFComment({ "lock/nullness: This permits null element when using a custom comparator that allows null" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class TreeMap<K, V> extends AbstractMap<K, V> implements NavigableMap<K, V>, Cloneable, java.io.Serializable {

    @Positive
    public TreeMap() {
    @Positive
    }

    @Positive
    public TreeMap(@Nullable Comparator<? super K> comparator) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public TreeMap(@PolyNonEmpty Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public TreeMap(@PolyNonEmpty SortedMap<K, ? extends V> m) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    public boolean containsKey(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @Nullable
    @Positive
    public V get(@GuardSatisfied TreeMap<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Comparator<? super K> comparator(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @KeyFor("this")
    @Positive
    public K firstKey(@NonEmpty TreeMap<K, V> this);

    @Positive
    @KeyFor("this")
    @Positive
    public K lastKey(@NonEmpty TreeMap<K, V> this);

    @Positive
    public void putAll(@GuardSatisfied TreeMap<K, V> this, Map<? extends K, ? extends V> map);

    @Positive
    final Entry<K, V> getEntry(Object key);

    @Positive
    final Entry<K, V> getEntryUsingComparator(Object key);

    @Positive
    final Entry<K, V> getCeilingEntry(K key);

    @Positive
    final Entry<K, V> getFloorEntry(K key);

    @Positive
    final Entry<K, V> getHigherEntry(K key);

    @Positive
    final Entry<K, V> getLowerEntry(K key);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(@GuardSatisfied TreeMap<K, V> this, K key, V value);

    @Positive
    @Override
    @Positive
    public V putIfAbsent(K key, V value);

    @Positive
    @Override
    @Positive
    public V computeIfAbsent(K key, Function<? super K, ? extends V> mappingFunction);

    @Positive
    @Override
    @Positive
    public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends V> remappingFunction);

    @Positive
    @Override
    @Positive
    public V compute(K key, BiFunction<? super K, ? super V, ? extends V> remappingFunction);

    @Positive
    @Override
    @Positive
    public V merge(K key, V value, BiFunction<? super V, ? super V, ? extends V> remappingFunction);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    public void clear(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    public Object clone(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    public Map.@Nullable Entry<K, V> firstEntry();

    @Positive
    public Map.@Nullable Entry<K, V> lastEntry();

    @Positive
    public Map.@Nullable Entry<K, V> pollFirstEntry(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    public Map.@Nullable Entry<K, V> pollLastEntry(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    public Map.@Nullable Entry<K, V> lowerEntry(K key);

    @Positive
    @Nullable
    @Positive
    public K lowerKey(K key);

    @Positive
    public Map.@Nullable Entry<K, V> floorEntry(K key);

    @Positive
    @Nullable
    @Positive
    public K floorKey(K key);

    @Positive
    public Map.@Nullable Entry<K, V> ceilingEntry(K key);

    @Positive
    @Nullable
    @Positive
    public K ceilingKey(K key);

    @Positive
    public Map.@Nullable Entry<K, V> higherEntry(K key);

    @Positive
    @Nullable
    @Positive
    public K higherKey(K key);

    @Positive
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableSet<@KeyFor({ "this" }) K> navigableKeySet(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableSet<@KeyFor({ "this" }) K> descendingKeySet(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    public Collection<V> values(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableMap<K, V> descendingMap(@GuardSatisfied TreeMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableMap<K, V> subMap(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied K fromKey, boolean fromInclusive, @GuardSatisfied K toKey, boolean toInclusive);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableMap<K, V> headMap(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied K toKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    public NavigableMap<K, V> tailMap(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied K fromKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    public SortedMap<K, V> subMap(@GuardSatisfied TreeMap<K, V> this, @GuardSatisfied K fromKey, @GuardSatisfied K toKey);

    @Positive
    @SideEffectFree
    @Positive
    public SortedMap<K, V> headMap(@GuardSatisfied TreeMap<K, V> this, K toKey);

    @Positive
    @SideEffectFree
    @Positive
    public SortedMap<K, V> tailMap(@GuardSatisfied TreeMap<K, V> this, K fromKey);

    @Positive
    @Override
    @Positive
    public boolean replace(K key, V oldValue, V newValue);

    @Positive
    @Override
    @Positive
    public V replace(K key, V value);

    @Positive
    @Override
    @Positive
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    @Override
    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    class Values extends AbstractCollection<V> {

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
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        public void clear();

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<V> spliterator();
    @Positive
    }

    @Positive
    class EntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<Map.Entry<K, V>> iterator();

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
        @SideEffectFree
    @Positive
        public Spliterator<Map.Entry<K, V>> spliterator();
    @Positive
    }

    @Positive
    Iterator<K> keyIterator();

    @Positive
    Iterator<K> descendingKeyIterator();

    @Positive
    static final class KeySet<E> extends AbstractSet<E> implements NavigableSet<E> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        public Iterator<E> descendingIterator();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        public void clear();

    @Positive
        public E lower(E e);

    @Positive
        public E floor(E e);

    @Positive
        public E ceiling(E e);

    @Positive
        public E higher(E e);

    @Positive
        public E first();

    @Positive
        public E last();

    @Positive
        public Comparator<? super E> comparator();

    @Positive
        public E pollFirst();

    @Positive
        public E pollLast();

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        public NavigableSet<E> subSet(E fromElement, boolean fromInclusive, E toElement, boolean toInclusive);

    @Positive
        public NavigableSet<E> headSet(E toElement, boolean inclusive);

    @Positive
        public NavigableSet<E> tailSet(E fromElement, boolean inclusive);

    @Positive
        public SortedSet<E> subSet(E fromElement, E toElement);

    @Positive
        public SortedSet<E> headSet(E toElement);

    @Positive
        public SortedSet<E> tailSet(E fromElement);

    @Positive
        public NavigableSet<E> descendingSet();

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<E> spliterator();
    @Positive
    }

    @Positive
    abstract class PrivateEntryIterator<T> implements Iterator<T> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        final Entry<K, V> nextEntry();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        final Entry<K, V> prevEntry();

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    final class EntryIterator extends PrivateEntryIterator<Map.Entry<K, V>> {

    @Positive
        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);
    @Positive
    }

    @Positive
    final class ValueIterator extends PrivateEntryIterator<V> {

    @Positive
        public V next(@NonEmpty ValueIterator this);
    @Positive
    }

    @Positive
    final class KeyIterator extends PrivateEntryIterator<K> {

    @Positive
        public K next(@NonEmpty KeyIterator this);
    @Positive
    }

    @Positive
    final class DescendingKeyIterator extends PrivateEntryIterator<K> {

    @Positive
        public K next(@NonEmpty DescendingKeyIterator this);

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    final int compare(Object k1, Object k2);

    @Positive
    static final boolean valEquals(Object o1, Object o2);

    @Positive
    static <K, V> Map.Entry<K, V> exportEntry(TreeMap.Entry<K, V> e);

    @Positive
    static <K, V> K keyOrNull(TreeMap.Entry<K, V> e);

    @Positive
    static <K> K key(@NonNull Entry<K, ?> e);

    @Positive
    abstract static class NavigableSubMap<K, V> extends AbstractMap<K, V> implements NavigableMap<K, V>, java.io.Serializable {

    @Positive
        final boolean tooLow(Object key);

    @Positive
        final boolean tooHigh(Object key);

    @Positive
        final boolean inRange(Object key);

    @Positive
        final boolean inClosedRange(Object key);

    @Positive
        final boolean inRange(Object key, boolean inclusive);

    @Positive
        final TreeMap.Entry<K, V> absLowest();

    @Positive
        final TreeMap.Entry<K, V> absHighest();

    @Positive
        final TreeMap.Entry<K, V> absCeiling(K key);

    @Positive
        final TreeMap.Entry<K, V> absHigher(K key);

    @Positive
        final TreeMap.Entry<K, V> absFloor(K key);

    @Positive
        final TreeMap.Entry<K, V> absLower(K key);

    @Positive
        final TreeMap.Entry<K, V> absHighFence();

    @Positive
        final TreeMap.Entry<K, V> absLowFence();

    @Positive
        abstract TreeMap.Entry<K, V> subLowest();

    @Positive
        abstract TreeMap.Entry<K, V> subHighest();

    @Positive
        abstract TreeMap.Entry<K, V> subCeiling(K key);

    @Positive
        abstract TreeMap.Entry<K, V> subHigher(K key);

    @Positive
        abstract TreeMap.Entry<K, V> subFloor(K key);

    @Positive
        abstract TreeMap.Entry<K, V> subLower(K key);

    @Positive
        abstract Iterator<K> keyIterator();

    @Positive
        abstract Spliterator<K> keySpliterator();

    @Positive
        abstract Iterator<K> descendingKeyIterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public final boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        public final V put(K key, V value);

    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        public V merge(K key, V value, BiFunction<? super V, ? super V, ? extends V> remappingFunction);

    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends V> mappingFunction);

    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends V> remappingFunction);

    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends V> remappingFunction);

    @Positive
        public final V get(Object key);

    @Positive
        public final V remove(Object key);

    @Positive
        public final Map.Entry<K, V> ceilingEntry(K key);

    @Positive
        public final K ceilingKey(K key);

    @Positive
        public final Map.Entry<K, V> higherEntry(K key);

    @Positive
        public final K higherKey(K key);

    @Positive
        public final Map.Entry<K, V> floorEntry(K key);

    @Positive
        public final K floorKey(K key);

    @Positive
        public final Map.Entry<K, V> lowerEntry(K key);

    @Positive
        public final K lowerKey(K key);

    @Positive
        public final K firstKey();

    @Positive
        public final K lastKey();

    @Positive
        public final Map.Entry<K, V> firstEntry();

    @Positive
        public final Map.Entry<K, V> lastEntry();

    @Positive
        public final Map.Entry<K, V> pollFirstEntry();

    @Positive
        public final Map.Entry<K, V> pollLastEntry();

    @Positive
        @SideEffectFree
    @Positive
        public final NavigableSet<K> navigableKeySet();

    @Positive
        public final Set<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> descendingKeySet();

    @Positive
        @SideEffectFree
    @Positive
        public final SortedMap<K, V> subMap(K fromKey, K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public final SortedMap<K, V> headMap(K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public final SortedMap<K, V> tailMap(K fromKey);

    @Positive
        abstract class EntrySetView extends AbstractSet<Map.Entry<K, V>> {

    @Positive
            @Pure
    @Positive
            public int size();

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
            public boolean isEmpty();

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public boolean contains(@UnknownSignedness Object o);

    @Positive
            public boolean remove(@UnknownSignedness Object o);
    @Positive
        }

    @Positive
        abstract class SubMapIterator<T> implements Iterator<T> {

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public final boolean hasNext();

    @Positive
            @SideEffectsOnly("this")
    @Positive
            final TreeMap.Entry<K, V> nextEntry();

    @Positive
            @SideEffectsOnly("this")
    @Positive
            final TreeMap.Entry<K, V> prevEntry();

    @Positive
            final void removeAscending();

    @Positive
            final void removeDescending();
    @Positive
        }

    @Positive
        final class SubMapEntryIterator extends SubMapIterator<Map.Entry<K, V>> {

    @Positive
            public Map.Entry<K, V> next(@NonEmpty SubMapEntryIterator this);

    @Positive
            public void remove();
    @Positive
        }

    @Positive
        final class DescendingSubMapEntryIterator extends SubMapIterator<Map.Entry<K, V>> {

    @Positive
            public Map.Entry<K, V> next(@NonEmpty DescendingSubMapEntryIterator this);

    @Positive
            public void remove();
    @Positive
        }

    @Positive
        final class SubMapKeyIterator extends SubMapIterator<K> implements Spliterator<K> {

    @Positive
            public K next(@NonEmpty SubMapKeyIterator this);

    @Positive
            public void remove();

    @Positive
            public Spliterator<K> trySplit();

    @Positive
            public void forEachRemaining(Consumer<? super K> action);

    @Positive
            public boolean tryAdvance(Consumer<? super K> action);

    @Positive
            public long estimateSize();

    @Positive
            public int characteristics();

    @Positive
            public final Comparator<? super K> getComparator();
    @Positive
        }

    @Positive
        final class DescendingSubMapKeyIterator extends SubMapIterator<K> implements Spliterator<K> {

    @Positive
            public K next(@NonEmpty DescendingSubMapKeyIterator this);

    @Positive
            public void remove();

    @Positive
            public Spliterator<K> trySplit();

    @Positive
            public void forEachRemaining(Consumer<? super K> action);

    @Positive
            public boolean tryAdvance(Consumer<? super K> action);

    @Positive
            public long estimateSize();

    @Positive
            public int characteristics();
    @Positive
        }
    @Positive
    }

    @Positive
    static final class AscendingSubMap<K, V> extends NavigableSubMap<K, V> {

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> descendingMap();

    @Positive
        Iterator<K> keyIterator();

    @Positive
        Spliterator<K> keySpliterator();

    @Positive
        Iterator<K> descendingKeyIterator();

    @Positive
        final class AscendingEntrySetView extends EntrySetView {

    @Positive
            public Iterator<Map.Entry<K, V>> iterator();
    @Positive
        }

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        TreeMap.Entry<K, V> subLowest();

    @Positive
        TreeMap.Entry<K, V> subHighest();

    @Positive
        TreeMap.Entry<K, V> subCeiling(K key);

    @Positive
        TreeMap.Entry<K, V> subHigher(K key);

    @Positive
        TreeMap.Entry<K, V> subFloor(K key);

    @Positive
        TreeMap.Entry<K, V> subLower(K key);
    @Positive
    }

    @Positive
    static final class DescendingSubMap<K, V> extends NavigableSubMap<K, V> {

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> descendingMap();

    @Positive
        Iterator<K> keyIterator();

    @Positive
        Spliterator<K> keySpliterator();

    @Positive
        Iterator<K> descendingKeyIterator();

    @Positive
        final class DescendingEntrySetView extends EntrySetView {

    @Positive
            public Iterator<Map.Entry<K, V>> iterator();
    @Positive
        }

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        TreeMap.Entry<K, V> subLowest();

    @Positive
        TreeMap.Entry<K, V> subHighest();

    @Positive
        TreeMap.Entry<K, V> subCeiling(K key);

    @Positive
        TreeMap.Entry<K, V> subHigher(K key);

    @Positive
        TreeMap.Entry<K, V> subFloor(K key);

    @Positive
        TreeMap.Entry<K, V> subLower(K key);
    @Positive
    }

    @Positive
    private class SubMap extends AbstractMap<K, V> implements SortedMap<K, V>, java.io.Serializable {

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        public K lastKey();

    @Positive
        public K firstKey();

    @Positive
        @SideEffectFree
    @Positive
        public SortedMap<K, V> subMap(K fromKey, K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public SortedMap<K, V> headMap(K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public SortedMap<K, V> tailMap(K fromKey);

    @Positive
        public Comparator<? super K> comparator();
    @Positive
    }

    @Positive
    static final class Entry<K, V> implements Map.Entry<K, V> {

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
    final Entry<K, V> getFirstEntry();

    @Positive
    final Entry<K, V> getLastEntry();

    @Positive
    static <K, V> TreeMap.Entry<K, V> successor(Entry<K, V> t);

    @Positive
    static <K, V> Entry<K, V> predecessor(Entry<K, V> t);

    @Positive
    void readTreeSet(int size, java.io.ObjectInputStream s, V defaultVal) throws java.io.IOException, ClassNotFoundException;

    @Positive
    void addAllForTreeSet(SortedSet<? extends K> set, V defaultVal);

    @Positive
    static <K> Spliterator<K> keySpliteratorFor(NavigableMap<K, ?> m);

    @Positive
    final Spliterator<K> keySpliterator();

    @Positive
    final Spliterator<K> descendingKeySpliterator();

    @Positive
    static class TreeMapSpliterator<K, V> {

    @Positive
        final int getEstimate();

    @Positive
        public final long estimateSize();
    @Positive
    }

    @Positive
    static final class KeySpliterator<K, V> extends TreeMapSpliterator<K, V> implements Spliterator<K> {

    @Positive
        public KeySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super K> action);

    @Positive
        public boolean tryAdvance(Consumer<? super K> action);

    @Positive
        public int characteristics();

    @Positive
        public final Comparator<? super K> getComparator();
    @Positive
    }

    @Positive
    static final class DescendingKeySpliterator<K, V> extends TreeMapSpliterator<K, V> implements Spliterator<K> {

    @Positive
        public DescendingKeySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super K> action);

    @Positive
        public boolean tryAdvance(Consumer<? super K> action);

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static final class ValueSpliterator<K, V> extends TreeMapSpliterator<K, V> implements Spliterator<V> {

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
    static final class EntrySpliterator<K, V> extends TreeMapSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

    @Positive
        public EntrySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public Comparator<Map.Entry<K, V>> getComparator();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
