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
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
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
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.MinLen;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
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
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import java.util.stream.IntStream;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "lock", "index", "nonempty", "nullness" })
    @Positive
public class Collections {

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T extends Comparable<? super T>> void sort(List<T> list);

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    public static <T> void sort(List<T> list, @Nullable Comparator<? super T> c);

    @Positive
    public static <T> int binarySearch(List<? extends Comparable<? super T>> list, T key);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> int binarySearch(List<? extends T> list, T key, @Nullable Comparator<? super T> c);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public static void reverse(@GuardSatisfied List<?> list);

    @Positive
    public static void shuffle(@GuardSatisfied List<?> list);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public static void shuffle(@GuardSatisfied List<?> list, Random rnd);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public static void swap(@GuardSatisfied List<?> list, int i, int j);

    @Positive
    public static <T> void fill(@GuardSatisfied List<? super T> list, T obj);

    @Positive
    public static <T> void copy(List<? super T> dest, List<? extends T> src);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static <T extends Object & Comparable<? super T>> T min(Collection<? extends T> coll);

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static <T> T min(Collection<? extends T> coll, @Nullable Comparator<? super T> comp);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static <T extends Object & Comparable<? super T>> T max(Collection<? extends T> coll);

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static <T> T max(Collection<? extends T> coll, @Nullable Comparator<? super T> comp);

    @Positive
    public static void rotate(@GuardSatisfied List<?> list, int distance);

    @Positive
    public static <T> boolean replaceAll(List<T> list, @Nullable T oldVal, T newVal);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public static int indexOfSubList(@GuardSatisfied List<?> source, @GuardSatisfied List<?> target);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public static int lastIndexOfSubList(@GuardSatisfied List<?> source, @GuardSatisfied List<?> target);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> Collection<T> unmodifiableCollection(@PolyGrowShrink @PolyNonEmpty Collection<? extends T> c);

    @Positive
    static class UnmodifiableCollection<E> implements Collection<E>, Serializable {

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
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.UnmodifiableCollection<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public <T> T[] toArray(IntFunction<T[]> f);

    @Positive
        public String toString();

    @Positive
        @SideEffectFree
    @Positive
        @PolyGrowShrink
    @Positive
        @PolyNonEmpty
    @Positive
        public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty UnmodifiableCollection<E> this);

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean addAll(Collection<? extends E> coll);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public void clear();

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> Set<T> unmodifiableSet(@PolyNonEmpty Set<? extends T> s);

    @Positive
    static class UnmodifiableSet<E> extends UnmodifiableCollection<E> implements Set<E>, Serializable {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> SortedSet<T> unmodifiableSortedSet(@PolyNonEmpty SortedSet<T> s);

    @Positive
    static class UnmodifiableSortedSet<E> extends UnmodifiableSet<E> implements SortedSet<E>, Serializable {

    @Positive
        public Comparator<? super E> comparator();

    @Positive
        public SortedSet<E> subSet(E fromElement, E toElement);

    @Positive
        public SortedSet<E> headSet(E toElement);

    @Positive
        public SortedSet<E> tailSet(E fromElement);

    @Positive
        public E first();

    @Positive
        public E last();
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> NavigableSet<T> unmodifiableNavigableSet(@PolyNonEmpty NavigableSet<T> s);

    @Positive
    static class UnmodifiableNavigableSet<E> extends UnmodifiableSortedSet<E> implements NavigableSet<E>, Serializable {

    @Positive
        private static class EmptyNavigableSet<E> extends UnmodifiableNavigableSet<E> implements Serializable {

    @Positive
            @SideEffectFree
    @Positive
            public EmptyNavigableSet() {
    @Positive
            }
    @Positive
        }

    @Positive
        public E lower(E e);

    @Positive
        public E floor(E e);

    @Positive
        public E ceiling(E e);

    @Positive
        public E higher(E e);

    @Positive
        public E pollFirst();

    @Positive
        public E pollLast();

    @Positive
        public NavigableSet<E> descendingSet();

    @Positive
        public Iterator<E> descendingIterator();

    @Positive
        public NavigableSet<E> subSet(E fromElement, boolean fromInclusive, E toElement, boolean toInclusive);

    @Positive
        public NavigableSet<E> headSet(E toElement, boolean inclusive);

    @Positive
        public NavigableSet<E> tailSet(E fromElement, boolean inclusive);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> List<T> unmodifiableList(@PolyGrowShrink @PolyNonEmpty List<? extends T> list);

    @Positive
    static class UnmodifiableList<E> extends UnmodifiableCollection<E> implements List<E> {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public E get(int index);

    @Positive
        public E set(int index, E element);

    @Positive
        public void add(int index, E element);

    @Positive
        public E remove(int index);

    @Positive
        public int indexOf(Object o);

    @Positive
        public int lastIndexOf(Object o);

    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        @PolyGrowShrink
    @Positive
        @PolyNonEmpty
    @Positive
        public ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty UnmodifiableList<E> this);

    @Positive
        public ListIterator<E> listIterator(final int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }

    @Positive
    static class UnmodifiableRandomAccessList<E> extends UnmodifiableList<E> implements RandomAccess {

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @PolyNonEmpty
    @Positive
    public static <K, V> Map<K, V> unmodifiableMap(@PolyNonEmpty Map<? extends K, ? extends V> m);

    @Positive
    private static class UnmodifiableMap<K, V> implements Map<K, V>, Serializable {

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
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object val);

    @Positive
        public V get(Object key);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        public V put(K key, V value);

    @Positive
        public V remove(Object key);

    @Positive
        public void putAll(Map<? extends K, ? extends V> m);

    @Positive
        public void clear();

    @Positive
        public Set<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        public Collection<V> values();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Pure
    @Positive
        public V getOrDefault(Object k, V defaultValue);

    @Positive
        @Override
    @Positive
        public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

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
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        static class UnmodifiableEntrySet<K, V> extends UnmodifiableSet<Map.Entry<K, V>> {

    @Positive
            static <K, V> Consumer<Map.Entry<? extends K, ? extends V>> entryConsumer(Consumer<? super Entry<K, V>> action);

    @Positive
            public void forEach(Consumer<? super Entry<K, V>> action);

    @Positive
            static final class UnmodifiableEntrySetSpliterator<K, V> implements Spliterator<Entry<K, V>> {

    @Positive
                @Override
    @Positive
                public boolean tryAdvance(Consumer<? super Entry<K, V>> action);

    @Positive
                @Override
    @Positive
                public void forEachRemaining(Consumer<? super Entry<K, V>> action);

    @Positive
                @Override
    @Positive
                public Spliterator<Entry<K, V>> trySplit();

    @Positive
                @Override
    @Positive
                public long estimateSize();

    @Positive
                @Override
    @Positive
                public long getExactSizeIfKnown();

    @Positive
                @Override
    @Positive
                public int characteristics();

    @Positive
                @Override
    @Positive
                public boolean hasCharacteristics(int characteristics);

    @Positive
                @Override
    @Positive
                public Comparator<? super Entry<K, V>> getComparator();
    @Positive
            }

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public Spliterator<Entry<K, V>> spliterator();

    @Positive
            @Override
    @Positive
            public Stream<Entry<K, V>> stream();

    @Positive
            @Override
    @Positive
            public Stream<Entry<K, V>> parallelStream();

    @Positive
            public Iterator<Map.Entry<K, V>> iterator();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public Object[] toArray();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            @Nullable
    @Positive
            public <T> T[] toArray(@PolyNull T[] a);

    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public boolean contains(@UnknownSignedness Object o);

    @Positive
            @Pure
    @Positive
            public boolean containsAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
            public boolean equals(Object o);

    @Positive
            private static class UnmodifiableEntry<K, V> implements Map.Entry<K, V> {

    @Positive
                public K getKey();

    @Positive
                public V getValue();

    @Positive
                public V setValue(V value);

    @Positive
                public int hashCode();

    @Positive
                public boolean equals(Object o);

    @Positive
                public String toString();
    @Positive
            }
    @Positive
        }
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @PolyNonEmpty
    @Positive
    public static <K, V> SortedMap<K, V> unmodifiableSortedMap(@PolyNonEmpty SortedMap<K, ? extends V> m);

    @Positive
    static class UnmodifiableSortedMap<K, V> extends UnmodifiableMap<K, V> implements SortedMap<K, V>, Serializable {

    @Positive
        public Comparator<? super K> comparator();

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
        public K firstKey();

    @Positive
        public K lastKey();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @PolyNonEmpty
    @Positive
    public static <K, V> NavigableMap<K, V> unmodifiableNavigableMap(@PolyNonEmpty NavigableMap<K, ? extends V> m);

    @Positive
    static class UnmodifiableNavigableMap<K, V> extends UnmodifiableSortedMap<K, V> implements NavigableMap<K, V>, Serializable {

    @Positive
        private static class EmptyNavigableMap<K, V> extends UnmodifiableNavigableMap<K, V> implements Serializable {

    @Positive
            @Override
    @Positive
            @SideEffectFree
    @Positive
            public NavigableSet<K> navigableKeySet();
    @Positive
        }

    @Positive
        public K lowerKey(K key);

    @Positive
        public K floorKey(K key);

    @Positive
        public K ceilingKey(K key);

    @Positive
        public K higherKey(K key);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> lowerEntry(K key);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> floorEntry(K key);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> ceilingEntry(K key);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> higherEntry(K key);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> firstEntry();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public Entry<K, V> lastEntry();

    @Positive
        public Entry<K, V> pollFirstEntry();

    @Positive
        public Entry<K, V> pollLastEntry();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> descendingMap();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> navigableKeySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> descendingKeySet();

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
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> Collection<T> synchronizedCollection(@PolyGrowShrink @PolyNonEmpty Collection<T> c);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    static <T> Collection<T> synchronizedCollection(@PolyGrowShrink @PolyNonEmpty Collection<T> c, Object mutex);

    @Positive
    static class SynchronizedCollection<E> implements Collection<E>, Serializable {

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
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.SynchronizedCollection<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public <T> T[] toArray(IntFunction<T[]> f);

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean addAll(Collection<? extends E> coll);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public void clear();

    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> consumer);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();
    @Positive
    }

    @Positive
    public static <T> Set<T> synchronizedSet(Set<T> s);

    @Positive
    static <T> Set<T> synchronizedSet(Set<T> s, Object mutex);

    @Positive
    static class SynchronizedSet<E> extends SynchronizedCollection<E> implements Set<E> {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static <T> SortedSet<T> synchronizedSortedSet(SortedSet<T> s);

    @Positive
    static class SynchronizedSortedSet<E> extends SynchronizedSet<E> implements SortedSet<E> {

    @Positive
        public Comparator<? super E> comparator();

    @Positive
        public SortedSet<E> subSet(E fromElement, E toElement);

    @Positive
        public SortedSet<E> headSet(E toElement);

    @Positive
        public SortedSet<E> tailSet(E fromElement);

    @Positive
        public E first();

    @Positive
        public E last();
    @Positive
    }

    @Positive
    public static <T> NavigableSet<T> synchronizedNavigableSet(NavigableSet<T> s);

    @Positive
    static class SynchronizedNavigableSet<E> extends SynchronizedSortedSet<E> implements NavigableSet<E> {

    @Positive
        public E lower(E e);

    @Positive
        public E floor(E e);

    @Positive
        public E ceiling(E e);

    @Positive
        public E higher(E e);

    @Positive
        public E pollFirst();

    @Positive
        public E pollLast();

    @Positive
        public NavigableSet<E> descendingSet();

    @Positive
        public Iterator<E> descendingIterator();

    @Positive
        public NavigableSet<E> subSet(E fromElement, E toElement);

    @Positive
        public NavigableSet<E> headSet(E toElement);

    @Positive
        public NavigableSet<E> tailSet(E fromElement);

    @Positive
        public NavigableSet<E> subSet(E fromElement, boolean fromInclusive, E toElement, boolean toInclusive);

    @Positive
        public NavigableSet<E> headSet(E toElement, boolean inclusive);

    @Positive
        public NavigableSet<E> tailSet(E fromElement, boolean inclusive);
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> List<T> synchronizedList(@PolyGrowShrink @PolyNonEmpty List<T> list);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    static <T> List<T> synchronizedList(@PolyGrowShrink @PolyNonEmpty List<T> list, Object mutex);

    @Positive
    static class SynchronizedList<E> extends SynchronizedCollection<E> implements List<E> {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public E get(int index);

    @Positive
        public E set(int index, E element);

    @Positive
        public void add(int index, E element);

    @Positive
        public E remove(int index);

    @Positive
        public int indexOf(Object o);

    @Positive
        public int lastIndexOf(Object o);

    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        public ListIterator<E> listIterator();

    @Positive
        public ListIterator<E> listIterator(int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);
    @Positive
    }

    @Positive
    static class SynchronizedRandomAccessList<E> extends SynchronizedList<E> implements RandomAccess {

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }

    @Positive
    public static <K, V> Map<K, V> synchronizedMap(Map<K, V> m);

    @Positive
    private static class SynchronizedMap<K, V> implements Map<K, V>, Serializable {

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
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object value);

    @Positive
        public V get(Object key);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        public V put(K key, V value);

    @Positive
        public V remove(Object key);

    @Positive
        public void putAll(Map<? extends K, ? extends V> map);

    @Positive
        public void clear();

    @Positive
        public Set<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        public Collection<V> values();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public V getOrDefault(Object k, V defaultValue);

    @Positive
        @Override
    @Positive
        public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

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
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);
    @Positive
    }

    @Positive
    public static <K, V> SortedMap<K, V> synchronizedSortedMap(SortedMap<K, V> m);

    @Positive
    static class SynchronizedSortedMap<K, V> extends SynchronizedMap<K, V> implements SortedMap<K, V> {

    @Positive
        public Comparator<? super K> comparator();

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
        public K firstKey();

    @Positive
        public K lastKey();
    @Positive
    }

    @Positive
    public static <K, V> NavigableMap<K, V> synchronizedNavigableMap(NavigableMap<K, V> m);

    @Positive
    static class SynchronizedNavigableMap<K, V> extends SynchronizedSortedMap<K, V> implements NavigableMap<K, V> {

    @Positive
        public Entry<K, V> lowerEntry(K key);

    @Positive
        public K lowerKey(K key);

    @Positive
        public Entry<K, V> floorEntry(K key);

    @Positive
        public K floorKey(K key);

    @Positive
        public Entry<K, V> ceilingEntry(K key);

    @Positive
        public K ceilingKey(K key);

    @Positive
        public Entry<K, V> higherEntry(K key);

    @Positive
        public K higherKey(K key);

    @Positive
        public Entry<K, V> firstEntry();

    @Positive
        public Entry<K, V> lastEntry();

    @Positive
        public Entry<K, V> pollFirstEntry();

    @Positive
        public Entry<K, V> pollLastEntry();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> descendingMap();

    @Positive
        public NavigableSet<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> navigableKeySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> descendingKeySet();

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
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <E> Collection<E> checkedCollection(@PolyGrowShrink @PolyNonEmpty Collection<E> c, Class<E> type);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <T> T[] zeroLengthArray(Class<T> type);

    @Positive
    static class CheckedCollection<E> implements Collection<E>, Serializable {

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        E typeCheck(Object o);

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
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.CheckedCollection<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public <T> T[] toArray(IntFunction<T[]> f);

    @Positive
        public String toString();

    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        public void clear();

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> coll);

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        Collection<E> checkedCopyOf(Collection<? extends E> coll);

    @Positive
        public boolean addAll(Collection<? extends E> coll);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <E> Queue<E> checkedQueue(@PolyGrowShrink @PolyNonEmpty Queue<E> queue, Class<E> type);

    @Positive
    static class CheckedQueue<E> extends CheckedCollection<E> implements Queue<E>, Serializable {

    @Positive
        public E element();

    @Positive
        @Pure
    @Positive
        public boolean equals(Object o);

    @Positive
        @Pure
    @Positive
        public int hashCode();

    @Positive
        @Pure
    @Positive
        public E peek();

    @Positive
        public E poll();

    @Positive
        public E remove();

    @Positive
        public boolean offer(E e);
    @Positive
    }

    @Positive
    public static <E> Set<E> checkedSet(Set<E> s, Class<E> type);

    @Positive
    static class CheckedSet<E> extends CheckedCollection<E> implements Set<E>, Serializable {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static <E> SortedSet<E> checkedSortedSet(SortedSet<E> s, Class<E> type);

    @Positive
    static class CheckedSortedSet<E> extends CheckedSet<E> implements SortedSet<E>, Serializable {

    @Positive
        public Comparator<? super E> comparator();

    @Positive
        public E first();

    @Positive
        public E last();

    @Positive
        public SortedSet<E> subSet(E fromElement, E toElement);

    @Positive
        public SortedSet<E> headSet(E toElement);

    @Positive
        public SortedSet<E> tailSet(E fromElement);
    @Positive
    }

    @Positive
    public static <E> NavigableSet<E> checkedNavigableSet(NavigableSet<E> s, Class<E> type);

    @Positive
    static class CheckedNavigableSet<E> extends CheckedSortedSet<E> implements NavigableSet<E>, Serializable {

    @Positive
        public E lower(E e);

    @Positive
        public E floor(E e);

    @Positive
        public E ceiling(E e);

    @Positive
        public E higher(E e);

    @Positive
        public E pollFirst();

    @Positive
        public E pollLast();

    @Positive
        public NavigableSet<E> descendingSet();

    @Positive
        public Iterator<E> descendingIterator();

    @Positive
        public NavigableSet<E> subSet(E fromElement, E toElement);

    @Positive
        public NavigableSet<E> headSet(E toElement);

    @Positive
        public NavigableSet<E> tailSet(E fromElement);

    @Positive
        public NavigableSet<E> subSet(E fromElement, boolean fromInclusive, E toElement, boolean toInclusive);

    @Positive
        public NavigableSet<E> headSet(E toElement, boolean inclusive);

    @Positive
        public NavigableSet<E> tailSet(E fromElement, boolean inclusive);
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <E> List<E> checkedList(@PolyGrowShrink @PolyNonEmpty List<E> list, Class<E> type);

    @Positive
    static class CheckedList<E> extends CheckedCollection<E> implements List<E> {

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public E get(int index);

    @Positive
        public E remove(int index);

    @Positive
        public int indexOf(Object o);

    @Positive
        public int lastIndexOf(Object o);

    @Positive
        public E set(int index, E element);

    @Positive
        public void add(int index, E element);

    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        public ListIterator<E> listIterator();

    @Positive
        public ListIterator<E> listIterator(final int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);
    @Positive
    }

    @Positive
    static class CheckedRandomAccessList<E> extends CheckedList<E> implements RandomAccess {

    @Positive
        public List<E> subList(int fromIndex, int toIndex);
    @Positive
    }

    @Positive
    public static <K, V> Map<K, V> checkedMap(Map<K, V> m, Class<K> keyType, Class<V> valueType);

    @Positive
    private static class CheckedMap<K, V> implements Map<K, V>, Serializable {

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
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object v);

    @Positive
        public V get(Object key);

    @Positive
        public V remove(Object key);

    @Positive
        public void clear();

    @Positive
        public Set<K> keySet();

    @Positive
        public Collection<V> values();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        public String toString();

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        public V put(K key, V value);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public void putAll(Map<? extends K, ? extends V> t);

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        @Override
    @Positive
        public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

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
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        static class CheckedEntrySet<K, V> implements Set<Map.Entry<K, V>> {

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
            public String toString();

    @Positive
            public int hashCode();

    @Positive
            public void clear();

    @Positive
            @EnsuresNonEmpty("this")
    @Positive
            public boolean add(Map.Entry<K, V> e);

    @Positive
            public boolean addAll(Collection<? extends Map.Entry<K, V>> coll);

    @Positive
            public Iterator<Map.Entry<K, V>> iterator();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            public Object[] toArray();

    @Positive
            @SuppressWarnings("unchecked")
    @Positive
            @Nullable
    @Positive
            public <T> T[] toArray(@PolyNull T[] a);

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public boolean contains(@UnknownSignedness Object o);

    @Positive
            @Pure
    @Positive
            public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
            public boolean remove(@UnknownSignedness Object o);

    @Positive
            public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
            public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
            public boolean equals(Object o);

    @Positive
            static <K, V, T> CheckedEntry<K, V, T> checkedEntry(Map.Entry<K, V> e, Class<T> valueType);

    @Positive
            private static class CheckedEntry<K, V, T> implements Map.Entry<K, V> {

    @Positive
                public K getKey();

    @Positive
                public V getValue();

    @Positive
                public int hashCode();

    @Positive
                public String toString();

    @Positive
                public V setValue(V value);

    @Positive
                public boolean equals(Object o);
    @Positive
            }
    @Positive
        }
    @Positive
    }

    @Positive
    public static <K, V> SortedMap<K, V> checkedSortedMap(SortedMap<K, V> m, Class<K> keyType, Class<V> valueType);

    @Positive
    static class CheckedSortedMap<K, V> extends CheckedMap<K, V> implements SortedMap<K, V>, Serializable {

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        public K firstKey();

    @Positive
        public K lastKey();

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
    }

    @Positive
    public static <K, V> NavigableMap<K, V> checkedNavigableMap(NavigableMap<K, V> m, Class<K> keyType, Class<V> valueType);

    @Positive
    static class CheckedNavigableMap<K, V> extends CheckedSortedMap<K, V> implements NavigableMap<K, V>, Serializable {

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        public K firstKey();

    @Positive
        public K lastKey();

    @Positive
        public Entry<K, V> lowerEntry(K key);

    @Positive
        public K lowerKey(K key);

    @Positive
        public Entry<K, V> floorEntry(K key);

    @Positive
        public K floorKey(K key);

    @Positive
        public Entry<K, V> ceilingEntry(K key);

    @Positive
        public K ceilingKey(K key);

    @Positive
        public Entry<K, V> higherEntry(K key);

    @Positive
        public K higherKey(K key);

    @Positive
        public Entry<K, V> firstEntry();

    @Positive
        public Entry<K, V> lastEntry();

    @Positive
        public Entry<K, V> pollFirstEntry();

    @Positive
        public Entry<K, V> pollLastEntry();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> descendingMap();

    @Positive
        public NavigableSet<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> navigableKeySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> descendingKeySet();

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> subMap(K fromKey, K toKey);

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> headMap(K toKey);

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public NavigableMap<K, V> tailMap(K fromKey);

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
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <T> Iterator<T> emptyIterator();

    @Positive
    private static class EmptyIterator<E> implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty EmptyIterator<E> this);

    @Positive
        public void remove(@NonEmpty EmptyIterator<E> this);

    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super E> action);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <T> ListIterator<T> emptyListIterator();

    @Positive
    private static class EmptyListIterator<E> extends EmptyIterator<E> implements ListIterator<E> {

    @Positive
        public boolean hasPrevious();

    @Positive
        public E previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <T> Enumeration<T> emptyEnumeration();

    @Positive
    private static class EmptyEnumeration<E> implements Enumeration<E> {

    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasMoreElements();

    @Positive
        public E nextElement(@NonEmpty EmptyEnumeration<E> this);

    @Positive
        public Iterator<E> asIterator();
    @Positive
    }

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    public static final Set EMPTY_SET;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static final <T> Set<T> emptySet();

    @Positive
    private static class EmptySet<E> extends AbstractSet<E> implements Serializable {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

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
        public void clear();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object obj);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.EmptySet<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <E> SortedSet<E> emptySortedSet();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static <E> NavigableSet<E> emptyNavigableSet();

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    public static final List EMPTY_LIST;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static final <T> List<T> emptyList();

    @Positive
    private static class EmptyList<E> extends AbstractList<E> implements RandomAccess, Serializable {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        public ListIterator<E> listIterator();

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
        public void clear();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object obj);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.EmptyList<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public E get(int index);

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();
    @Positive
    }

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    public static final Map EMPTY_MAP;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static final <K, V> Map<K, V> emptyMap();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static final <K, V> SortedMap<K, V> emptySortedMap();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @SideEffectFree
    @Positive
    public static final <K, V> NavigableMap<K, V> emptyNavigableMap();

    @Positive
    private static class EmptyMap<K, V> extends AbstractMap<K, V> implements Serializable {

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
        public void clear();

    @Positive
        @Pure
    @Positive
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object value);

    @Positive
        public V get(Object key);

    @Positive
        public Set<K> keySet();

    @Positive
        public Collection<V> values();

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Pure
    @Positive
        public V getOrDefault(Object k, V defaultValue);

    @Positive
        @Override
    @Positive
        public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

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
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);
    @Positive
    }

    @Positive
    public static <T> Set<T> singleton(T o);

    @Positive
    static <E> Iterator<E> singletonIterator(final E e);

    @Positive
    static <T> Spliterator<T> singletonSpliterator(final T element);

    @Positive
    private static class SingletonSet<E> extends AbstractSet<E> implements Serializable {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

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
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @ArrayLen(1)
    @Positive
    public static <T> List<T> singletonList(T o);

    @Positive
    @ArrayLen(1)
    @Positive
    private static class SingletonList<E> extends AbstractList<E> implements RandomAccess, Serializable {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

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
        public boolean contains(@UnknownSignedness Object obj);

    @Positive
        public E get(int index);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static <K, V> Map<K, V> singletonMap(K key, V value);

    @Positive
    private static class SingletonMap<K, V> extends AbstractMap<K, V> implements Serializable {

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
        @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object value);

    @Positive
        public V get(Object key);

    @Positive
        public Set<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        public Collection<V> values();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public V getOrDefault(Object key, V defaultValue);

    @Positive
        @Override
    @Positive
        public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
        @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

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
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    public static <T> List<T> nCopies(@NonNegative int n, T o);

    @Positive
    private static class CopiesList<E> extends AbstractList<E> implements RandomAccess, Serializable {

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
        public boolean contains(@UnknownSignedness Object obj);

    @Positive
        public int indexOf(Object o);

    @Positive
        public int lastIndexOf(Object o);

    @Positive
        public E get(int index);

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.CopiesList<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> Comparator<T> reverseOrder();

    @Positive
    private static class ReverseComparator implements Comparator<Comparable<Object>>, Serializable {

    @Positive
        public int compare(Comparable<Object> c1, Comparable<Object> c2);

    @Positive
        @Override
    @Positive
        public Comparator<Comparable<Object>> reversed();
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> Comparator<T> reverseOrder(@Nullable Comparator<T> cmp);

    @Positive
    private static class ReverseComparator2<T> implements Comparator<T>, Serializable {

    @Positive
        public int compare(T t1, T t2);

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public Comparator<T> reversed();
    @Positive
    }

    @Positive
    public static <T> Enumeration<T> enumeration(final Collection<T> c);

    @Positive
    public static <T> ArrayList<T> list(Enumeration<T> e);

    @Positive
    static boolean eq(Object o1, Object o2);

    @Positive
    @NonNegative
    @Positive
    public static int frequency(Collection<?> c, @Nullable Object o);

    @Positive
    public static boolean disjoint(Collection<?> c1, Collection<?> c2);

    @Positive
    @SafeVarargs
    @Positive
    public static <T> boolean addAll(@GuardSatisfied Collection<? super T> c, T... elements);

    @Positive
    @SideEffectFree
    @Positive
    public static <E> Set<E> newSetFromMap(Map<E, Boolean> map);

    @Positive
    private static class SetFromMap<E> extends AbstractSet<E> implements Set<E>, Serializable {

    @Positive
        public void clear();

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
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.SetFromMap<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public String toString();

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object o);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public static <T> Queue<T> asLifoQueue(@PolyGrowShrink @PolyNonEmpty Deque<T> deque);

    @Positive
    static class AsLIFOQueue<E> extends AbstractQueue<E> implements Queue<E>, Serializable {

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        public boolean offer(E e);

    @Positive
        public E poll();

    @Positive
        public E remove();

    @Positive
        @Pure
    @Positive
        public E peek();

    @Positive
        public E element();

    @Positive
        public void clear();

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
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(Collections.AsLIFOQueue<@PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public <T> T[] toArray(IntFunction<T[]> f);

    @Positive
        public String toString();

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @SideEffectFree
    @Positive
        @Override
    @Positive
        public Spliterator<E> spliterator();

    @Positive
        @Override
    @Positive
        public Stream<E> stream();

    @Positive
        @Override
    @Positive
        public Stream<E> parallelStream();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
