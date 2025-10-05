/*
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
package java.util.concurrent;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
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
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.AbstractCollection;
    @Positive
import java.util.AbstractMap;
    @Positive
import java.util.AbstractSet;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.NavigableSet;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedMap;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.concurrent.atomic.LongAdder;

    @Positive
public class ConcurrentSkipListMap<K, V> extends AbstractMap<K, V> implements ConcurrentNavigableMap<K, V>, Cloneable, Serializable {

    @Positive
    static final class Node<K, V> {
    @Positive
    }

    @Positive
    static final class Index<K, V> {
    @Positive
    }

    @Positive
    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
    static int cpr(Comparator c, Object x, Object y);

    @Positive
    final Node<K, V> baseHead();

    @Positive
    static <K, V> void unlinkNode(Node<K, V> b, Node<K, V> n);

    @Positive
    final long getAdderCount();

    @Positive
    static <K, V> boolean addIndices(Index<K, V> q, int skips, Index<K, V> x, Comparator<? super K> cmp);

    @Positive
    final V doRemove(Object key, Object value);

    @Positive
    final Node<K, V> findFirst();

    @Positive
    final AbstractMap.SimpleImmutableEntry<K, V> findFirstEntry();

    @Positive
    final Node<K, V> findLast();

    @Positive
    final AbstractMap.SimpleImmutableEntry<K, V> findLastEntry();

    @Positive
    final Node<K, V> findNear(K key, int rel, Comparator<? super K> cmp);

    @Positive
    final AbstractMap.SimpleImmutableEntry<K, V> findNearEntry(K key, int rel, Comparator<? super K> cmp);

    @Positive
    public ConcurrentSkipListMap() {
    @Positive
    }

    @Positive
    public ConcurrentSkipListMap(Comparator<? super K> comparator) {
    @Positive
    }

    @Positive
    public ConcurrentSkipListMap(Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    public ConcurrentSkipListMap(SortedMap<K, ? extends V> m) {
    @Positive
    }

    @Positive
    public ConcurrentSkipListMap<K, V> clone();

    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    public V get(Object key);

    @Positive
    @Pure
    @Positive
    public V getOrDefault(@GuardSatisfied @UnknownSignedness Object key, V defaultValue);

    @Positive
    public V put(K key, V value);

    @Positive
    public V remove(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied @UnknownSignedness Object value);

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
    public void clear();

    @Positive
    @PolyNull
    @Positive
    public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mappingFunction);

    @Positive
    @PolyNull
    @Positive
    public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @PolyNull
    @Positive
    public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @PolyNull
    @Positive
    public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    public NavigableSet<K> keySet();

    @Positive
    @SideEffectFree
    @Positive
    public NavigableSet<K> navigableKeySet();

    @Positive
    public Collection<V> values();

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<K, V>> entrySet();

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> descendingMap();

    @Positive
    @SideEffectFree
    @Positive
    public NavigableSet<K> descendingKeySet();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public V putIfAbsent(K key, V value);

    @Positive
    public boolean remove(@GuardSatisfied @UnknownSignedness Object key, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    public boolean replace(K key, V oldValue, V newValue);

    @Positive
    public V replace(K key, V value);

    @Positive
    public Comparator<? super K> comparator();

    @Positive
    public K firstKey(@NonEmpty ConcurrentSkipListMap<K, V> this);

    @Positive
    public K lastKey(@NonEmpty ConcurrentSkipListMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> subMap(K fromKey, K toKey);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> headMap(K toKey);

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentNavigableMap<K, V> tailMap(K fromKey);

    @Positive
    public Map.Entry<K, V> lowerEntry(K key);

    @Positive
    public K lowerKey(K key);

    @Positive
    public Map.Entry<K, V> floorEntry(K key);

    @Positive
    public K floorKey(K key);

    @Positive
    public Map.Entry<K, V> ceilingEntry(K key);

    @Positive
    public K ceilingKey(K key);

    @Positive
    public Map.Entry<K, V> higherEntry(K key);

    @Positive
    public K higherKey(K key);

    @Positive
    public Map.Entry<K, V> firstEntry();

    @Positive
    public Map.Entry<K, V> lastEntry();

    @Positive
    public Map.Entry<K, V> pollFirstEntry();

    @Positive
    public Map.Entry<K, V> pollLastEntry();

    @Positive
    abstract class Iter<T> implements Iterator<T> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        final void advance(Node<K, V> b);

    @Positive
        public final void remove();
    @Positive
    }

    @Positive
    final class ValueIterator extends Iter<V> {

    @Positive
        public V next(@NonEmpty ValueIterator this);
    @Positive
    }

    @Positive
    final class KeyIterator extends Iter<K> {

    @Positive
        public K next(@NonEmpty KeyIterator this);
    @Positive
    }

    @Positive
    final class EntryIterator extends Iter<Map.Entry<K, V>> {

    @Positive
        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);
    @Positive
    }

    @Positive
    static final <E> List<E> toList(Collection<E> c);

    @Positive
    static final class KeySet<K, V> extends AbstractSet<K> implements NavigableSet<K> {

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
        public void clear();

    @Positive
        public K lower(K e);

    @Positive
        public K floor(K e);

    @Positive
        public K ceiling(K e);

    @Positive
        public K higher(K e);

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        public K first();

    @Positive
        public K last();

    @Positive
        public K pollFirst();

    @Positive
        public K pollLast();

    @Positive
        public Iterator<K> iterator();

    @Positive
        public boolean equals(Object o);

    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(KeySet<@PolyNull @PolySigned K, V> this);

    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public Iterator<K> descendingIterator();

    @Positive
        public NavigableSet<K> subSet(K fromElement, boolean fromInclusive, K toElement, boolean toInclusive);

    @Positive
        public NavigableSet<K> headSet(K toElement, boolean inclusive);

    @Positive
        public NavigableSet<K> tailSet(K fromElement, boolean inclusive);

    @Positive
        public NavigableSet<K> subSet(K fromElement, K toElement);

    @Positive
        public NavigableSet<K> headSet(K toElement);

    @Positive
        public NavigableSet<K> tailSet(K fromElement);

    @Positive
        public NavigableSet<K> descendingSet();

    @Positive
        public Spliterator<K> spliterator();
    @Positive
    }

    @Positive
    static final class Values<K, V> extends AbstractCollection<V> {

    @Positive
        public Iterator<V> iterator();

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
        public void clear();

    @Positive
        public Object[] toArray();

    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public Spliterator<V> spliterator();

    @Positive
        public boolean removeIf(Predicate<? super V> filter);
    @Positive
    }

    @Positive
    static final class EntrySet<K, V> extends AbstractSet<Map.Entry<K, V>> {

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
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        public void clear();

    @Positive
        public boolean equals(Object o);

    @Positive
        public Object[] toArray();

    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public Spliterator<Map.Entry<K, V>> spliterator();

    @Positive
        public boolean removeIf(Predicate<? super Entry<K, V>> filter);
    @Positive
    }

    @Positive
    static final class SubMap<K, V> extends AbstractMap<K, V> implements ConcurrentNavigableMap<K, V>, Serializable {

    @Positive
        boolean tooLow(Object key, Comparator<? super K> cmp);

    @Positive
        boolean tooHigh(Object key, Comparator<? super K> cmp);

    @Positive
        boolean inBounds(Object key, Comparator<? super K> cmp);

    @Positive
        void checkKeyBounds(K key, Comparator<? super K> cmp);

    @Positive
        boolean isBeforeEnd(ConcurrentSkipListMap.Node<K, V> n, Comparator<? super K> cmp);

    @Positive
        ConcurrentSkipListMap.Node<K, V> loNode(Comparator<? super K> cmp);

    @Positive
        ConcurrentSkipListMap.Node<K, V> hiNode(Comparator<? super K> cmp);

    @Positive
        K lowestKey(@NonEmpty SubMap<K, V> this);

    @Positive
        K highestKey(@NonEmpty SubMap<K, V> this);

    @Positive
        Map.Entry<K, V> lowestEntry();

    @Positive
        Map.Entry<K, V> highestEntry();

    @Positive
        Map.Entry<K, V> removeLowest();

    @Positive
        Map.Entry<K, V> removeHighest();

    @Positive
        Map.Entry<K, V> getNearEntry(K key, int rel);

    @Positive
        K getNearKey(K key, int rel);

    @Positive
        @Pure
    @Positive
        public boolean containsKey(@UnknownSignedness Object key);

    @Positive
        public V get(Object key);

    @Positive
        public V put(K key, V value);

    @Positive
        public V remove(Object key);

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
        public boolean containsValue(@UnknownSignedness Object value);

    @Positive
        public void clear();

    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

    @Positive
        public boolean replace(K key, V oldValue, V newValue);

    @Positive
        public V replace(K key, V value);

    @Positive
        public Comparator<? super K> comparator();

    @Positive
        SubMap<K, V> newSubMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> headMap(K toKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> tailMap(K fromKey, boolean inclusive);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> subMap(K fromKey, K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> headMap(K toKey);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> tailMap(K fromKey);

    @Positive
        @SideEffectFree
    @Positive
        public SubMap<K, V> descendingMap();

    @Positive
        public Map.Entry<K, V> ceilingEntry(K key);

    @Positive
        public K ceilingKey(K key);

    @Positive
        public Map.Entry<K, V> lowerEntry(K key);

    @Positive
        public K lowerKey(K key);

    @Positive
        public Map.Entry<K, V> floorEntry(K key);

    @Positive
        public K floorKey(K key);

    @Positive
        public Map.Entry<K, V> higherEntry(K key);

    @Positive
        public K higherKey(K key);

    @Positive
        public K firstKey();

    @Positive
        public K lastKey();

    @Positive
        public Map.Entry<K, V> firstEntry();

    @Positive
        public Map.Entry<K, V> lastEntry();

    @Positive
        public Map.Entry<K, V> pollFirstEntry();

    @Positive
        public Map.Entry<K, V> pollLastEntry();

    @Positive
        public NavigableSet<K> keySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> navigableKeySet();

    @Positive
        public Collection<V> values();

    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        @SideEffectFree
    @Positive
        public NavigableSet<K> descendingKeySet();

    @Positive
        abstract class SubMapIter<T> implements Iterator<T>, Spliterator<T> {

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public final boolean hasNext();

    @Positive
            final void advance(@NonEmpty SubMapIter<T> this);

    @Positive
            public void remove();

    @Positive
            public Spliterator<T> trySplit();

    @Positive
            public boolean tryAdvance(Consumer<? super T> action);

    @Positive
            public void forEachRemaining(Consumer<? super T> action);

    @Positive
            public long estimateSize();
    @Positive
        }

    @Positive
        final class SubMapValueIterator extends SubMapIter<V> {

    @Positive
            public V next(@NonEmpty SubMapValueIterator this);

    @Positive
            public int characteristics();
    @Positive
        }

    @Positive
        final class SubMapKeyIterator extends SubMapIter<K> {

    @Positive
            public K next(@NonEmpty SubMapKeyIterator this);

    @Positive
            public int characteristics();

    @Positive
            public final Comparator<? super K> getComparator();
    @Positive
        }

    @Positive
        final class SubMapEntryIterator extends SubMapIter<Map.Entry<K, V>> {

    @Positive
            public Map.Entry<K, V> next(@NonEmpty SubMapEntryIterator this);

    @Positive
            public int characteristics();
    @Positive
        }
    @Positive
    }

    @Positive
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    boolean removeEntryIf(Predicate<? super Entry<K, V>> function);

    @Positive
    boolean removeValueIf(Predicate<? super V> function);

    @Positive
    abstract static class CSLMSpliterator<K, V> {

    @Positive
        public final long estimateSize();
    @Positive
    }

    @Positive
    static final class KeySpliterator<K, V> extends CSLMSpliterator<K, V> implements Spliterator<K> {

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
    final KeySpliterator<K, V> keySpliterator();

    @Positive
    static final class ValueSpliterator<K, V> extends CSLMSpliterator<K, V> implements Spliterator<V> {

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
    final ValueSpliterator<K, V> valueSpliterator();

    @Positive
    static final class EntrySpliterator<K, V> extends CSLMSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

    @Positive
        public EntrySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public int characteristics();

    @Positive
        public final Comparator<Map.Entry<K, V>> getComparator();
    @Positive
    }

    @Positive
    final EntrySpliterator<K, V> entrySpliterator();
    @Positive
}

// CFWR semantic augmentation - variant 1
