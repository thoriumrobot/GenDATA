/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.reflect.ParameterizedType;
    @Positive
import java.lang.reflect.Type;
    @Positive
import java.util.AbstractMap;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.concurrent.atomic.AtomicReference;
    @Positive
import java.util.concurrent.locks.LockSupport;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.DoubleBinaryOperator;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.IntBinaryOperator;
    @Positive
import java.util.function.LongBinaryOperator;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.ToDoubleBiFunction;
    @Positive
import java.util.function.ToDoubleFunction;
    @Positive
import java.util.function.ToIntBiFunction;
    @Positive
import java.util.function.ToIntFunction;
    @Positive
import java.util.function.ToLongBiFunction;
    @Positive
import java.util.function.ToLongFunction;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.misc.Unsafe;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class ConcurrentHashMap<K extends @NonNull Object, V extends @NonNull Object> extends AbstractMap<K, V> implements ConcurrentMap<K, V>, Serializable {

    @Positive
    static class Node<K, V> implements Map.Entry<K, V> {

    @Positive
        public final K getKey();

    @Positive
        public final V getValue();

    @Positive
        public final int hashCode();

    @Positive
        public final String toString();

    @Positive
        public final V setValue(V value);

    @Positive
        public final boolean equals(Object o);

    @Positive
        Node<K, V> find(int h, Object k);
    @Positive
    }

    @Positive
    static final int spread(int h);

    @Positive
    static Class<?> comparableClassFor(Object x);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    static int compareComparables(Class<?> kc, Object k, Object x);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static final <K, V> Node<K, V> tabAt(Node<K, V>[] tab, int i);

    @Positive
    static final <K, V> boolean casTabAt(Node<K, V>[] tab, int i, Node<K, V> c, Node<K, V> v);

    @Positive
    static final <K, V> void setTabAt(Node<K, V>[] tab, int i, Node<K, V> v);

    @Positive
    public ConcurrentHashMap() {
    @Positive
    }

    @Positive
    public ConcurrentHashMap(int initialCapacity) {
    @Positive
    }

    @Positive
    public ConcurrentHashMap(Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    public ConcurrentHashMap(int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public ConcurrentHashMap(int initialCapacity, float loadFactor, int concurrencyLevel) {
    @Positive
    }

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
    @Nullable
    @Positive
    public V get(@UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(K key, V value);

    @Positive
    final V putVal(K key, V value, boolean onlyIfAbsent);

    @Positive
    public void putAll(Map<? extends K, ? extends V> m);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    final V replaceNode(Object key, V value, Object cv);

    @Positive
    public void clear();

    @Positive
    @SideEffectFree
    @Positive
    public KeySetView<@KeyFor({ "this" }) K, V> keySet();

    @Positive
    @SideEffectFree
    @Positive
    public Collection<V> values();

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet();

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    static class Segment<K, V> extends ReentrantLock implements Serializable {
    @Positive
    }

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V putIfAbsent(K key, V value);

    @Positive
    public boolean remove(@GuardSatisfied @UnknownSignedness Object key, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    public boolean replace(K key, V oldValue, V newValue);

    @Positive
    @Nullable
    @Positive
    public V replace(K key, V value);

    @Positive
    @Pure
    @Positive
    public V getOrDefault(@GuardSatisfied @UnknownSignedness Object key, V defaultValue);

    @Positive
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    boolean removeEntryIf(Predicate<? super Entry<K, V>> function);

    @Positive
    boolean removeValueIf(Predicate<? super V> function);

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
    public V compute(K key, BiFunction<? super K, ? super @Nullable V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @PolyNull
    @Positive
    public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> remappingFunction);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @SideEffectFree
    @Positive
    public Enumeration<@KeyFor({ "this" }) K> keys();

    @Positive
    @SideEffectFree
    @Positive
    public Enumeration<V> elements();

    @Positive
    public long mappingCount();

    @Positive
    public static <K> KeySetView<K, Boolean> newKeySet();

    @Positive
    public static <K> KeySetView<K, Boolean> newKeySet(int initialCapacity);

    @Positive
    public KeySetView<K, V> keySet(V mappedValue);

    @Positive
    static final class ForwardingNode<K, V> extends Node<K, V> {

    @Positive
        Node<K, V> find(int h, Object k);
    @Positive
    }

    @Positive
    static final class ReservationNode<K, V> extends Node<K, V> {

    @Positive
        Node<K, V> find(int h, Object k);
    @Positive
    }

    @Positive
    static final int resizeStamp(int n);

    @Positive
    final Node<K, V>[] helpTransfer(Node<K, V>[] tab, Node<K, V> f);

    @Positive
    @jdk.internal.vm.annotation.Contended
    @Positive
    static final class CounterCell {
    @Positive
    }

    @Positive
    final long sumCount();

    @Positive
    static <K, V> Node<K, V> untreeify(Node<K, V> b);

    @Positive
    static final class TreeNode<K, V> extends Node<K, V> {

    @Positive
        Node<K, V> find(int h, Object k);

    @Positive
        final TreeNode<K, V> findTreeNode(int h, Object k, Class<?> kc);
    @Positive
    }

    @Positive
    static final class TreeBin<K, V> extends Node<K, V> {

    @Positive
        static int tieBreakOrder(Object a, Object b);

    @Positive
        final Node<K, V> find(int h, Object k);

    @Positive
        final TreeNode<K, V> putTreeVal(int h, K k, V v);

    @Positive
        final boolean removeTreeNode(TreeNode<K, V> p);

    @Positive
        static <K, V> TreeNode<K, V> rotateLeft(TreeNode<K, V> root, TreeNode<K, V> p);

    @Positive
        static <K, V> TreeNode<K, V> rotateRight(TreeNode<K, V> root, TreeNode<K, V> p);

    @Positive
        static <K, V> TreeNode<K, V> balanceInsertion(TreeNode<K, V> root, TreeNode<K, V> x);

    @Positive
        static <K, V> TreeNode<K, V> balanceDeletion(TreeNode<K, V> root, TreeNode<K, V> x);

    @Positive
        static <K, V> boolean checkInvariants(TreeNode<K, V> t);
    @Positive
    }

    @Positive
    static final class TableStack<K, V> {
    @Positive
    }

    @Positive
    static class Traverser<K, V> {

    @Positive
        final Node<K, V> advance();
    @Positive
    }

    @Positive
    static class BaseIterator<K, V> extends Traverser<K, V> {

    @Positive
        @Pure
    @Positive
        public final boolean hasNext();

    @Positive
        @Pure
    @Positive
        public final boolean hasMoreElements();

    @Positive
        public final void remove();
    @Positive
    }

    @Positive
    static final class KeyIterator<K, V> extends BaseIterator<K, V> implements Iterator<K>, Enumeration<K> {

    @Positive
        public final K next(@NonEmpty KeyIterator<K, V> this);

    @Positive
        public final K nextElement(@NonEmpty KeyIterator<K, V> this);
    @Positive
    }

    @Positive
    static final class ValueIterator<K, V> extends BaseIterator<K, V> implements Iterator<V>, Enumeration<V> {

    @Positive
        public final V next(@NonEmpty ValueIterator<K, V> this);

    @Positive
        public final V nextElement(@NonEmpty ValueIterator<K, V> this);
    @Positive
    }

    @Positive
    static final class EntryIterator<K, V> extends BaseIterator<K, V> implements Iterator<Map.Entry<K, V>> {

    @Positive
        public final Map.Entry<K, V> next(@NonEmpty EntryIterator<K, V> this);
    @Positive
    }

    @Positive
    static final class MapEntry<K, V> implements Map.Entry<K, V> {

    @Positive
        public K getKey();

    @Positive
        public V getValue();

    @Positive
        public int hashCode();

    @Positive
        public String toString();

    @Positive
        public boolean equals(Object o);

    @Positive
        public V setValue(V value);
    @Positive
    }

    @Positive
    static final class KeySpliterator<K, V> extends Traverser<K, V> implements Spliterator<K> {

    @Positive
        public KeySpliterator<K, V> trySplit();

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
    static final class ValueSpliterator<K, V> extends Traverser<K, V> implements Spliterator<V> {

    @Positive
        public ValueSpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super V> action);

    @Positive
        public boolean tryAdvance(Consumer<? super V> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static final class EntrySpliterator<K, V> extends Traverser<K, V> implements Spliterator<Map.Entry<K, V>> {

    @Positive
        public EntrySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public boolean tryAdvance(Consumer<? super Map.Entry<K, V>> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    final int batchFor(long b);

    @Positive
    public void forEach(long parallelismThreshold, BiConsumer<? super K, ? super V> action);

    @Positive
    public <U> void forEach(long parallelismThreshold, BiFunction<? super K, ? super V, ? extends U> transformer, Consumer<? super U> action);

    @Positive
    public <U> U search(long parallelismThreshold, BiFunction<? super K, ? super V, ? extends U> searchFunction);

    @Positive
    public <U> U reduce(long parallelismThreshold, BiFunction<? super K, ? super V, ? extends U> transformer, BiFunction<? super U, ? super U, ? extends U> reducer);

    @Positive
    public double reduceToDouble(long parallelismThreshold, ToDoubleBiFunction<? super K, ? super V> transformer, double basis, DoubleBinaryOperator reducer);

    @Positive
    public long reduceToLong(long parallelismThreshold, ToLongBiFunction<? super K, ? super V> transformer, long basis, LongBinaryOperator reducer);

    @Positive
    public int reduceToInt(long parallelismThreshold, ToIntBiFunction<? super K, ? super V> transformer, int basis, IntBinaryOperator reducer);

    @Positive
    public void forEachKey(long parallelismThreshold, Consumer<? super K> action);

    @Positive
    public <U> void forEachKey(long parallelismThreshold, Function<? super K, ? extends U> transformer, Consumer<? super U> action);

    @Positive
    public <U> U searchKeys(long parallelismThreshold, Function<? super K, ? extends U> searchFunction);

    @Positive
    public K reduceKeys(long parallelismThreshold, BiFunction<? super K, ? super K, ? extends K> reducer);

    @Positive
    public <U> U reduceKeys(long parallelismThreshold, Function<? super K, ? extends U> transformer, BiFunction<? super U, ? super U, ? extends U> reducer);

    @Positive
    public double reduceKeysToDouble(long parallelismThreshold, ToDoubleFunction<? super K> transformer, double basis, DoubleBinaryOperator reducer);

    @Positive
    public long reduceKeysToLong(long parallelismThreshold, ToLongFunction<? super K> transformer, long basis, LongBinaryOperator reducer);

    @Positive
    public int reduceKeysToInt(long parallelismThreshold, ToIntFunction<? super K> transformer, int basis, IntBinaryOperator reducer);

    @Positive
    public void forEachValue(long parallelismThreshold, Consumer<? super V> action);

    @Positive
    public <U> void forEachValue(long parallelismThreshold, Function<? super V, ? extends U> transformer, Consumer<? super U> action);

    @Positive
    public <U> U searchValues(long parallelismThreshold, Function<? super V, ? extends U> searchFunction);

    @Positive
    public V reduceValues(long parallelismThreshold, BiFunction<? super V, ? super V, ? extends V> reducer);

    @Positive
    public <U> U reduceValues(long parallelismThreshold, Function<? super V, ? extends U> transformer, BiFunction<? super U, ? super U, ? extends U> reducer);

    @Positive
    public double reduceValuesToDouble(long parallelismThreshold, ToDoubleFunction<? super V> transformer, double basis, DoubleBinaryOperator reducer);

    @Positive
    public long reduceValuesToLong(long parallelismThreshold, ToLongFunction<? super V> transformer, long basis, LongBinaryOperator reducer);

    @Positive
    public int reduceValuesToInt(long parallelismThreshold, ToIntFunction<? super V> transformer, int basis, IntBinaryOperator reducer);

    @Positive
    public void forEachEntry(long parallelismThreshold, Consumer<? super Map.Entry<K, V>> action);

    @Positive
    public <U> void forEachEntry(long parallelismThreshold, Function<Map.Entry<K, V>, ? extends U> transformer, Consumer<? super U> action);

    @Positive
    public <U> U searchEntries(long parallelismThreshold, Function<Map.Entry<K, V>, ? extends U> searchFunction);

    @Positive
    public Map.Entry<K, V> reduceEntries(long parallelismThreshold, BiFunction<Map.Entry<K, V>, Map.Entry<K, V>, ? extends Map.Entry<K, V>> reducer);

    @Positive
    public <U> U reduceEntries(long parallelismThreshold, Function<Map.Entry<K, V>, ? extends U> transformer, BiFunction<? super U, ? super U, ? extends U> reducer);

    @Positive
    public double reduceEntriesToDouble(long parallelismThreshold, ToDoubleFunction<Map.Entry<K, V>> transformer, double basis, DoubleBinaryOperator reducer);

    @Positive
    public long reduceEntriesToLong(long parallelismThreshold, ToLongFunction<Map.Entry<K, V>> transformer, long basis, LongBinaryOperator reducer);

    @Positive
    public int reduceEntriesToInt(long parallelismThreshold, ToIntFunction<Map.Entry<K, V>> transformer, int basis, IntBinaryOperator reducer);

    @Positive
    abstract static class CollectionView<K, V, E> implements Collection<E>, java.io.Serializable {

    @Positive
        public ConcurrentHashMap<K, V> getMap();

    @Positive
        public final void clear();

    @Positive
        @Pure
    @Positive
        public final int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public final boolean isEmpty();

    @Positive
        @SideEffectFree
    @Positive
        public abstract Iterator<E> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public abstract boolean contains(@UnknownSignedness Object o);

    @Positive
        public abstract boolean remove(@UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public final Object[] toArray(CollectionView<K, V, @PolyNull @PolySigned E> this);

    @Positive
        @SideEffectFree
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public final <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public final String toString();

    @Positive
        @Pure
    @Positive
        public final boolean containsAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        public boolean removeAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        public final boolean retainAll(Collection<? extends @NonNull @UnknownSignedness Object> c);
    @Positive
    }

    @Positive
    public static class KeySetView<K, V> extends CollectionView<K, V, K> implements Set<K>, java.io.Serializable {

    @Positive
        public V getMappedValue();

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
        public Iterator<K> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(K e);

    @Positive
        public boolean addAll(Collection<? extends K> c);

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object o);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<K> spliterator();

    @Positive
        public void forEach(Consumer<? super K> action);
    @Positive
    }

    @Positive
    static final class ValuesView<K, V> extends CollectionView<K, V, V> implements Collection<V>, java.io.Serializable {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean contains(@UnknownSignedness Object o);

    @Positive
        public final boolean remove(@UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        public final Iterator<V> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public final boolean add(V e);

    @Positive
        public final boolean addAll(Collection<? extends V> c);

    @Positive
        @Override
    @Positive
        public boolean removeAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        public boolean removeIf(Predicate<? super V> filter);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<V> spliterator();

    @Positive
        public void forEach(Consumer<? super V> action);
    @Positive
    }

    @Positive
    static final class EntrySetView<K, V> extends CollectionView<K, V, Map.Entry<K, V>> implements Set<Map.Entry<K, V>>, java.io.Serializable {

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
        public Iterator<Map.Entry<K, V>> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(Entry<K, V> e);

    @Positive
        public boolean addAll(Collection<? extends Entry<K, V>> c);

    @Positive
        public boolean removeIf(Predicate<? super Entry<K, V>> filter);

    @Positive
        public final int hashCode();

    @Positive
        public final boolean equals(Object o);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<Map.Entry<K, V>> spliterator();

    @Positive
        public void forEach(Consumer<? super Map.Entry<K, V>> action);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    abstract static class BulkTask<K, V, R> extends CountedCompleter<R> {

    @Positive
        final Node<K, V> advance();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachKeyTask<K, V> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachValueTask<K, V> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachEntryTask<K, V> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachMappingTask<K, V> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachTransformedKeyTask<K, V, U> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachTransformedValueTask<K, V, U> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachTransformedEntryTask<K, V, U> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ForEachTransformedMappingTask<K, V, U> extends BulkTask<K, V, Void> {

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class SearchKeysTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class SearchValuesTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class SearchEntriesTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class SearchMappingsTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ReduceKeysTask<K, V> extends BulkTask<K, V, K> {

    @Positive
        public final K getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ReduceValuesTask<K, V> extends BulkTask<K, V, V> {

    @Positive
        public final V getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class ReduceEntriesTask<K, V> extends BulkTask<K, V, Map.Entry<K, V>> {

    @Positive
        public final Map.Entry<K, V> getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceKeysTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceValuesTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceEntriesTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceMappingsTask<K, V, U> extends BulkTask<K, V, U> {

    @Positive
        public final U getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceKeysToDoubleTask<K, V> extends BulkTask<K, V, Double> {

    @Positive
        public final Double getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceValuesToDoubleTask<K, V> extends BulkTask<K, V, Double> {

    @Positive
        public final Double getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceEntriesToDoubleTask<K, V> extends BulkTask<K, V, Double> {

    @Positive
        public final Double getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceMappingsToDoubleTask<K, V> extends BulkTask<K, V, Double> {

    @Positive
        public final Double getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceKeysToLongTask<K, V> extends BulkTask<K, V, Long> {

    @Positive
        public final Long getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceValuesToLongTask<K, V> extends BulkTask<K, V, Long> {

    @Positive
        public final Long getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceEntriesToLongTask<K, V> extends BulkTask<K, V, Long> {

    @Positive
        public final Long getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceMappingsToLongTask<K, V> extends BulkTask<K, V, Long> {

    @Positive
        public final Long getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceKeysToIntTask<K, V> extends BulkTask<K, V, Integer> {

    @Positive
        public final Integer getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceValuesToIntTask<K, V> extends BulkTask<K, V, Integer> {

    @Positive
        public final Integer getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceEntriesToIntTask<K, V> extends BulkTask<K, V, Integer> {

    @Positive
        public final Integer getRawResult();

    @Positive
        public final void compute();
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static final class MapReduceMappingsToIntTask<K, V> extends BulkTask<K, V, Integer> {

    @Positive
        public final Integer getRawResult();

    @Positive
        public final void compute();
    @Positive
    }
    @Positive
}
