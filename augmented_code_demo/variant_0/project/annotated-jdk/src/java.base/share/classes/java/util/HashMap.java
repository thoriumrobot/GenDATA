/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.reflect.ParameterizedType;
    @Positive
import java.lang.reflect.Type;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class HashMap<K, V> extends AbstractMap<K, V> implements Map<K, V>, Cloneable, Serializable {

    @Positive
    static class Node<K, V> implements Map.Entry<K, V> {

    @Positive
        public final K getKey();

    @Positive
        public final V getValue();

    @Positive
        public final String toString();

    @Positive
        public final int hashCode();

    @Positive
        public final V setValue(V newValue);

    @Positive
        public final boolean equals(Object o);
    @Positive
    }

    @Positive
    static final int hash(@Nullable Object key);

    @Positive
    static Class<?> comparableClassFor(Object x);

    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    static int compareComparables(Class<?> kc, Object k, Object x);

    @Positive
    static final int tableSizeFor(int cap);

    @Positive
    public HashMap(@NonNegative int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public HashMap(@NonNegative int initialCapacity) {
    @Positive
    }

    @Positive
    public HashMap() {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public HashMap(@PolyNonEmpty Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    final void putMapEntries(Map<? extends K, ? extends V> m, boolean evict);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied HashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied HashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public V get(@GuardSatisfied HashMap<K, V> this, @UnknownSignedness @GuardSatisfied @Nullable Object key);

    @Positive
    final Node<K, V> getNode(@Nullable Object key);

    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied HashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(@GuardSatisfied HashMap<K, V> this, K key, V value);

    @Positive
    final V putVal(int hash, K key, V value, boolean onlyIfAbsent, boolean evict);

    @Positive
    @SuppressWarnings("cast.unsafe")
    @Positive
    final Node<K, V>[] resize();

    @Positive
    final void treeifyBin(Node<K, V>[] tab, int hash);

    @Positive
    public void putAll(@GuardSatisfied HashMap<K, V> this, Map<? extends K, ? extends V> m);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied HashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object key);

    @Positive
    final Node<K, V> removeNode(int hash, @Nullable Object key, @Nullable Object value, boolean matchValue, boolean movable);

    @Positive
    public void clear(@GuardSatisfied HashMap<K, V> this);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied HashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied HashMap<K, V> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    final <T> T[] prepareArray(T[] a);

    @Positive
    <T> T[] keysToArray(T[] a);

    @Positive
    <T> T[] valuesToArray(T[] a);

    @Positive
    final class KeySet extends AbstractSet<K> {

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public final int size();

    @Positive
        public final void clear();

    @Positive
        @SideEffectFree
    @Positive
        public final Iterator<K> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public final boolean remove(@Nullable @UnknownSignedness Object key);

    @Positive
        @SideEffectFree
    @Positive
        public final Spliterator<K> spliterator();

    @Positive
        public Object[] toArray();

    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public final void forEach(Consumer<? super K> action);
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Collection<V> values(@GuardSatisfied HashMap<K, V> this);

    @Positive
    final class Values extends AbstractCollection<V> {

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public final int size();

    @Positive
        public final void clear();

    @Positive
        @SideEffectFree
    @Positive
        public final Iterator<V> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        public final Spliterator<V> spliterator();

    @Positive
        public Object[] toArray();

    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public final void forEach(Consumer<? super V> action);
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied HashMap<K, V> this);

    @Positive
    final class EntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public final int size();

    @Positive
        public final void clear();

    @Positive
        @SideEffectFree
    @Positive
        public final Iterator<Map.Entry<K, V>> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public final boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        public final Spliterator<Map.Entry<K, V>> spliterator();

    @Positive
        public final void forEach(Consumer<? super Map.Entry<K, V>> action);
    @Positive
    }

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public V getOrDefault(@GuardSatisfied @Nullable @UnknownSignedness Object key, V defaultValue);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public V putIfAbsent(K key, V value);

    @Positive
    @Override
    @Positive
    public boolean remove(@GuardSatisfied @Nullable @UnknownSignedness Object key, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    public boolean replace(K key, V oldValue, V newValue);

    @Positive
    @Override
    @Positive
    @Nullable
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
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    @Override
    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public Object clone(@GuardSatisfied HashMap<K, V> this);

    @Positive
    final float loadFactor();

    @Positive
    final int capacity();

    @Positive
    abstract class HashIterator {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        final Node<K, V> nextNode(@NonEmpty HashIterator this);

    @Positive
        public final void remove();
    @Positive
    }

    @Positive
    final class KeyIterator extends HashIterator implements Iterator<K> {

    @Positive
        public final K next(@NonEmpty KeyIterator this);
    @Positive
    }

    @Positive
    final class ValueIterator extends HashIterator implements Iterator<V> {

    @Positive
        public final V next(@NonEmpty ValueIterator this);
    @Positive
    }

    @Positive
    final class EntryIterator extends HashIterator implements Iterator<Map.Entry<K, V>> {

    @Positive
        public final Map.Entry<K, V> next(@NonEmpty EntryIterator this);
    @Positive
    }

    @Positive
    static class HashMapSpliterator<K, V> {

    @Positive
        final int getFence();

    @Positive
        public final long estimateSize();
    @Positive
    }

    @Positive
    static final class KeySpliterator<K, V> extends HashMapSpliterator<K, V> implements Spliterator<K> {

    @Positive
        public KeySpliterator<K, V> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super K> action);

    @Positive
        public boolean tryAdvance(Consumer<? super K> action);

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    static final class ValueSpliterator<K, V> extends HashMapSpliterator<K, V> implements Spliterator<V> {

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
    static final class EntrySpliterator<K, V> extends HashMapSpliterator<K, V> implements Spliterator<Map.Entry<K, V>> {

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
    Node<K, V> newNode(int hash, K key, V value, Node<K, V> next);

    @Positive
    Node<K, V> replacementNode(Node<K, V> p, Node<K, V> next);

    @Positive
    TreeNode<K, V> newTreeNode(int hash, K key, V value, Node<K, V> next);

    @Positive
    TreeNode<K, V> replacementTreeNode(Node<K, V> p, Node<K, V> next);

    @Positive
    void reinitialize();

    @Positive
    void afterNodeAccess(Node<K, V> p);

    @Positive
    void afterNodeInsertion(boolean evict);

    @Positive
    void afterNodeRemoval(Node<K, V> p);

    @Positive
    void internalWriteEntries(java.io.ObjectOutputStream s) throws IOException;

    @Positive
    static final class TreeNode<K, V> extends LinkedHashMap.Entry<K, V> {

    @Positive
        final TreeNode<K, V> root();

    @Positive
        static <K, V> void moveRootToFront(Node<K, V>[] tab, TreeNode<K, V> root);

    @Positive
        final TreeNode<K, V> find(int h, Object k, Class<?> kc);

    @Positive
        final TreeNode<K, V> getTreeNode(int h, Object k);

    @Positive
        static int tieBreakOrder(Object a, Object b);

    @Positive
        final void treeify(Node<K, V>[] tab);

    @Positive
        final Node<K, V> untreeify(HashMap<K, V> map);

    @Positive
        final TreeNode<K, V> putTreeVal(HashMap<K, V> map, Node<K, V>[] tab, int h, K k, V v);

    @Positive
        final void removeTreeNode(HashMap<K, V> map, Node<K, V>[] tab, boolean movable);

    @Positive
        final void split(HashMap<K, V> map, Node<K, V>[] tab, int index, int bit);

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
}

// CFWR semantic augmentation - variant 0
