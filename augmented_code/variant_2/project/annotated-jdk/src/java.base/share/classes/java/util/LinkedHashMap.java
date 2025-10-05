/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
import java.util.function.Consumer;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.io.IOException;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class LinkedHashMap<K, V> extends HashMap<K, V> implements Map<K, V> {

    @Positive
    static class Entry<K, V> extends HashMap.Node<K, V> {
    @Positive
    }

    @Positive
    void reinitialize();

    @Positive
    Node<K, V> newNode(int hash, K key, V value, Node<K, V> e);

    @Positive
    Node<K, V> replacementNode(Node<K, V> p, Node<K, V> next);

    @Positive
    TreeNode<K, V> newTreeNode(int hash, K key, V value, Node<K, V> next);

    @Positive
    TreeNode<K, V> replacementTreeNode(Node<K, V> p, Node<K, V> next);

    @Positive
    void afterNodeRemoval(Node<K, V> e);

    @Positive
    void afterNodeInsertion(boolean evict);

    @Positive
    void afterNodeAccess(Node<K, V> e);

    @Positive
    void internalWriteEntries(java.io.ObjectOutputStream s) throws IOException;

    @Positive
    public LinkedHashMap(@NonNegative int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public LinkedHashMap(@NonNegative int initialCapacity) {
    @Positive
    }

    @Positive
    public LinkedHashMap() {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public LinkedHashMap(@PolyNonEmpty Map<? extends K, ? extends V> m) {
    @Positive
    }

    @Positive
    public LinkedHashMap(@NonNegative int initialCapacity, float loadFactor, boolean accessOrder) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied LinkedHashMap<K, V> this, @GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public V get(@GuardSatisfied LinkedHashMap<K, V> this, @UnknownSignedness @GuardSatisfied @Nullable Object key);

    @Positive
    @Pure
    @Positive
    public V getOrDefault(@Nullable Object key, V defaultValue);

    @Positive
    public void clear(@GuardSatisfied LinkedHashMap<K, V> this);

    @Positive
    protected boolean removeEldestEntry(@GuardSatisfied LinkedHashMap<K, V> this, Map.Entry<K, V> eldest);

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor({ "this" }) K> keySet();

    @Positive
    @Override
    @Positive
    final <T> T[] keysToArray(T[] a);

    @Positive
    @Override
    @Positive
    final <T> T[] valuesToArray(T[] a);

    @Positive
    final class LinkedKeySet extends AbstractSet<K> {

    @Positive
        @Pure
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
    public Collection<V> values();

    @Positive
    final class LinkedValues extends AbstractCollection<V> {

    @Positive
        @Pure
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
    public Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied LinkedHashMap<K, V> this);

    @Positive
    final class LinkedEntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @Pure
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
    public void forEach(BiConsumer<? super K, ? super V> action);

    @Positive
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function);

    @Positive
    abstract class LinkedHashIterator {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        final LinkedHashMap.Entry<K, V> nextNode(@NonEmpty LinkedHashIterator this);

    @Positive
        public final void remove();
    @Positive
    }

    @Positive
    final class LinkedKeyIterator extends LinkedHashIterator implements Iterator<K> {

    @Positive
        public final K next(@NonEmpty LinkedKeyIterator this);
    @Positive
    }

    @Positive
    final class LinkedValueIterator extends LinkedHashIterator implements Iterator<V> {

    @Positive
        public final V next(@NonEmpty LinkedValueIterator this);
    @Positive
    }

    @Positive
    final class LinkedEntryIterator extends LinkedHashIterator implements Iterator<Map.Entry<K, V>> {

    @Positive
        public final Map.Entry<K, V> next(@NonEmpty LinkedEntryIterator this);
    @Positive
    }
    @Positive
}
